"""
This class implements the 3 compulsory mehtods
a submission is required to have namely __init__, train, test
The main idea is to have a module.py for each phase
model selection -> transformation selection -> training -> testing
ie.
selection.py -> transformations.py -> training.py -> testing.py

It is possible to have multiple function/classes in these modules
and which one to use and it's arguments can be defined in the config.hjson
This allow for rapid comparison between two or more competing methods

NOTE:
A hjson is a normal json but allows comments, so no black magic here.
"""
import json
import os
import threading
import time
import types
from collections import OrderedDict
from functools import partial

# Import the challenge algorithm (model) API from algorithm.py
import algorithm
import numpy as np
import selection
import tensorflow as tf
import testing
import torch
import torch.cuda as cutorch
import training
import transformations
import utils
from torch_adapter import TFDataLoader, TFDataset
from utils import BASEDIR, DEVICE, LOGGER

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.logging.set_verbosity(tf.logging.ERROR)

# If apex's amp is available, import it and set a flag to use it
try:
    from apex import amp

    # Make the use of amp's scaled loss seemless to the training so
    # loss.backward() performs scaled_loss.backward() and training
    # doesn't need to pay attention
    def amp_loss(predictions, labels, loss_fn, optimizer):
        loss = loss_fn(predictions, labels)
        if hasattr(optimizer, '_amp_stash'):
            loss.backward = partial(scaled_loss_helper, loss=loss, optimizer=optimizer)
        return loss

    def scaled_loss_helper(loss, optimizer):
        with amp.scale_loss(loss, optimizer) as scale_loss:
            scale_loss.backward()

    USE_AMP = True
except Exception:
    USE_AMP = False
    pass

THREADS = [
    threading.Thread(target=lambda: torch.cuda.synchronize()),
    threading.Thread(target=lambda: tf.Session())
]
[t.start() for t in THREADS]


class Model(algorithm.Algorithm):
    def __init__(self, metadata):
        self.birthday = time.time()
        self.config = utils.Config(os.path.join(BASEDIR, "config.hjson"))
        bohb_conf_path = os.path.join(BASEDIR, 'bohb_config.json')
        if os.path.isfile(bohb_conf_path):
            self.side_load_config(bohb_conf_path)

        LOGGER.info("INIT START: " + str(time.time()))
        super(Model, self).__init__(metadata)
        # This flag allows you to enable the inbuilt cudnn auto-tuner to find the best
        # algorithm to use for your hardware. Benchmark mode is good whenever your input sizes
        # for your network do not vary
        # https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do/5936
        torch.backends.cudnn.benchmark = self.config.cudnn_benchmark
        torch.backends.cudnn.deterministic = self.config.cudnn_deterministic

        # Seeds
        tf.random.set_random_seed(self.config.tf_seed)
        np.random.seed(self.config.np_random_seed)
        torch.manual_seed(self.config.torch_manual_seed)
        torch.cuda.manual_seed_all(self.config.torch_cuda_manual_seed_all)
        # In-/Out Dimensions from the train dataset's metadata
        row_count, col_count = metadata.get_matrix_size(0)
        channel = metadata.get_num_channels(0)
        sequence_size = metadata.get_sequence_size()
        self.input_dim = [sequence_size, row_count, col_count, channel]
        self.output_dim = metadata.get_output_size()
        self.num_train_samples = metadata.size()
        self.num_test_samples = None
        self.metadata = metadata

        # Store the current dataset's path, loader and the currently used
        # model, optimizer and lossfunction
        self._tf_train_set = None
        self._tf_test_set = None
        self.train_dl = None
        self.test_dl = None
        self.transforms = None
        self.model = None
        self.optimizer = None
        self.lr_scheduler = None
        self.loss_fn = None

        # Set the algorithms to use from the config file
        self.select_model = getattr(selection, self.config.model_selector)
        self.select_transformations = getattr(
            transformations, self.config.transformations_selector
        )
        self.trainer = getattr(training, self.config.trainer)
        trainer_args = self.config.trainer_args[self.config.trainer]
        self.trainer = self.trainer if isinstance(self.trainer,
                                                  types.FunctionType) else self.trainer(
                                                      **trainer_args
                                                  )
        self.tester = getattr(testing, self.config.tester)
        tester_args = self.config.tester_args[self.config.tester]
        self.tester = self.tester if isinstance(self.tester,
                                                types.FunctionType) else self.tester(
                                                    **tester_args
                                                )

        # Attributes for managing time budget
        # Cumulated number of training steps
        self.starting_budget = None
        self.current_remaining_time = None
        self.make_final_prediction = False
        self.final_prediction_made = False
        self.done_training = False

        self.training_round = 0  # keep track how often we entered train
        self.train_time = []
        self.testing_round = 0  # keep track how often we entered test
        self.test_time = []
        self._session = None
        LOGGER.info("INIT END: {}".format(time.time()))

    @property
    def session(self):
        if self._session is None:
            [t.join() for t in THREADS]
            self._session = tf.Session()
        return self._session

    def side_load_config(self, bohb_conf_path):
        '''
        This overrides all setting config defined in the bohb_config
        with the only requirement that their type must match

        ATTENTION: If bohb_config.json can define dictionaries as values as well
        so it is possible to overwrite a whole subhirachy.
        '''
        with open(bohb_conf_path, 'r') as file:
            bohb_conf = json.load(file)

        dicts = [self.config.__dict__]
        while len(dicts) > 0:
            d = dicts.pop()
            for k, v in d.items():
                if k in bohb_conf.keys() and isinstance(bohb_conf[k], type(d[k])):
                    d[k] = bohb_conf[k]
                elif isinstance(v, OrderedDict):
                    dicts.append(v)

    def setup_model(self, ds_temp):
        selected = self.select_model(
            self, ds_temp, **self.config.model_selector_args[self.config.model_selector]
        )
        self.model, self.loss_fn, self.optimizer, self.lr_scheduler = selected

    def setup_transforms(self, ds_temp):
        self.transforms = self.select_transformations(
            self, ds_temp, **self.config.transformations_selector_args[
                self.config.transformations_selector]
        )

    def split_dataset(self, ds_temp, transform):
        # [train_percent, validation_percent, ...]
        split_percentages = (
            self.config.dataset_split_ratio / np.sum(self.config.dataset_split_ratio)
        )
        split_num = np.round((self.num_train_samples * split_percentages))

        # For now use fermi approx to give n classes a chance of
        # (1/n)^(3*n) to not be included at all in the validation set
        # at least
        if split_num[1] < ds_temp.num_classes * 3 and split_num[1] > 0.:
            LOGGER.warning(
                'Validation split too small to be representative: {0} < {1}'.format(
                    split_num[1], ds_temp.num_classes * 3
                )
            )
            split_percentages = np.array(
                (
                    self.num_train_samples - (ds_temp.num_classes * 5),
                    (ds_temp.num_classes * 5)
                )
            )
        assert (sum(split_num) == self.num_train_samples)

        tfdataset = ds_temp.dataset
        tfdataset.shuffle(self.num_train_samples)
        tfdataset_remaining = tfdataset

        tfdataset_train = tfdataset_remaining.take(split_num[0])
        tfdataset_remaining = tfdataset_remaining.skip(split_num[0])

        tfdataset_val = tfdataset_remaining.take(split_num[1])
        tfdataset_remaining = tfdataset_remaining.skip(split_num[1])

        ds_train = TFDataset(
            session=self.session,
            dataset=tfdataset_train,
            num_samples=int(split_num[0]),
            transform_sample=transform['samples'],
            transform_label=transform['labels']
        )
        ds_train.min_shape = ds_temp.min_shape
        ds_train.median_shape = ds_temp.median_shape
        ds_train.mean_shape = ds_temp.mean_shape
        ds_train.std_shape = ds_temp.std_shape
        ds_train.max_shape = ds_temp.max_shape
        ds_train.is_multilabel = ds_temp.is_multilabel

        if split_num[1] > 0.:
            ds_val = TFDataset(
                session=self.session,
                dataset=tfdataset_val,
                num_samples=int(split_num[1]),
                transform_sample=transform['samples'],
                transform_label=transform['labels']
            )
            ds_val.min_shape = ds_temp.min_shape
            ds_val.median_shape = ds_temp.median_shape
            ds_val.mean_shape = ds_temp.mean_shape
            ds_val.std_shape = ds_temp.std_shape
            ds_val.max_shape = ds_temp.max_shape
            ds_val.is_multilabel = ds_temp.is_multilabel

        self.train_dl = {
            'train':
                TFDataLoader(ds_train, **self.config.dataloader_args['train']),
            'val':
                TFDataLoader(ds_val, **self.config.dataloader_args['train'])
                if split_num[1] > 0. else None
        }

        if self.config.check_for_shuffling:
            LOGGER.debug('SANITY CHECKING SHUFFLING')
            ds_train = TFDataset(
                session=self.session,
                dataset=tfdataset_train,
                num_samples=int(split_num[0])
            )
            ds_train.reset()
            ds_temp.reset()
            dset1 = [e for e in ds_train]
            dset2 = [e for e in ds_train]
            dset3 = [e for e in ds_temp][:int(split_num[0])]
            i = 0
            e1vse2 = []
            e2vse3 = []
            for e1, e2, e3 in zip(dset1, dset2, dset3):
                if i % 100 == 0:
                    LOGGER.debug('Checking i: {}'.format(i))
                e1vse2.append((np.all((e1[0] == e2[0]))))
                e2vse3.append((np.all((e2[0] == e3[0]))))
                i += 1
            LOGGER.debug('E1 == E2: {}\t should be False'.format(np.all(e1vse2)))
            LOGGER.debug('E2 == E3: {}\t should be False'.format(np.all(e2vse3)))

    def setup_train_dataset(self, dataset, ds_temp=None):
        self.split_dataset(ds_temp, self.transforms['train'])
        if self.config.benchmark_transformations:
            self.train_dl['train'].dataset.benchmark_transofrmations()

    def train(self, dataset, remaining_time_budget=None):
        train_start = time.time()
        self.current_remaining_time = remaining_time_budget
        LOGGER.info("REMAINING TIME: {0:.2f}".format(remaining_time_budget))
        if self.final_prediction_made:
            return
        dataset_changed = self._tf_train_set != dataset
        if dataset_changed:
            t_s = time.time()
            self.starting_budget = t_s - self.birthday + remaining_time_budget
            self._tf_train_set = dataset
            # Create a temporary handle to inspect data
            dataset = dataset.prefetch(self.config.dataloader_args['train']['batch_size'])
            ds_temp = TFDataset(self.session, dataset, self.num_train_samples)
            ds_temp.scan_all(10)
            LOGGER.info(
                'SETUP MODEL PREPERATION TOOK: {0:.4f} s'.format(time.time() - t_s)
            )

            t_s = time.time()
            self.setup_model(ds_temp)
            if self.model is None:
                return None
            LOGGER.info('SETTING UP THE MODEL TOOK: {0:.4f} s'.format(time.time() - t_s))
            LOGGER.info(
                'FROZEN LAYERS: {}/{}'.format(
                    len([x for x in self.model.parameters() if not x.requires_grad]),
                    len([x for x in self.model.parameters()])
                )
            )

            t_s = time.time()
            self.setup_transforms(ds_temp)
            LOGGER.info(
                'ADDING TRANSFORMATIONS TOOK: {0:.4f} s'.format(time.time() - t_s)
            )
            t_s = time.time()
            self.setup_train_dataset(dataset, ds_temp)
            LOGGER.info(
                'SETTING UP THE TRAINSET TOOK: {0:.4f} s'.format(time.time() - t_s)
            )

            # Finally move the model to gpu
            self.model.to(DEVICE)
            self.loss_fn.to(DEVICE)
            if (
                USE_AMP and self.config.use_amp and
                hasattr(self.model, 'amp_compatible') and self.model.amp_compatible
            ):
                self.model, self.optimizer = amp.initialize(
                    self.model, self.optimizer, **self.config.amp_args
                )
                self.loss_fn = partial(
                    amp_loss, loss_fn=self.loss_fn, optimizer=self.optimizer
                )

        LOGGER.info('BATCH SIZE:\t\t{}'.format(self.train_dl['train'].batch_size))
        train_start = time.time()
        train_args = self.config.trainer_args[
            self.config.trainer] if isinstance(self.trainer, types.FunctionType) else {}
        self.trainer(self, remaining_time_budget, **train_args)

        LOGGER.info("TRAINING TOOK: {0:.6g}".format(time.time() - train_start))
        self.training_round += 1
        self.train_time.append(time.time() - train_start)

    def setup_test_dataset(self, dataset, ds_temp=None):
        ds = TFDataset(
            session=self.session,
            dataset=dataset,
            num_samples=int(1e6),
            transform_sample=self.transforms['test']['samples'],
            transform_label=self.transforms['test']['labels']
        )
        loader_args = self.config.dataloader_args['test']
        self.test_dl = TFDataLoader(ds, **loader_args)

    def test(self, dataset, remaining_time_budget=None):
        test_start = time.time()
        self.current_remaining_time = remaining_time_budget
        LOGGER.info("REMAINING TIME: " + str(remaining_time_budget))
        if self.config.benchmark_time_till_first_prediction:
            LOGGER.error(
                'TIME TILL FIRST PREDICTION: {0}'.format(time.time() - self.birthday)
            )
            LOGGER.error('TRAIN ERR SHAPE: {0}'.format(self.trainer.train_err.shape))
            return None
        if self.final_prediction_made:
            return None
        if self.make_final_prediction:
            self.final_prediction_made = True
            self.done_training = True
        if (
            self.config.earlystop is not None and
            time.time() - self.birthday > self.config.earlystop
        ):
            self.done_training = True
            return None

        dataset_changed = self._tf_test_set != dataset
        if dataset_changed:
            self._tf_test_set = dataset
            test_initial_batch_size = self.config.dataloader_args['test'].pop(
                'test_initial_batch_size'
            )
            self.config.dataloader_args['test'].update(
                {'batch_size': test_initial_batch_size}
            )
            # Create a temporary handle to inspect data
            dataset = dataset.prefetch(self.config.dataloader_args['test']['batch_size'])
            ds_temp = TFDataset(self.session, dataset)
            self.setup_test_dataset(dataset, ds_temp)

        LOGGER.info('BATCH SIZE: {}'.format(self.test_dl.batch_size))
        test_finished = False
        while not test_finished:
            try:
                test_args = self.config.tester_args[self.config.tester] if isinstance(
                    self.tester, types.FunctionType
                ) else {}
                predicitons = self.tester(self, remaining_time_budget, **test_args)
                test_finished = True
            except RuntimeError as e:
                # If we are out of vram reduce the batchsize by 25 and try again
                # but dont lower it below 16
                if 'CUDA out of memory.' not in e.args[0]:
                    raise e
                loader_args = self.config.dataloader_args['test']
                loader_args.update(
                    {'batch_size': max(16, int(self.test_dl.batch_size - 25))}
                )
                self.test_dl.dataset.dataset = self.test_dl.dataset.dataset.prefetch(
                    loader_args['batch_size']
                )
                self.test_dl = TFDataLoader(self.test_dl.dataset, **loader_args)
                self.test_dl = TFDataLoader(self.test_dl.dataset, **loader_args)
                self.test_dl.dataset.reset()
                LOGGER.info('BATCH SIZE CHANGED: {}'.format(self.test_dl.batch_size))

        LOGGER.info(
            'AVERAGE VRAM USAGE: {0:.2f} MB'.format(
                np.mean(cutorch.memory_cached()) / 1024**2
            )
        )
        LOGGER.info("TESTING TOOK: {0:.6g}".format(time.time() - test_start))
        LOGGER.info(80 * '#')
        self.testing_round += 1
        self.test_time.append(time.time() - test_start)
        return predicitons
