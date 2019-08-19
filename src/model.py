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
import re
import threading
import time
import types
from collections import OrderedDict
from functools import partial, wraps

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


def parse_cumem_error(err_str):
    mem_search = re.search(
        r"Tried to allocate ([0-9].*? [G|M])iB.*\; ([0-9]*.*? [G|M])iB free", err_str
    )
    tried, free = mem_search.groups()
    tried = float(tried[:-2]) * 1024 if tried[-1] == 'G' else float(tried[:-2])
    free = float(free[:-2]) * 1024 if free[-1] == 'G' else float(free[:-2])
    return tried, free


def BSGuard(f, autodl_model, dataloader_attr, reset_on_fail):
    @wraps(f)
    def decorated(*args, **kwargs):
        while True:
            try:
                return f(*args, **kwargs)
            except RuntimeError as e:
                LOGGER.warn('CAUGHT VMEM ERROR! SCALING DOWN BATCH-SIZE!')
                if 'CUDA out of memory.' not in e.args[0]:
                    raise e
                tried_mem, free_mem = parse_cumem_error(e.args[0])
                mem_downscale = free_mem / tried_mem
                loader = getattr(autodl_model, dataloader_attr)
                loader.batch_size = int(loader.batch_size * mem_downscale)
                if reset_on_fail:
                    getattr(autodl_model, dataloader_attr).dataset.reset()
                LOGGER.warn(
                    'BATCH-SIZE NOW IS {}'.format(
                        getattr(autodl_model, dataloader_attr).batch_sampler.batch_size
                    )
                )

    return decorated


class Model(algorithm.Algorithm):
    def __init__(self, metadata):
        self.birthday = time.time()
        self.config = utils.Config(os.path.join(BASEDIR, "config.hjson"))
        bohb_conf_path = os.path.join(BASEDIR, 'sideload_config.json')
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

        ################
        # Metadata Stuff
        ################
        # In-/Out Dimensions from the train dataset's metadata
        row_count, col_count = metadata.get_matrix_size(0)
        channel = metadata.get_num_channels(0)
        sequence_size = metadata.get_sequence_size()
        self.input_dim = [sequence_size, row_count, col_count, channel]
        self.output_dim = metadata.get_output_size()
        self.num_train_samples = metadata.size()
        self.num_test_samples = None
        self.metadata = metadata
        test_metadata_filename = self.metadata.get_dataset_name(
        ).replace('train', 'test') + '/metadata.textproto'
        self.num_test_samples = [
            int(line.split(':')[1])
            for line in open(test_metadata_filename, 'r').readlines()
            if 'sample_count' in line
        ][0]

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
        self.trainer = BSGuard(
            self.trainer if isinstance(self.trainer, types.FunctionType) else
            self.trainer(**trainer_args), self, 'train_dl', True
        )
        self.tester = getattr(testing, self.config.tester)
        tester_args = self.config.tester_args[self.config.tester]
        self.tester = BSGuard(
            self.tester if isinstance(self.tester, types.FunctionType) else
            self.tester(**tester_args).__call__, self, 'test_dl', True
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
        self._trainer_args = None
        self._tester_args = None
        LOGGER.info("INIT END: {}".format(time.time()))

    @property
    def session(self):
        if self._session is None:
            [t.join() for t in THREADS]
            self._session = tf.Session()
        return self._session

    @property
    def trainer_args(self):
        if self._trainer_args is None:
            self._trainer_args = self.config.trainer_args[
                self.config.trainer
            ] if isinstance(getattr(training, self.config.trainer),
                            types.FunctionType) else {}
        return self._trainer_args

    @property
    def tester_args(self):
        if self._tester_args is None:
            self._tester_args = self.config.tester_args[self.config.tester] if isinstance(
                getattr(testing, self.config.tester), types.FunctionType
            ) else {}
        return self._tester_args

    def side_load_config(self, bohb_conf_path):
        '''
        This overrides all setting config defined in the sideload_config
        with the only requirement that their type must match

        ATTENTION: If sideload_config.json can define dictionaries as values as well
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

    def check_for_shuffling(self, ds_temp, transform):
        LOGGER.debug('SANITY CHECKING SHUFFLING')
        ds_train = TFDataset(
            session=self.session,
            dataset=self.test_dl.dataset,
            num_samples=self.num_train_samples
        )
        ds_train.reset()
        ds_temp.reset()
        dset1 = [e for e in ds_train]
        dset2 = [e for e in ds_train]
        dset3 = [e for e in ds_temp][:self.num_train_samples]
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

    def setup_train_dataset(self, ds_temp):
        ds_temp.transform_sample = self.transforms['train']['samples']
        ds_temp.transform_label = self.transforms['train']['labels']
        loader_args = self.config.dataloader_args['train']
        self.train_dl = TFDataLoader(ds_temp, **loader_args)
        if self.config.benchmark_transformations:
            self.train_dl.dataset.benchmark_transofrmations()
        if self.config.check_for_shuffling:
            self.check_for_shuffling()

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
            dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
            ds_temp = TFDataset(self.session, dataset, self.num_train_samples)
            ds_temp.scan_all(10)
            ds_temp.reset()
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
            self.setup_train_dataset(ds_temp)
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

        LOGGER.info('BATCH SIZE:\t\t{}'.format(self.train_dl.batch_size))
        train_start = time.time()

        self.trainer(self, remaining_time_budget, **self.trainer_args)

        LOGGER.info("TRAINING TOOK: {0:.6g}".format(time.time() - train_start))
        self.training_round += 1
        self.train_time.append(time.time() - train_start)

    def setup_test_dataset(self, dataset):
        ds = TFDataset(
            session=self.session,
            dataset=dataset,
            num_samples=int(self.num_test_samples),
            transform_sample=self.transforms['test']['samples'],
            transform_label=self.transforms['test']['labels']
        )
        loader_args = self.config.dataloader_args['test']
        self.test_dl = TFDataLoader(ds, **loader_args)

    def test(self, dataset, remaining_time_budget=None):
        test_start = time.time()
        self.current_remaining_time = remaining_time_budget
        LOGGER.info("REMAINING TIME: {}".format(remaining_time_budget))
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
            dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
            self.setup_test_dataset(dataset)

        LOGGER.info('BATCH SIZE: {}'.format(self.test_dl.batch_size))

        predicitons = self.tester(self, remaining_time_budget, **self.tester_args)

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
