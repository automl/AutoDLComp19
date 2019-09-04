"""
This class implements the 3 compulsory methods
a submission is required to have namely __init__, train, test

The main idea behind the current structure is to split up
the architecture, train and eval as model agnostic as possible
to accommodate the different modalities and models we need to train.

As compared to last version of the architecture, mutually shared logic
has been centralized. Only the models' selection
(their transformations included) and is now the central hub for defining
the executed pipline's logic.

The biggest change is the config.hjson as it no longer allows
multiple trainers/testers/


NOTE:
A hjson is a normal json but allows comments, so no black magic here.
"""
import json
import os
import threading
import time
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
import utils
from torch_adapter import TFAdapterSet, TFDataset
from utils import BASEDIR, DEVICE, LOGGER, MB, BSGuard, memprofile

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
        self.done_training = False
        self.config = utils.Config(os.path.join(BASEDIR, "config.hjson"))
        sideload_conf_path = os.path.join(BASEDIR, 'sideload_config.json')
        if os.path.isfile(sideload_conf_path):
            self.side_load_config(sideload_conf_path)

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
        self.metadata = metadata

        # row_count, col_count = metadata.get_matrix_size(0)
        # channel = metadata.get_num_channels(0)
        # sequence_size = metadata.get_sequence_size()
        # self.input_dim = [sequence_size, row_count, col_count, channel]
        # self.output_dim = metadata.get_output_size()

        # Train/Test length
        self.train_num_samples = metadata.size()
        self.test_num_samples = [
            int(line.split(':')[1]) for line in open(
                metadata.get_dataset_name().replace('train', 'test') +
                '/metadata.textproto', 'r'
            ).readlines() if 'sample_count' in line
        ][0]

        # Init the selector with the config
        self.selector = selection.Selector(self.config.selection)

        # Store the current dataset's path, loader and the currently used
        # model, optimizer and lossfunction
        self._tf_train_set = None
        self.train_dl = None
        self.train_loader_args = {}

        self._tf_test_set = None
        self.test_dl = None
        self.test_loader_args = {}

        self.transforms = None

        self.model = None
        self.optimizer = None
        self.loss_fn = None
        self.lr_scheduler = None

        self.policy_fn = None

        self.trainer = None
        self.trainer_args = getattr(self.config, 'trainer_args', {})

        self.tester = None
        self.tester_args = getattr(self.config, 'tester_args', {})

        # Attributes for managing time budget
        # Cumulated number of training steps
        self.starting_budget = None
        self.current_remaining_time = None
        self.make_final_prediction = False
        self.final_prediction_made = False

        self.train_time = []
        self.test_time = []

        self._session = None
        LOGGER.info("INIT END: {}".format(time.time()))

    @property
    def session(self):
        if self._session is None:
            [t.join() for t in THREADS]
            self._session = tf.Session()
        return self._session

    def side_load_config(self, sideload_conf_path):
        '''
        This overrides all configs defined in the sideload_config.
        To target specific leafs in a hierarchy use '.' to separate
        the parent in the leafs' path:
        ie. { selection.video.optim_args.lr: 0.1} would overwrite the
        video optimizer's initial learning rate

        ATTENTION: The sideload_config.json can define dictionaries as values as well
        so it is possible to overwrite a whole subhierarchy.
        '''
        def walk_dict(d, side_conf, p=''):
            for k, v in d.items():
                if isinstance(v, OrderedDict):
                    walk_dict(v, side_conf, p + k + '.')
                elif p + k in side_conf:
                    if not isinstance(d[k], type(side_conf[p + k])):
                        LOGGER.warning(
                            'Overriding config at "{}" value with another type! {} => {}'.
                            format(p + k, type(d[k]), type(side_conf[p + k]))
                        )
                    d[k] = side_conf[p + k]

        with open(sideload_conf_path, 'r') as file:
            side_conf = json.load(file)

        walk_dict(self.config.__dict__, side_conf)

    def _setup_dataset(self, dataset, target):
        loader_args = getattr(self, target + '_loader_args')
        ds = TFAdapterSet(
            session=self.session,
            dataset=dataset,
            num_samples=int(getattr(self, target + '_num_samples')),
            transformations=self.transforms[target],
            **loader_args
        )
        setattr(self, target + '_dl', ds)

    def _init_train(self, dataset, remaining_time_budget):
        t_s = time.time()
        self.starting_budget = t_s - self.birthday + remaining_time_budget

        self._tf_train_set = dataset

        # Create a temporary handle to inspect data
        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        ds_temp = TFDataset(self.session, dataset, self.train_num_samples)
        ds_temp.scan(100)
        LOGGER.info(
            'FETCHING ADDITIONAL METADATA TOOK: {0:.4f} s'.format(time.time() - t_s)
        )

        ######### SELECT MODEL
        t_s = time.time()
        selected = self.selector.select(self, ds_temp)
        self.__dict__.update(selected)
        if self.model is None:
            return None
        LOGGER.info('SETTING UP THE MODEL TOOK: {0:.4f} s'.format(time.time() - t_s))

        ########## SETUP TRAINLOADER
        t_s = time.time()
        self._setup_dataset(self._tf_train_set, 'train')
        datashape = {
            k: getattr(ds_temp, k)
            for k in [
                'num_samples',
                'num_classes',
                'is_multilabel',
                'min_shape',
                'max_shape',
                'median_shape',
                'mean_shape',
                'std_shape',
            ]
        }
        self.train_dl.dataset.__dict__.update(datashape)
        LOGGER.info(
            'SETTING UP THE TRAINLOADER TOOK: {0:.4f} s'.format(time.time() - t_s)
        )

        ########## SETUP TRAINER
        test_cache_size = (
            self.test_num_samples if
            getattr(self.config, 'tester_args', {'use_cache': True})['use_cache'] else 0
        )
        self.trainer = training.PolicyTrainer(
            **self.trainer_args,
            preserve_ram_for_nele=test_cache_size,
            policy_fn=self.policy_fn
        )
        self.trainer = BSGuard(self.trainer, self.train_dl, True)

        # Finally move the model/loss_fn to gpu and setup amp if wanted
        self.model.to(DEVICE, non_blocking=True)
        self.loss_fn.to(DEVICE, non_blocking=True)
        if (
            USE_AMP and self.config.use_amp and hasattr(self.model, 'amp_compatible') and
            self.model.amp_compatible
        ):
            self.model, self.optimizer = amp.initialize(
                self.model, self.optimizer, **self.config.amp_args
            )
            self.loss_fn = partial(
                amp_loss, loss_fn=self.loss_fn, optimizer=self.optimizer
            )

    def _init_test(self, dataset, remaining_time_budget):
        self._tf_test_set = dataset

        ########## SETUP TESTLOADER
        t_s = time.time()
        self._setup_dataset(self._tf_test_set, 'test')
        LOGGER.info('SETTING UP THE TESTLOADER TOOK: {0:.4f} s'.format(time.time() - t_s))

        ########## SETUP TESTER
        self.tester = testing.DefaultPredictor(**self.tester_args)
        self.tester = BSGuard(self.tester, self.test_dl, True)

    @memprofile(precision=2)
    def train(self, dataset, remaining_time_budget=None):
        train_start = time.time()
        self.current_remaining_time = remaining_time_budget
        LOGGER.info("REMAINING TIME: {0:.2f}".format(remaining_time_budget))
        if self.final_prediction_made:
            return
        dataset_changed = self._tf_train_set != dataset
        if dataset_changed:
            self._init_train(dataset, remaining_time_budget)
            if self.model is None:
                return

        LOGGER.info('BATCH SIZE:\t\t{}'.format(self.train_dl.batch_size))
        train_start = time.time()

        self.trainer(self, remaining_time_budget)

        LOGGER.info("TRAINING TOOK: {0:.6g}".format(time.time() - train_start))
        LOGGER.info(
            'AVERAGE VRAM USAGE: {0:.2f} MB'.format(
                np.mean(cutorch.memory_cached()) / MB
            )
        )
        self.train_time.append(time.time() - train_start)

    @memprofile(precision=2)
    def test(self, dataset, remaining_time_budget=None):
        test_start = time.time()
        self.current_remaining_time = remaining_time_budget
        LOGGER.info("REMAINING TIME: {0:.2f}".format(remaining_time_budget))
        if self.config.benchmark_time_till_first_prediction:
            LOGGER.error(
                'TIME TILL FIRST PREDICTION: {0}'.format(time.time() - self.birthday)
            )
            LOGGER.error('BATCHES PROCESSED: {0}'.format(self.trainer.batch_counter))
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
            self._init_test(dataset, remaining_time_budget)

        LOGGER.info('BATCH SIZE: {}'.format(self.test_dl.batch_size))

        predictions = self.tester(self, remaining_time_budget)

        LOGGER.info("TESTING TOOK: {0:.6g}".format(time.time() - test_start))
        LOGGER.info(
            'AVERAGE VRAM USAGE: {0:.2f} MB'.format(
                np.mean(cutorch.memory_cached()) / MB
            )
        )
        LOGGER.info(30 * '#' + ' LET' 'S GO FOR ANOTHER ROUND ' + 30 * '#')
        self.test_time.append(time.time() - test_start)
        return predictions
