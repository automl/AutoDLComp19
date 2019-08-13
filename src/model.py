# Modified by: Shangeth Rajaa, ZhengYing, Isabelle Guyon
"""An example of code submission for the AutoDL challenge in PyTorch.

It implements 3 compulsory methods: __init__, train, and test.
model.py follows the template of the abstract class algorithm.py found
in folder ingestion_program/.

The dataset is in TFRecords and Tensorflow is used to read TFRecords and get the
Numpy array which can be used in PyTorch to convert it into Torch Tensor.

To create a valid submission, zip model.py together with other necessary files
such as Python modules/packages, pre-trained weights. The final zip file should
not exceed 300MB.
"""
import os
import time
import types
from functools import partial

# Import the challenge algorithm (model) API from algorithm.py
import numpy as np
import torch
import tensorflow as tf

import algorithm
import selection
import transformations
import training
import testing
import utils
from torch_adapter import TFDataset
from utils import LOGGER


# If apex's amp is available, import it and set a flag to use it
try:
    from apex import amp
    USE_AMP = True
except Exception:
    USE_AMP = False
    pass

if USE_AMP:
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

# Set the device which torch should use
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BASEDIR = os.path.dirname(os.path.abspath(__file__))


class Model(algorithm.Algorithm):
    def __init__(self, metadata):
        self.birthday = time.time()
        self.config = utils.Config(os.path.join(BASEDIR, "config.hjson"))
        LOGGER.info("INIT START: " + str(time.time()))
        super(Model, self).__init__(metadata)
        # This flag allows you to enable the inbuilt cudnn auto-tuner to find the best
        # algorithm to use for your hardware. Benchmark mode is good whenever your input sizes
        # for your network do not vary
        # https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do/5936
        torch.backends.cudnn.benchmark = self.config.cudnn_benchmark
        torch.backends.cudnn.deterministic = self.config.cudnn_deterministic

        # Seeds
        np.random.seed = self.config.np_random_seed
        torch.manual_seed = self.config.torch_manual_seed
        torch.cuda.manual_seed_all = self.config.torch_cuda_manual_seed_all

        # Assume model.py and config.hjson are always in the same folder. Could possibly
        # do this in a nicer fashion, but it must still run during the submission on
        # codalab.

        # In-/Out Dimensions from the train dataset's metadata
        row_count, col_count = metadata.get_matrix_size(0)
        channel = metadata.get_num_channels(0)
        sequence_size = metadata.get_sequence_size()
        self.input_dim = [sequence_size, row_count, col_count, channel]
        self.output_dim = metadata.get_output_size()
        self.num_train_samples = metadata.size()
        self.num_test_samples = None

        # Store the current dataset's path, loader and the currently used
        # model, optimizer and lossfunction
        self.tf_train_set = None
        self.tf_test_set = None
        self.train_dl = None
        self.test_dl = None
        self.transforms = None
        self.model = None
        self.optimizer = None
        self.lr_scheduler = None
        self.loss_fn = None

        # Set the algorithms to use from the config file
        self.select_model = getattr(
            selection,
            self.config.model_selector
        )
        self.select_transformations = getattr(
            transformations,
            self.config.transformations_selector
        )
        self.trainer = getattr(
            training,
            self.config.trainer
        )
        trainer_args = self.config.trainer_args[self.config.trainer]
        self.trainer = self.trainer if isinstance(
            self.trainer,
            types.FunctionType
        ) else self.trainer(**trainer_args)
        self.tester = getattr(
            testing,
            self.config.tester
        )
        tester_args = self.config.tester_args[self.config.tester]
        self.tester = self.tester if isinstance(
            self.tester,
            types.FunctionType
        ) else self.tester(**tester_args)

        # Attributes for managing time budget
        # Cumulated number of training steps
        self.make_final_prediction = False
        self.final_prediction_made = False
        self.done_training = False

        self.training_round = 0  # flag indicating if we are in the first round of training
        self.train_time = []
        self.testing_round = 0  # flag indicating if we are in the first round of testing
        self.test_time = []

        self.session = tf.Session()
        LOGGER.info("INIT END: " + str(time.time()))

    def setup_model(self, ds_temp):
        selected = self.select_model(
            self.session,
            ds_temp,
            **self.config.model_selector_args[
                self.config.model_selector
            ]
        )
        self.model, self.loss_fn, self.optimizer, self.lr_scheduler = selected

    def setup_transforms(self, ds_temp):
        self.transforms = self.select_transformations(
            self,
            ds_temp,
            **self.config.transformations_selector_args[
                self.config.transformations_selector
            ]
        )

    def split_dataset(self, ds_temp, transform):
        # [train_percent, validation_percent, ...]
        split_percentages = (
            self.config.dataset_split_ratio
            / np.sum(self.config.dataset_split_ratio)
        )
        split_num = np.round((self.num_train_samples * split_percentages))

        if split_num[1] < ds_temp.num_classes * 3 and split_num[1] > 0.:
            split_percentages = np.array((
                self.num_train_samples - (ds_temp.num_classes * 5), (ds_temp.num_classes * 5)
            ))
        assert(sum(split_num) == self.num_train_samples)

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
            'train': torch.utils.data.DataLoader(
                ds_train,
                **self.config.dataloader_args['train']
            ),
            'val': torch.utils.data.DataLoader(
                ds_val,
                **self.config.dataloader_args['train']
            ) if split_num[1] > 0. else None
        }

    def setup_train_dataset(self, dataset, ds_temp=None):
        self.split_dataset(ds_temp, self.transforms['train'])
        if self.config.benchmark_transformations:
            self.train_dl['train'].dataset.benchmark_transofrmations()

    def train(self, dataset, remaining_time_budget=None):
        if self.final_prediction_made:
            return
        dataset_changed = self.tf_train_set != dataset
        if dataset_changed:
            self.tf_train_set = dataset
            # Create a temporary handle to inspect data
            dataset = dataset.prefetch(
                self.config.dataloader_args['train']['batch_size']
            )
            ds_temp = TFDataset(self.session, dataset, self.num_train_samples)
            ds_temp.scan_all(50)

            self.setup_model(ds_temp)
            self.setup_transforms(ds_temp)
            self.setup_train_dataset(dataset, ds_temp)

            # Finally move the model to gpu
            self.model.to(DEVICE)
            self.loss_fn.to(DEVICE)
            if (
                USE_AMP
                and hasattr(self.model, 'amp_compatible')
                and self.model.amp_compatible
            ):
                self.model, self.optimizer = amp.initialize(
                    self.model, self.optimizer, **self.config.amp_args
                )
                self.loss_fn = partial(
                    amp_loss, loss_fn=self.loss_fn, optimizer=self.optimizer
                )

        train_start = time.time()
        train_args = self.config.trainer_args[
            self.config.trainer
        ] if isinstance(self.trainer, types.FunctionType) else {}
        self.trainer(
            self,
            remaining_time_budget,
            **train_args
        )
        LOGGER.info("TRAINING TOOK: " + str(time.time() - train_start))
        self.training_round += 1
        self.train_time.append(time.time() - train_start)

    def setup_test_dataset(self, dataset, ds_temp=None):
        ds = TFDataset(
            session=self.session,
            dataset=dataset,
            num_samples=int(self.num_test_samples),
            transform_sample=self.transforms['test']['samples'],
            transform_label=self.transforms['test']['labels']
        )
        ds.min_shape = ds_temp.min_shape
        ds.median_shape = ds_temp.median_shape
        ds.mean_shape = ds_temp.mean_shape
        ds.std_shape = ds_temp.std_shape
        ds.max_shape = ds_temp.max_shape
        ds.is_multilabel = ds_temp.is_multilabel
        self.test_dl = torch.utils.data.DataLoader(
            ds,
            **self.config.dataloader_args['test']
        )

    def test(self, dataset, remaining_time_budget=None):
        if self.config.benchmark_time_till_first_prediction:
            LOGGER.error('TIME TILL FIRST PREDICTION: {0}'.format(time.time() - self.birthday))
            LOGGER.error('TRAIN ERR SHAPE: {0}'.format(self.trainer.train_err.shape))
            return None
        if self.final_prediction_made:
            return None
        if self.make_final_prediction:
            self.final_prediction_made = True
            self.done_training = True
        if (
            self.config.earlystop is not None
            and time.time() - self.birthday > self.config.earlystop
        ):
            self.done_training = True
            return None

        dataset_changed = self.tf_test_set != dataset
        if dataset_changed:
            self.tf_test_set = dataset
            # Create a temporary handle to inspect data
            dataset = dataset.prefetch(
                self.config.dataloader_args['test']['batch_size']
            )
            ds_temp = TFDataset(self.session, dataset)
            ds_temp.scan_all()
            self.num_test_samples = ds_temp.num_samples
            self.setup_test_dataset(dataset, ds_temp)

        test_start = time.time()
        test_args = self.config.tester_args[
            self.config.tester
        ] if isinstance(self.tester, types.FunctionType) else {}
        predicitons = self.tester(
            self,
            remaining_time_budget,
            **test_args
        )
        LOGGER.info("TESTING TOOK: " + str(time.time() - test_start))
        self.testing_round += 1
        self.test_time.append(time.time() - test_start)
        return predicitons
