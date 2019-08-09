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
import training
import transformations
import utils
from utils import LOGGER
from dataset_kakaobrain import TFDataset
from dataloader_kakaobrain import FixedSizeDataLoader


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

# Set seeds
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

# Set the device which torch should use
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BASEDIR = os.path.dirname(os.path.abspath(__file__))


class Model(algorithm.Algorithm):
    def __init__(self, metadata):
        self.birthday = time.time()
        LOGGER.info("INIT START: " + str(time.time()))
        super(Model, self).__init__(metadata)
        # This flag allows you to enable the inbuilt cudnn auto-tuner to find the best
        # algorithm to use for your hardware. Benchmark mode is good whenever your input sizes
        # for your network do not vary
        # https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do/5936
        torch.backends.cudnn.benchmark = True

        # Assume model.py and config.hjson are always in the same folder. Could possibly
        # do this in a nicer fashion, but it must still run during the submission on
        # codalab.
        self.config = utils.Config(os.path.join(BASEDIR, "config.hjson"))

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
        self.model = None
        self.optimizer = None
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
        self.trainer = self.trainer if isinstance(
            self.trainer,
            types.FunctionType
        ) else self.trainer()

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
        self.model, self.loss_fn, self.optimizer, amp_compatible = selected
        self.model.to(DEVICE)
        self.loss_fn.to(DEVICE)
        if USE_AMP and amp_compatible:
            self.model, self.optimizer = amp.initialize(
                self.model, self.optimizer, **self.config.amp_args
            )
            self.loss_fn = partial(
                amp_loss, loss_fn=self.loss_fn, optimizer=self.optimizer
            )

    def split_dataset(self, ds_temp, transform):
        # [train_percent, validation_percent, ...]
        split_percentages = (
            self.config.dataset_split_ratio
            / np.sum(self.config.dataset_split_ratio)
        )
        split_num = np.round((self.num_train_samples * split_percentages))
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
        ds_train.max_shape = ds_temp.max_shape
        ds_train.is_multilabel = ds_temp.is_multilabel

        ds_val = TFDataset(
            session=self.session,
            dataset=tfdataset_val,
            num_samples=int(split_num[1]),
            transform_sample=transform['samples'],
            transform_label=transform['labels']
        )
        ds_val.min_shape = ds_temp.min_shape
        ds_val.median_shape = ds_temp.median_shape
        ds_val.max_shape = ds_temp.max_shape
        ds_val.is_multilabel = ds_temp.is_multilabel

        self.train_dl = {
            'train': FixedSizeDataLoader(
                ds_train,
                steps=int(split_num[0]),
                **self.config.dataloader_args['train']
            ),
            'val': FixedSizeDataLoader(
                ds_val,
                steps=int(split_num[1]),
                **self.config.dataloader_args['train']
            )
        }

    def setup_train_dataset(self, dataset, ds_temp=None):
        transf_dict = self.select_transformations(
            ds_temp, self.model,
            **self.config.transformations_selector_args[
                self.config.transformations_selector
            ]
        )
        self.split_dataset(ds_temp, transf_dict['train'])

    def train(self, dataset, remaining_time_budget=None):
        if self.final_prediction_made:
            return
        dataset_changed = self.tf_train_set != dataset
        if dataset_changed:
            self.tf_train_set = dataset
            # Create a temporary handle to inspect data
            ds_temp = TFDataset(self.session, dataset, self.num_train_samples)
            ds_temp.scan2()

            self.setup_model(ds_temp)
            self.setup_train_dataset(dataset, ds_temp)

        train_start = time.time()
        self.make_final_prediction = self.trainer(
            self,
            self.train_dl['train'],
            self.train_dl['val'],
            remaining_time_budget,
            **self.config.trainer_args[
                self.config.trainer
            ]
        )
        self.training_round += 1
        self.train_time.append(time.time() - train_start)

    def setup_test_dataset(self, dataset, ds_temp=None):
        transf_dict = self.select_transformations(
            ds_temp, self.model,
            **self.config.transformations_selector_args[
                self.config.transformations_selector
            ]
        )
        ds = TFDataset(
            session=self.session,
            dataset=dataset,
            num_samples=self.num_test_samples,
            transform_sample=transf_dict['test']['samples'],
            transform_label=transf_dict['test']['labels']
        )
        # TODO(Philipp J.): what is the difference between torch.utils.data.DataLoader/FixedSizeDataLoader
        self.test_dl = torch.utils.data.DataLoader(
            ds,
            **self.config.dataloader_args['test']
        )

    def test(self, dataset, remaining_time_budget=None):
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
            ds_temp = TFDataset(self.session, dataset, 1e10)
            ds_temp.scan2()
            self.num_test_samples = ds_temp.num_samples

            self.setup_test_dataset(dataset, ds_temp)

        test_start = time.time()

        predictions = None
        e = enumerate(self.test_dl)
        self.model.eval()
        with torch.no_grad():
            for i, (data, _) in e:
                LOGGER.info('test: ' + str(i))
                data = data.cuda()
                output = self.model(data)
                predictions = output if predictions is None \
                    else torch.cat((predictions, output), 0)

        LOGGER.info("TESTING END: " + str(time.time()))
        self.testing_round += 1
        self.test_time.append(time.time() - test_start)
        return predictions.cpu().numpy()
