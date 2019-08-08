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
        self.current_train_dataset = None
        self.current_test_dataset = None
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

    def split_dataset(self, dataset, transform):
        # [train_percent, validation_percent, ...]
        split_percentages = (
            self.config.dataset_split_ratio
            / np.sum(self.config.dataset_split_ratio)
        )
        split_num = np.round((self.num_examples_train * split_percentages))
        assert(sum(split_num) == self.num_examples_train)

        dataset.shuffle(self.num_examples_train)

        dataset_remaining = dataset
        dataset_train = dataset_remaining.take(split_num[0])
        dataset_remaining = dataset.skip(split_num[0])
        dataset_val = dataset_remaining.take(split_num[1])
        dataset_remaining = dataset_remaining.skip(split_num[1])

        ds_train = TFDataset(
            session=self.session,
            dataset=dataset_train,
            num_samples=int(split_num[0]),
            transform=transform
        )

        dl_train = FixedSizeDataLoader(
            ds_train,
            steps=int(split_num[0]),
            batch_size=self.parser_args.batch_size_train,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
            drop_last=False
        )

        ds_val = TFDataset(
            session=self.session,
            dataset=dataset_val,
            num_samples=int(split_num[1]),
            transform=transform
        )

        dl_val = FixedSizeDataLoader(
            ds_val,
            steps=int(split_num[1]),
            batch_size=self.parser_args.batch_size_train,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
            drop_last=False
        )

        self.train_ewm_window = np.ceil(split_num[0] / self.parser_args.batch_size)
        self.val_ewm_window = np.ceil(split_num[0] / self.parser_args.batch_size)

        return dl_train, dl_val

    def setup_train_dataset(self, dataset, ds_temp=None):
        transf_dict = self.select_transformations(
            self.current_train_dataset, self.model,
            **self.config.transformations_selector_args[
                self.config.transformations_selector
            ]
        )
        self.current_train_dataset = FixedSizeDataLoader(
            dataset,
            self.config.dataloader_args,
            transf_dict['train']['samples'],
            transf_dict['test']['labels']
        )

    def setup_test_dataset(self, dataset, ds_temp=None):
        transf_dict = self.select_transformations(
            self.current_train_dataset, self.model,
            **self.config.transformations_selector_args[
                self.config.transformations_selector
            ]
        )
        ds = TFDataset(
            session=self.session,
            dataset=dataset,
            num_samples=self.num_samples_testing,
            transform=transform
        )

        dl = torch.utils.data.DataLoader(
            ds,
            batch_size=self.parser_args.batch_size_test,
            drop_last=False
        )

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

    def train(self, dataset, remaining_time_budget=None):
        if self.final_prediction_made:
            return
        dataset_changed = self.current_train_dataset != dataset
        if dataset_changed:
            # Create a temporary handle to inspect data
            ds_temp = TFDataset(self.session, dataset, self.num_train_samples)
            ds_temp.scan2()
            # self.
            self.setup_model(ds_temp)
            self.setup_train_dataset(dataset, ds_temp)

        self.training_round += 1
        train_start = time.time()
        self.make_final_prediction = self.trainer(
            self,
            self.current_train_dataset['train'],
            self.current_train_dataset['val'],
            remaining_time_budget
        )
        self.train_time.append(time.time() - train_start)

    def test(self, dataset, remaining_time_budget=None):
        if self.final_prediction_made:
            return None
        if self.make_final_prediction:
            self.final_prediction_made = True
            self.done_training = True
        if (
            self.config.earlystop is not None
            and time.time() - self.birthday > 300
        ):
            self.done_training = True
            return None

        dataset_changed = self.current_test_dataset != dataset
        if dataset_changed:
            self.setup_test_dataset(dataset)

        self.testing_round += 1
        test_start = time.time()

        predictions = None
        self.model.eval()
        with torch.no_grad():
            for i, (data, _) in enumerate(self.current_test_dataset):
                LOGGER.info('test: ' + str(i))
                data = data.cuda()
                output = self.model(data)
                predictions = output if predictions is None \
                    else torch.cat((predictions, output), 0)

        LOGGER.info("TESTING END: " + str(time.time()))
        self.test_time.append(time.time() - test_start)
        return predictions
