# Modified by: Shangeth Rajaa, ZhengYing, Isabelle Guyon
"""An example of code submission for the AutoDL challenge in PyTorch.

It implements 3 compulsory methods: __init__, train, and test.
model.py follows the template of the abstract class algorithm.py found
in folder AutoDL_ingestion_program/.

The dataset is in TFRecords and Tensorflow is used to read TFRecords and get
the Numpy array which can be used in PyTorch to convert it into Torch Tensor.

To create a valid submission, zip model.py together with other necessary files
such as Python modules/packages, pre-trained weights. The final zip file should
not exceed 300MB.
"""
"""
Search for '# PYTORCH' to get directly to PyTorch Code.
"""
import datetime
# Other useful modules
import logging
import os
import sys
import time

# Import the challenge algorithm (model) API from algorithm.py
import algorithm
import bohb
import configuration
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data_utils
import torchvision
from ops.load_dataloader import get_model_for_loader
from ops.load_models import load_loss_criterion, load_model_and_optimizer
from ops.temporal_shift import make_temporal_pool
from opts import parser
from AutoDL_sample_code_submission_pytorch.pytorch_dataset import PytorchDataset
from torch.nn.init import constant_, xavier_uniform_
from transforms import (
    GroupCenterCrop, GroupNormalize, GroupScale, IdentityTransform, SelectSamples, Stack,
    ToPilFormat, ToTorchFormatTensor
)

np.random.seed(42)
torch.manual_seed(1)
np.random.seed(1)


class ParserMock():
    # mock class for handing over the correct arguments
    def __init__(self):
        self._parser_args = parser.parse_known_args()[0]
        setattr(
            self._parser_args, 'finetune_model',
            '/home/dingsda/autodl/AutoDLComp19/src/video3/pretrained_models/somethingv2_rgb_epoch_16_checkpoint.pth.tar'
        )
        setattr(self._parser_args, 'arch', 'resnet50')
        setattr(self._parser_args, 'batch_size', 1)
        setattr(self._parser_args, 'modality', 'RGB')
        setattr(self._parser_args, 'classification_type', 'multiclass')
        setattr(self._parser_args, 'shift', True)
        setattr(self._parser_args, 'shift_div', 9)
        setattr(self._parser_args, 'shift_place', 'blockres')
        setattr(self._parser_args, 'dense_sample', True)
        setattr(self._parser_args, 'print', True)

    def set_attr(self, attr, val):
        setattr(self._parser_args, attr, val)

    def parse_args(self):
        return self._parser_args


class Model(algorithm.Algorithm):
    def __init__(self, metadata):
        print('INIT')
        super(Model, self).__init__(metadata)
        self.no_more_training = False
        self.output_dim = self.metadata_.get_output_size()
        self.num_examples_train = self.metadata_.size()
        row_count, col_count = self.metadata_.get_matrix_size(0)
        channel = self.metadata_.get_num_channels(0)
        sequence_size = self.metadata_.get_sequence_size()
        print('INPUT SHAPE :', row_count, col_count, channel, sequence_size)

        # getting an object for the PyTorch Model class for Model Class
        # use CUDA if available
        parser = ParserMock()
        parser.set_attr('num_classes', self.output_dim)

        self.parser_args = parser.parse_args()
        self.config = configuration.get_configspace(model_name=self.parser_args.arch
                                                   ).get_default_configuration()
        self.model, self.optimizer = load_model_and_optimizer(
            self.parser_args, self.config
        )
        self.criterion = load_loss_criterion(self.parser_args)

        if torch.cuda.is_available():
            self.model.cuda()

        # Attributes for managing time budget
        # Cumulated number of training steps
        self.birthday = time.time()
        self.total_train_time = 0
        self.cumulated_num_steps = 0
        self.estimated_time_per_step = None
        self.total_test_time = 0
        self.cumulated_num_tests = 0
        self.estimated_time_test = None
        self.trained = False
        self.done_training = False

        # Do not use the tensorflow dataset but the dataset path directly
        self.use_dataset_path = True

        # PYTORCH
        # Critical number for early stopping
        self.num_epochs_we_want_to_train = 100


    def train(self, dataset, remaining_time_budget=None):
        print('TRAINING')

        steps_to_train = self.get_steps_to_train(remaining_time_budget)
        if steps_to_train <= 0:
            print_log("Not enough time remaining for training. " +
                  "Estimated time for training per step: {:.2f}, "\
                  .format(self.estimated_time_per_step) +
                  "but remaining time budget is: {:.2f}. "\
                  .format(remaining_time_budget) +
                  "Skipping...")
            self.done_training = True
        else:
            msg_est = ""
            if self.estimated_time_per_step:
                msg_est = "estimated time for this: " +\
                          "{:.2f} sec.".format(steps_to_train * self.estimated_time_per_step)
            print_log(
                "Begin training for another {} steps...{}".format(
                    steps_to_train, msg_est
                )
            )

            if not hasattr(self, 'trainloader'):
                print('BATCH SIZE: ' + str(self.parser_args.batch_size))
                self.trainloader = self.get_dataloader(
                    dataset=dataset,
                    parser_args=self.parser_args,
                    train=True
                )

            train_start = time.time()

            # PYTORCH
            #Training loop inside
            self.trainloop(self.model, self.criterion, self.optimizer, steps=steps_to_train, parser_args=self.parser_args)

            train_end = time.time()

            # Update for time budget managing
            train_duration = train_end - train_start
            self.total_train_time += train_duration
            self.cumulated_num_steps += steps_to_train
            self.estimated_time_per_step = self.total_train_time / self.cumulated_num_steps
            print_log("{} steps trained. {:.2f} sec used. ".format(steps_to_train, train_duration) +\
                  "Now total steps trained: {}. ".format(self.cumulated_num_steps) +\
                  "Total time used for training: {:.2f} sec. ".format(self.total_train_time) +\
                  "Current estimated time per step: {:.2e} sec.".format(self.estimated_time_per_step))


    def test(self, dataset, remaining_time_budget=None):
        print('TESTING')

        if self.done_training:
            return None

        if self.choose_to_stop_early():
            print_log("Oops! Choose to stop early for next call!")
            self.done_training = True
        test_begin = time.time()
        if remaining_time_budget and self.estimated_time_test and\
            self.estimated_time_test > remaining_time_budget:
            print_log("Not enough time for test. " +\
                  "Estimated time for test: {:.2e}, ".format(self.estimated_time_test) +\
                  "But remaining time budget is: {:.2f}. ".format(remaining_time_budget) +\
                  "Stop train/predict process by returning None.")
            return None

        msg_est = ""
        if self.estimated_time_test:
            msg_est = "estimated time: {:.2e} sec.".format(self.estimated_time_test)
        print_log("Begin testing...", msg_est)

        # PYTORCH
        if not hasattr(self, 'testloader'):
            self.testloader = self.get_dataloader(
                dataset, parser_args=self.parser_args, train=False
            )
        predictions = self.testloop(self.testloader)

        test_end = time.time()
        # Update some variables for time management
        test_duration = test_end - test_begin
        self.total_test_time += test_duration
        self.cumulated_num_tests += 1
        self.estimated_time_test = self.total_test_time / self.cumulated_num_tests
        print_log("[+] Successfully made one prediction. {:.2f} sec used. ".format(test_duration) +\
              "Total time used for testing: {:.2f} sec. ".format(self.total_test_time) +\
              "Current estimated time for test: {:.2e} sec.".format(self.estimated_time_test))
        return predictions


    def trainloop(self, model, criterion, optimizer, steps, parser_args):
        '''
        # PYTORCH
        Trainloop function does the actual training of the model
        1) it gets the X, y from tensorflow dataset.
        2) convert X, y to CUDA
        3) trains the model with the Tesors for given no of steps.
        '''
        self.model.train()
        data_iterator = iter(self.trainloader)
        for i in range(steps):
            try:
                images, labels = next(data_iterator)
            except StopIteration:
                data_iterator = iter(self.trainloader)
                images, labels = next(data_iterator)

            # manually convert one-hot encoding back to integer *doh*
            idx = np.argmax(labels, axis=1)
            labels_int = torch.LongTensor([labels[i][idx[i]] for i in range(len(idx))])
            print(images.shape)
            print(labels_int.shape)
            bohb.train_inner(model, optimizer, criterion, images, labels_int, i, parser_args)


    def testloop(self, dataloader):
        '''
        # PYTORCH
        testloop uses testdata to test the pytorch model and return onehot prediciton values.
        '''
        preds = []
        with torch.no_grad():
            self.model.eval()
            for images in dataloader:
                if torch.cuda.is_available():
                    images = images.cuda()
                log_ps = self.model(images)
                pred = torch.sigmoid(log_ps)  #.data > 0.5
                preds.append(pred.cpu().numpy())
        preds = np.concatenate(preds)
        return preds

    def get_dataloader(self, dataset, parser_args, train=False):
        model = get_model_for_loader(parser_args)
        train_augmentation = model.get_augmentation()
        input_mean = model.input_mean
        input_std = model.input_std
        num_segments = parser_args.num_segments

        dataset = PytorchDataset(
            dataset,
            nr_buffer_shards=32,
            transform_sample=torchvision.transforms.Compose(
                [
                    SelectSamples(num_segments),
                    ToPilFormat(),
                    train_augmentation,
                    Stack(roll=True),
                    ToTorchFormatTensor(div=False),
                    GroupNormalize(input_mean, input_std),
                ]
            )
        )
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=parser_args.batch_size,
            shuffle=train,
            num_workers=0,
            pin_memory=False
        )
        return dataloader


    def get_steps_to_train(self, remaining_time_budget):
        """Get number of steps for training according to `remaining_time_budget`.
        The strategy is:
          1. If no training is done before, train for 10 steps (ten batches);
          2. Otherwise, estimate training time per step and time needed for test,
             then compare to remaining time budget to compute a potential maximum
             number of steps (max_steps) that can be trained within time budget;
          3. Choose a number (steps_to_train) between 0 and max_steps and train for
             this many steps. Double it each time.
        """
        if not remaining_time_budget:  # This is never true in the competition anyway
            remaining_time_budget = 1200  # if no time limit is given, set to 20min

        if not self.estimated_time_per_step:
            steps_to_train = 10
        else:
            if self.estimated_time_test:
                tentative_estimated_time_test = self.estimated_time_test
            else:
                tentative_estimated_time_test = 50  # conservative estimation for test
            max_steps = int(
                (remaining_time_budget - tentative_estimated_time_test) /
                self.estimated_time_per_step
            )
            max_steps = max(max_steps, 1)
            if self.cumulated_num_tests < np.log(max_steps) / np.log(2):
                steps_to_train = int(
                    2**self.cumulated_num_tests
                )  # Double steps_to_train after each test
            else:
                steps_to_train = 0
        return steps_to_train


    def choose_to_stop_early(self):
        """The criterion to stop further training (thus finish train/predict
        process).
        """
        # return self.cumulated_num_tests > 10 # Limit to make 10 predictions
        # return np.random.rand() < self.early_stop_proba
        batch_size = self.parser_args.batch_size
        num_examples = self.metadata_.size()
        num_epochs = self.cumulated_num_steps * batch_size / num_examples
        print_log("Model already trained for {} epochs.".format(num_epochs))
        # Train for at least certain number of epochs then stop
        return num_epochs > self.num_epochs_we_want_to_train


##############################################################################
#### Above 3 methods (__init__, train, test) should always be implemented ####
##############################################################################


#### Can contain other functions too
def print_log(*content):
    """Logging function. (could've also used `import logging`.)"""
    now = datetime.datetime.now().strftime("%y-%m-%d %H:%M:%S")
    print("MODEL INFO: " + str(now) + " ", end='')
    print(*content)


def get_num_entries(tensor):
    """Return number of entries for a TensorFlow tensor.
    Args:
    tensor: a tf.Tensor or tf.SparseTensor object of shape
        (batch_size, sequence_size, row_count, col_count[, num_channels])
    Returns:
    num_entries: number of entries of each example, which is equal to
        sequence_size * row_count * col_count [* num_channels]
    """
    tensor_shape = tensor.shape
    assert (len(tensor_shape) > 1)
    num_entries = 1
    for i in tensor_shape[1:]:
        num_entries *= int(i)
    return num_entries
