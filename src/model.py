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
import time

import torch
import torch.nn as nn
import numpy as np

# Import the challenge algorithm (model) API from algorithm.py
import algorithm

import dataloading
import utils


# Set seeds
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
# TODO(Danny) tensorflow

# This flag allows you to enable the inbuilt cudnn auto-tuner to find the best
# algorithm to use for your hardware. Benchmark mode is good whenever your input sizes
# for your network do not vary
# https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do/5936
# TODO(Danny): How should we handle this
torch.backends.cudnn.benchmark = True


# PYTORCH
# Make pytorch model in torchModel class
class torchModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(torchModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, int((input_dim + output_dim) / 2))
        self.fc2 = nn.Linear(int((input_dim + output_dim) / 2), output_dim)
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = x.contiguous().view(x.size(0), -1)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.log_softmax(x)
        return x


class Model(algorithm.Algorithm):
    def __init__(self, metadata):
        super(Model, self).__init__(metadata)
        self.no_more_training = False
        self.output_dim = self.metadata_.get_output_size()
        self.num_examples_train = self.metadata_.size()
        row_count, col_count = self.metadata_.get_matrix_size(0)
        channel = self.metadata_.get_num_channels(0)
        sequence_size = self.metadata_.get_sequence_size()

        # Dataloading
        self.train_data_iterator = None

        # Attributes for preprocessing
        self.default_image_size = (112, 112)
        self.default_num_frames = 10
        self.default_shuffle_buffer = 100

        if row_count == -1 or col_count == -1:
            row_count = self.default_image_size[0]
            col_count = self.default_image_size[0]
        self.input_dim = row_count * col_count * channel * sequence_size

        # getting an object for the PyTorch Model class for Model Class
        # use CUDA if available
        self.pytorchmodel = torchModel(self.input_dim, self.output_dim)
        if torch.cuda.is_available():
            self.pytorchmodel.cuda()

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

        # PYTORCH
        # Critical number for early stopping
        self.num_epochs_we_want_to_train = 1

        # no of examples at each step/batch
        self.batch_size = 1000

    def trainloop(self, criterion, optimizer, train_data_iterator, steps):
        """
        # PYTORCH
        Trainloop function does the actual training of the model
        1) it gets the X, y from tensorflow dataset.
        2) convert X, y to CUDA
        3) trains the model with the Tesors for given no of steps.
        """

        for i in range(steps):
            images, labels = dataloading.get_torch_tensors(train_data_iterator)
            images = torch.Tensor(images)
            labels = torch.Tensor(labels)

            if torch.cuda.is_available():
                images = images.float().cuda()
                labels = labels.long().cuda()
            else:
                images = images.float()
                labels = labels.long()
            optimizer.zero_grad()

            log_ps = self.pytorchmodel(images)
            loss = criterion(log_ps, labels)
            loss.backward()
            optimizer.step()

    def train(self, dataset, remaining_time_budget=None):
        steps_to_train = self.get_steps_to_train(remaining_time_budget)
        if steps_to_train <= 0:
            utils.print_log(
                "Not enough time remaining for training. "
                + "Estimated time for training per step: {:.2f}, ".format(
                    self.estimated_time_per_step
                )
                + "but remaining time budget is: {:.2f}. ".format(remaining_time_budget)
                + "Skipping..."
            )
            self.done_training = True
        else:
            msg_est = ""
            if self.estimated_time_per_step:
                msg_est = "estimated time for this: " + "{:.2f} sec.".format(
                    steps_to_train * self.estimated_time_per_step
                )
            utils.print_log(
                "Begin training for another {} steps...{}".format(steps_to_train, msg_est)
            )

            train_start = time.time()

            # PYTORCH
            if not self.train_data_iterator:

                self.train_data_iterator = dataloading.input_function(
                    dataset,
                    self.default_num_frames,
                    self.default_image_size,
                    self.default_shuffle_buffer,
                    self.batch_size,
                    is_training=True,
                )
            # Training loop inside
            criterion = nn.NLLLoss()
            optimizer = torch.optim.Adam(self.pytorchmodel.parameters(), lr=1e-3)
            self.trainloop(
                criterion, optimizer, self.train_data_iterator, steps=steps_to_train
            )
            train_end = time.time()

            # Update for time budget managing
            train_duration = train_end - train_start
            self.total_train_time += train_duration
            self.cumulated_num_steps += steps_to_train
            self.estimated_time_per_step = (
                self.total_train_time / self.cumulated_num_steps
            )
            utils.print_log(
                "{} steps trained. {:.2f} sec used. ".format(
                    steps_to_train, train_duration
                )
                + "Now total steps trained: {}. ".format(self.cumulated_num_steps)
                + "Total time used for training: {:.2f} sec. ".format(
                    self.total_train_time
                )
                + "Current estimated time per step: {:.2e} sec.".format(
                    self.estimated_time_per_step
                )
            )

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
                (remaining_time_budget - tentative_estimated_time_test)
                / self.estimated_time_per_step
            )
            max_steps = max(max_steps, 1)
            if self.cumulated_num_tests < np.log(max_steps) / np.log(2):
                steps_to_train = int(
                    2 ** self.cumulated_num_tests
                )  # Double steps_to_train after each test
            else:
                steps_to_train = 0
        return steps_to_train

    def testloop(self, dataloader):
        """
        # PYTORCH
        testloop uses testdata to test the pytorch model and return onehot prediciton
        values.
        """
        preds = []
        with torch.no_grad():
            self.pytorchmodel.eval()
            for [images] in dataloader:
                if torch.cuda.is_available():
                    images = images.float().cuda()
                else:
                    images = images.float()
                log_ps = self.pytorchmodel(images)
                ps = torch.exp(log_ps)
                top_p, top_class = ps.topk(1, dim=1)
                preds.append(top_class.cpu().numpy())
        preds = np.concatenate(preds)
        onehot_preds = np.squeeze(np.eye(self.output_dim)[preds.reshape(-1)])
        return onehot_preds

    def choose_to_stop_early(self):
        """The criterion to stop further training (thus finish train/predict
        process).
        """
        # return self.cumulated_num_tests > 10 # Limit to make 10 predictions
        # return np.random.rand() < self.early_stop_proba
        batch_size = self.batch_size
        num_examples = self.metadata_.size()
        num_epochs = self.cumulated_num_steps * batch_size / num_examples
        utils.print_log("Model already trained for {} epochs.".format(num_epochs))

        # Train for at least certain number of epochs then stop
        return num_epochs > self.num_epochs_we_want_to_train

    def test(self, dataset, remaining_time_budget=None):
        if self.done_training:
            return None

        if self.choose_to_stop_early():
            utils.print_log("Oops! Choose to stop early for next call!")
            self.done_training = True
        test_begin = time.time()
        if (
            remaining_time_budget
            and self.estimated_time_test
            and self.estimated_time_test > remaining_time_budget
        ):
            utils.print_log(
                "Not enough time for test. "
                + "Estimated time for test: {:.2e}, ".format(self.estimated_time_test)
                + "But remaining time budget is: {:.2f}. ".format(remaining_time_budget)
                + "Stop train/predict process by returning None."
            )
            return None

        msg_est = ""
        if self.estimated_time_test:
            msg_est = "estimated time: {:.2e} sec.".format(self.estimated_time_test)
        utils.print_log("Begin testing...", msg_est)

        # PYTORCH
        testloader = dataloading.get_dataloader(
            dataset,
            batch_size=self.batch_size,
            default_num_frames=self.default_num_frames,
            default_image_size=self.default_image_size,
            train=False,
        )
        predictions = self.testloop(testloader)

        test_end = time.time()
        # Update some variables for time management
        test_duration = test_end - test_begin
        self.total_test_time += test_duration
        self.cumulated_num_tests += 1
        self.estimated_time_test = self.total_test_time / self.cumulated_num_tests
        utils.print_log(
            "[+] Successfully made one prediction. {:.2f} sec used. ".format(
                test_duration
            )
            + "Total time used for testing: {:.2f} sec. ".format(self.total_test_time)
            + "Current estimated time for test: {:.2e} sec.".format(
                self.estimated_time_test
            )
        )
        return predictions
