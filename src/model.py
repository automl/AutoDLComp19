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
import numpy as np

# Import the challenge algorithm (model) API from algorithm.py
import algorithm

import dataloading
import utils

import image.online_concrete
import image.online_meta
import image.models

# Disable tf device loggings
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


class Model(algorithm.Algorithm):
    def __init__(self, metadata):
        super(Model, self).__init__(metadata)
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

        # TODO(Danny): Document what metadata_ contains
        self.no_more_training = False
        self.output_dim = self.metadata_.get_output_size()

        utils.print_log("Metadata={}".format(self.metadata_.__dict__))

        try:
            self.config = utils.Config("config.hjson")
        except FileNotFoundError:
            self.config = utils.Config("src/config.hjson")

        utils.print_log("Config={}".format(self.config.__dict__))

        self.train_data_iterator = None
        self.model_input_sizes = None
        self.model = None

        if self.config.modality == "image":
            self.online_meta = image.online_meta.OnlineMeta(self.config, self.metadata_)
            self.online_concrete = image.online_concrete
        else:
            raise  # TODO(Danny): Some error message

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

    def _get_steps_to_train(self, remaining_time_budget):
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

    def train(self, dataset, remaining_time_budget=None):
        steps_to_train = self._get_steps_to_train(remaining_time_budget)
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
            return

        msg_est = ""
        if self.estimated_time_per_step:
            msg_est = "estimated time for this: " + "{:.2f} sec.".format(
                steps_to_train * self.estimated_time_per_step
            )
        utils.print_log(
            "Begin training for another {} steps...{}".format(steps_to_train, msg_est)
        )

        train_start = time.time()

        # AUTODL START
        self.model, model_input_sizes = self.online_meta.select_model()
        # TODO(Danny): make initialiazation work here. Currently there is a problem since
        # the last layer has a different shape than in the online parameters.
        # self.model = self.online_meta.initialize_model(self.model)
        unfrozen_parameters = self.online_meta.select_unfrozen_parameter(self.model)

        model_input_sizes_changed = model_input_sizes != self.model_input_sizes
        self.model_input_sizes = model_input_sizes
        if not self.train_data_iterator or model_input_sizes_changed:
            self.train_data_iterator = dataloading.input_function(
                dataset, self.config, self.model_input_sizes, is_training=True
            )
        self.online_concrete.trainloop(
            self.model,
            unfrozen_parameters,
            self.train_data_iterator,
            self.config,
            steps=steps_to_train,
        )

        # AUTODL END
        train_end = time.time()

        # Update for time budget managing
        train_duration = train_end - train_start
        self.total_train_time += train_duration
        self.cumulated_num_steps += steps_to_train
        self.estimated_time_per_step = self.total_train_time / self.cumulated_num_steps
        utils.print_log(
            "{} steps trained. {:.2f} sec used. ".format(steps_to_train, train_duration)
            + "Now total steps trained: {}. ".format(self.cumulated_num_steps)
            + "Total time used for training: {:.2f} sec. ".format(self.total_train_time)
            + "Current estimated time per step: {:.2e} sec.".format(
                self.estimated_time_per_step
            )
        )

    def _choose_to_stop_early(self):
        """The criterion to stop further training (thus finish train/predict
        process).
        """
        # return self.cumulated_num_tests > 10 # Limit to make 10 predictions
        # return np.random.rand() < self.early_stop_proba
        batch_size = self.config.batch_size
        num_examples = self.metadata_.size()
        num_epochs = self.cumulated_num_steps * batch_size / num_examples
        utils.print_log("Model already trained for {} epochs.".format(num_epochs))

        # Train for at least certain number of epochs then stop
        return num_epochs > self.config.num_epochs_we_want_to_train

    def test(self, dataset, remaining_time_budget=None):
        if self.done_training:
            return None

        if self._choose_to_stop_early():
            utils.print_log("Oops! Choose to stop early for next call!")
            self.done_training = True
        test_begin = time.time()
        not_enough_time_for_test = (
            remaining_time_budget
            and self.estimated_time_test
            and self.estimated_time_test > remaining_time_budget
        )
        if not_enough_time_for_test:
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
        # TODO(Danny): Only load testset once if it fits nicely in 24GB
        testloader = dataloading.get_dataloader(
            dataset, self.config, self.model_input_sizes, train=False
        )
        predictions = self.online_concrete.testloop(
            self.model, testloader, self.output_dim
        )

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
