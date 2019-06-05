# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#            http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Modified by: Zhengying Liu, Isabelle Guyon

"""An example of code submission for the AutoDL challenge.

It implements 3 compulsory methods: __init__, train, and test.
model.py follows the template of the abstract class algorithm.py found
in folder AutoDL_ingestion_program/.

To create a valid submission, zip model.py together with an empty
file called metadata (this just indicates your submission is a code submission
and has nothing to do with the dataset metadata.
"""

import tensorflow as tf
import os

tf.logging.set_verbosity(tf.logging.ERROR)

# Import the challenge algorithm (model) API from algorithm.py
import algorithm

# Utility packages
import time
import datetime
import numpy as np
import torch
import torch.nn as nn

# All importings by christopher (some may be unnescessary)
# Imports ################################################################
import sys
import time
import os.path

# Import Keras
from keras.layers import (
    Dense,
    Flatten,
    Dropout,
    ZeroPadding3D,
    Input,
    Activation,
    BatchNormalization,
    add,
    Reshape,
    GlobalAveragePooling2D,
)
from keras.models import Sequential, load_model, Model
from keras.optimizers import Adam
from keras.layers.convolutional import Conv3D, MaxPooling3D
import tensorflow as tf

# Imports for Bohb
import numpy as np
import datetime
import logging
import pickle

# HPOB
from hpbandster.core.worker import Worker
import hpbandster.core.nameserver as hpns
import hpbandster.core.result as hpres
from hpbandster.optimizers import BOHB
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH

################################################################
np.random.seed(42)

from sklearn.linear_model import LinearRegression


# Params ################################################################
# Nameserver params
nic_name = "lo"
port = 0
run_id = "bohb_run_1"
# BOHB params
dataset = "UCF101"
n_bohb_iterations = 20
min_budget = 0.3
max_budget = 0.3
eta = 6
n_workers = 1  # Number of paralell workers
num_gpu_workers = 1  # Number of GPU workers
continue_run = False  # args.continue_run # load previous run
continue_path = None
# Define where to save results
working_dir = os.curdir
currentDT = datetime.datetime.now()
curr_time = [
    currentDT.day,
    currentDT.month,
    currentDT.hour,
    currentDT.minute,
    currentDT.second,
]
result_dir = os.path.join(
    working_dir,
    "res/bohb_results_for_{}_niter:{}_Date:{}_{}_{}:{}:{}/".format(
        dataset, n_bohb_iterations, *curr_time
    ),
)
try:
    os.mkdir(os.path.join(working_dir, "res"))
except:
    pass
os.mkdir(result_dir)
result_file = "{}bohb_result.pkl".format(result_dir)
initial_configs_file = os.path.join(
    working_dir, "/random_init/bohb_{}_result.pkl".format(dataset)
)
# logging.basicConfig(level=logging.INFO) # DEBUG
# tf.logging.set_verbosity(tf.logging.ERROR)


class Model(algorithm.Algorithm):
    """Fully connected neural network with no hidden layer."""

    def __init__(self, metadata):
        super(Model, self).__init__(metadata)

        self.done_training = False
        self.metadata = metadata
        self.num_examples_train = 9537
        self.num_examples_test = 3783  # 3783

        self.output_dim = self.metadata_.get_output_size()
        self.num_examples_train = self.metadata_.size()
        # Get dataset name.
        self.dataset_name = self.metadata_.get_dataset_name().split("/")[-2].split(".")[0]
        print_log(
            "The dataset {} has {} training examples and {} classes.".format(
                self.dataset_name, self.num_examples_train, self.output_dim
            )
        )

        # Boolean True if example have fixed size
        row_count, col_count = self.metadata_.get_matrix_size(0)
        self.fixed_matrix_size = row_count > 0 and col_count > 0
        sequence_size = self.metadata_.get_sequence_size()
        self.fixed_sequence_size = sequence_size > 0

        # Change to True if you want to show device info at each operation
        log_device_placement = False
        self.config = tf.ConfigProto(log_device_placement=log_device_placement)

        # Attributes for preprocessing
        self.default_image_size = (128, 96)
        self.default_num_frames = 32
        self.default_shuffle_buffer = 100
        # Set batch size (for both training and testing)
        self.batch_size = 32
        # model parameters
        self.model = None

        # Attributes for managing time budget
        # Cumulated number of training steps
        self.birthday = time.time()
        self.train_begin_times = []
        self.test_begin_times = []
        self.li_steps_to_train = []
        self.li_cycle_length = []
        self.li_estimated_time = []
        self.time_estimator = LinearRegression()
        self.done_training = False
        # Critical number for early stopping
        # Depends on number of classes (output_dim)
        # see the function self.choose_to_stop_early() below for more details
        self.num_epochs_we_want_to_train = max(40, self.output_dim)

    def train(self, dataset, remaining_time_budget=None):
        """Train this algorithm on the tensorflow |dataset|.

        This method will be called REPEATEDLY during the whole training/predicting
        process. So your `train` method should be able to handle repeated calls and
        hopefully improve your model performance after each call.

        Args:
            dataset: a `tf.data.Dataset` object. Each of its examples is of the form
                        (example, labels)
                    where `example` is a dense 4-D Tensor of shape
                        (sequence_size, row_count, col_count, num_channels)
                    and `labels` is a 1-D Tensor of shape
                        (output_dim,).
                    Here `output_dim` represents number of classes of this
                    multilabel classification task.

                    IMPORTANT: some of the dimensions of `example` might be `None`,
                    which means the shape on this dimension might be variable. In this
                    case, some preprocessing technique should be applied in order to
                    feed the training of a neural network. For example, if an image
                    dataset has `example` of shape
                        (1, None, None, 3)
                    then the images in this datasets may have different sizes. On could
                    apply resizing, cropping or padding in order to have a fixed size
                    input tensor.

            remaining_time_budget: time remaining to execute train(). The method
                    should keep track of its execution time to avoid exceeding its time
                    budget. If remaining_time_budget is None, no time budget is imposed.
        """
        print_log("!!! TRAINING OWN !!!")
        # Check if we still can train ##################################################
        # if self.done_training:
        #    return
        train_start = time.time()
        self.train_begin_times.append(time.time())
        # if len(self.train_begin_times) >= 2:
        #    cycle_length = self.train_begin_times[-1] - self.train_begin_times[-2]
        #    self.li_cycle_length.append(cycle_length)

        # Get number of steps to train according to some strategy
        # steps_to_train = self.get_steps_to_train(remaining_time_budget)

        # if steps_to_train <= 0:
        #    self.done_training = True
        dataset = dataset.map(
            lambda *x: (self.preprocess_tensor_4d(x[0]), x[1]), num_parallel_calls=11
        )
        if True:
            if len(self.li_estimated_time) > 0:
                estimated_duration = self.li_estimated_time[-1]
                estimated_end_time = time.ctime(int(time.time() + estimated_duration))

            # Prepare input function for training
            # dataset = dataset.map(lambda *x: (self.preprocess_tensor_4d(x[0], True), x[1]))
            # dataset = dataset.shuffle(self.default_shuffle_buffer)
            # dataset = dataset.batch(batch_size=self.batch_size, drop_remainder=True)
            # dataset = dataset.cache(filename='tempdata')
            # TODO: better way but preprocessing needs other function
            # dataset = dataset.apply(tf.contrib.data.map_and_batch(
            #                map_func=lambda *x: (self.preprocess_tensor_4d(x[0]), x[1]),
            #                batch_size=self.batch_size))
            # examples, labels, _ = self.convert_dataset_to_pytorch(dataset, True)
            # Nameserver ################################################################
            # If nameserver fails, with try it reconnects for debugging uncomented
            # try:
            if continue_run == True:
                # TODO: continueing still not working need to set file and folder correctly
                previous_run = hpres.logged_results_to_HBS_result(directory=continue_path)
                result_logger = hpres.json_result_logger(
                    directory=continue_path, overwrite=False
                )
            else:
                result_logger = hpres.json_result_logger(
                    directory=result_dir, overwrite=True
                )

            # Start a nameserver
            host = hpns.nic_name_to_host(nic_name)
            ns = hpns.NameServer(
                run_id=run_id, host=host, port=port, working_directory=working_dir
            )
            ns_host, ns_port = ns.start()
            time.sleep(2)  # Wait for nameserver
            # Training ################################################################
            # Start local workers
            workers = []
            # here we have a single worker for small datasets multiple workers
            # possible with for i in number of workers and change id=1 to i
            worker = AUTOMLWorker(
                ######################################################
                # Nameserver params
                run_id=run_id,
                host=host,
                nameserver=ns_host,
                nameserver_port=ns_port,
                id=1,
                # timeout=120, sleep_interval = 0.5
            )
            worker.setup(  ######################################################
                # Train Params
                dataset=dataset,
                output_dim=self.output_dim,
                num_examples_train=self.num_examples_train,
                dataset_name=self.dataset_name,
                fixed_matrix_size=self.fixed_matrix_size,
                fixed_sequence_size=self.fixed_sequence_size,
                batch_size=self.batch_size,
                default_image_size=self.default_image_size,
                default_num_frames=self.default_num_frames,
                default_shuffle_buffer=self.default_shuffle_buffer,
                train=True,
                test_val_split=0.9,
            )
            worker.run(background=True)
            workers.append(worker)
            # Create HPO object and run it
            bohb = BOHB(
                configspace=workers[0].get_configspace(),
                eta=eta,
                run_id=run_id,
                host=host,
                nameserver=ns_host,
                nameserver_port=ns_port,
                result_logger=result_logger,
                min_budget=min_budget,
                max_budget=max_budget,
                # min_points_in_model=5,
                # continue optimizing
                # previous_result = random_initial_configurations,
                # finetune bohb
                # random_fraction=0.20,
                # top_n_percent=20)
            )
            # Get results
            result = bohb.run(n_iterations=n_bohb_iterations, min_n_workers=n_workers)
            print("Write result to file {}".format(result_file))
            with open(result_file, "wb") as f:
                pickle.dump(result, f)
            bohb.shutdown(shutdown_workers=True)
            ns.shutdown()
            # Belongs to nameserver try comented for debugging
            # finally:
            #    print('errors occured')
            #    bohb.shutdown(shutdown_workers=True)
            #    ns.shutdown()

            if True:
                # get all executed runs
                all_runs = result.get_all_runs()
                # get the 'dict' that translates config ids to the actual configurations
                id2conf = result.get_id2config_mapping()
                # Here is how you get he incumbent (best configuration)
                inc_id = result.get_incumbent_id()
                # let's grab the run on the highest budget
                inc_runs = result.get_runs_by_id(inc_id)
                inc_run = inc_runs[-1]
                # We have access to all information: the config, the loss observed during
                # optimization, and all the additional information
                inc_loss = inc_run.loss
                self.inc_config = id2conf[inc_id]["config"]
                # inc_test_loss = inc_run.info['test accuracy']
                # Print results
                print("Best found configuration:")
                print(self.inc_config)
                print("It achieved accuracies of %f (validation)." % (1 - inc_loss))
            config = self.inc_config
            worker.train_single_network(config, perc_steps_per_epoch=0.05)
            self.classifier = worker.classifier

            train_end = time.time()
            print("\n\n     DONE! \n time used: {}".format(train_end - train_start))

    def test(self, dataset, remaining_time_budget=None):
        """Test this algorithm on the tensorflow |dataset|.

        Args:
            Same as that of `train` method, except that the `labels` will be empty.
        Returns:
            predictions: A `numpy.ndarray` matrix of shape (sample_count, output_dim).
                    here `sample_count` is the number of examples in this dataset as test
                    set and `output_dim` is the number of labels to be predicted. The
                    values should be binary or in the interval [0,1].
                    IMPORTANT: if returns None, this means that the algorithm
                    chooses to stop training, and the whole train/test will stop. The
                    performance of the last prediction will be used to compute area under
                    learning curve.
        """
        print_log("!!! TESTING OWN !!!")
        # if self.done_training:
        #    return None

        # self.test_begin_times.append(time.time())

        # if self.choose_to_stop_early():
        #    self.done_training = True
        dataset = dataset.map(lambda *x: (self.preprocess_tensor_4d(x[0]), x[1]))
        # examples, labels, _ = self.convert_dataset_to_pytorch(dataset, False)
        # examples = examples.view(-1,self.nb_in)
        if continue_run == True:
            # TODO: continueing still not working need to set file and folder correctly
            previous_run = hpres.logged_results_to_HBS_result(directory=continue_path)
            result_logger = hpres.json_result_logger(
                directory=continue_path, overwrite=False
            )
        else:
            result_logger = hpres.json_result_logger(directory=result_dir, overwrite=True)

            # Start a nameserver
            host = hpns.nic_name_to_host(nic_name)
            ns = hpns.NameServer(
                run_id=run_id, host=host, port=port, working_directory=working_dir
            )
            ns_host, ns_port = ns.start()
            time.sleep(2)  # Wait for nameserver
        workers = []
        # here we have a single worker for small datasets multiple workers
        # possible with for i in number of workers and change id=1 to i
        worker = AUTOMLWorker(
            ######################################################
            # Nameserver params
            run_id=run_id,
            host=host,
            nameserver=ns_host,
            nameserver_port=ns_port,
            id=1,
            # timeout=120, sleep_interval = 0.5
        )
        worker.setup(  ######################################################
            # Train Params
            dataset=dataset,
            output_dim=self.output_dim,
            num_examples_train=self.num_examples_train,
            dataset_name=self.dataset_name,
            fixed_matrix_size=self.fixed_matrix_size,
            fixed_sequence_size=self.fixed_sequence_size,
            batch_size=self.batch_size,
            default_image_size=self.default_image_size,
            default_num_frames=self.default_num_frames,
            default_shuffle_buffer=self.default_shuffle_buffer,
            train=False,
        )
        worker.run(background=True)
        workers.append(worker)

        test_results = self.classifier.predict(input_fn=worker.test_input_fn)

        predictions = [x["probabilities"] for x in test_results]
        predictions = np.array(predictions)
        return predictions

    ##############################################################################
    #### Above 3 methods (__init__, train, test) should always be implemented ####
    ##############################################################################

    def init_model(self, examples, labels):
        self.nb_in = np.prod(examples.shape[1:])
        nb_out = labels.shape[1]

        return nn.Sequential(nn.Linear(self.nb_in, nb_out))

    def convert_dataset_to_pytorch(self, dataset, is_training):
        sess = tf.Session(config=self.config)

        # fix dataset size
        dataset = dataset.map(lambda *x: (self.preprocess_tensor_4d(x[0], False), x[1]))

        if is_training:
            # Shuffle input examples
            dataset = dataset.shuffle(buffer_size=self.default_shuffle_buffer)
            # Convert to RepeatDataset to train for several epochs
            dataset = dataset.repeat()

        iterator = dataset.make_one_shot_iterator()
        next_element = iterator.get_next()

        result = list()
        finished = False

        try:
            if is_training:
                for i in range(self.batch_size):
                    result.append(sess.run(next_element))
            else:
                while True:
                    result.append(sess.run(next_element))
        except tf.errors.OutOfRangeError:
            finished = True

        examples = torch.from_numpy(np.stack((list(zip(*result))[0])))
        labels = torch.from_numpy(np.stack((list(zip(*result))[1])))

        return examples, labels, finished

    def get_steps_to_train(self, remaining_time_budget):
        """Get number of steps for training according to `remaining_time_budget`.

        The strategy is:
            1. If no training is done before, train for 10 steps (ten batches);
            2. Otherwise, double the number of steps to train. Estimate the time
                 needed for training and test for this number of steps;
            3. Compare to remaining time budget. If not enough, stop. Otherwise,
                 proceed to training/test and go to step 2.
        """
        if remaining_time_budget is None:  # This is never true in the competition anyway
            remaining_time_budget = 1200  # if no time limit is given, set to 20min

        # for more conservative estimation
        remaining_time_budget = min(
            remaining_time_budget - 60, remaining_time_budget * 0.95
        )

        if len(self.li_steps_to_train) == 0:
            return 10
        else:
            steps_to_train = self.li_steps_to_train[-1] * 2

            # Estimate required time using linear regression
            X = np.array(self.li_steps_to_train).reshape(-1, 1)
            Y = np.array(self.li_cycle_length)
            self.time_estimator.fit(X, Y)
            X_test = np.array([steps_to_train]).reshape(-1, 1)
            Y_pred = self.time_estimator.predict(X_test)

            estimated_time = Y_pred[0]
            self.li_estimated_time.append(estimated_time)

            if estimated_time >= remaining_time_budget:
                return 0
            else:
                return steps_to_train

    def choose_to_stop_early(self):
        """The criterion to stop further training (thus finish train/predict
        process).
        """
        batch_size = self.batch_size
        num_examples = self.metadata_.size()
        num_epochs = sum(self.li_steps_to_train) * batch_size / num_examples
        print_log("Model already trained for {:.4f} epochs.".format(num_epochs))
        return (
            num_epochs > self.num_epochs_we_want_to_train
        )  # Train for at least certain number of epochs then stop

    def preprocess_tensor_4d(self, tensor_4d):
        """Preprocess a 4-D tensor (only when some dimensions are `None`, i.e.
        non-fixed). The output tensor wil have fixed, known shape.

        Args:
            tensor_4d: A Tensor of shape
                    [sequence_size, row_count, col_count, num_channels]
                    where some dimensions might be `None`.
        Returns:
            A 4-D Tensor with fixed, known shape.
        """
        tensor_4d_shape = tensor_4d.shape
        logger.info("Tensor shape before preprocessing: {}".format(tensor_4d_shape))

        if tensor_4d_shape[0] > 0 and tensor_4d_shape[0] < 10:
            num_frames = tensor_4d_shape[0]
        else:
            num_frames = self.default_num_frames
        if tensor_4d_shape[1] > 0:
            new_row_count = tensor_4d_shape[1]
        else:
            new_row_count = self.default_image_size[0]
        if tensor_4d_shape[2] > 0:
            new_col_count = tensor_4d_shape[2]
        else:
            new_col_count = self.default_image_size[1]
        new_row_count = self.default_image_size[0]
        new_col_count = self.default_image_size[1]

        if not tensor_4d_shape[0] > 0 or True:
            logger.info(
                "Detected that examples have variable sequence_size, will "
                + "randomly crop a sequence with num_frames = "
                + "{}".format(num_frames)
            )
            tensor_4d = crop_time_axis(tensor_4d, num_frames=num_frames)
        if not tensor_4d_shape[1] > 0 or not tensor_4d_shape[2] > 0 or True:
            logger.info(
                "Detected that examples have variable space size, will "
                + "resize space axes to (new_row_count, new_col_count) = "
                + "{}".format((new_row_count, new_col_count))
            )
            tensor_4d = resize_space_axes(
                tensor_4d, new_row_count=new_row_count, new_col_count=new_col_count
            )
        logger.info("Tensor shape after preprocessing: {}".format(tensor_4d.shape))
        return tensor_4d


def sigmoid_cross_entropy_with_logits(labels=None, logits=None):
    """Re-implementation of this function:
        https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits

    Let z = labels, x = logits, then return the sigmoid cross entropy
        max(x, 0) - x * z + log(1 + exp(-abs(x)))
    (Then sum over all classes.)
    """
    labels = tf.cast(labels, dtype=tf.float32)
    relu_logits = tf.nn.relu(logits)
    exp_logits = tf.exp(-tf.abs(logits))
    sigmoid_logits = tf.log(1 + exp_logits)
    element_wise_xent = relu_logits - labels * logits + sigmoid_logits
    return tf.reduce_sum(element_wise_xent)


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
    assert len(tensor_shape) > 1
    num_entries = 1
    for i in tensor_shape[1:]:
        num_entries *= int(i)
    return num_entries


def crop_time_axis(tensor_4d, num_frames, begin_index=None):
    """Given a 4-D tensor, take a slice of length `num_frames` on its time axis.

    Args:
        tensor_4d: A Tensor of shape
                [sequence_size, row_count, col_count, num_channels]
        num_frames: An integer representing the resulted chunk (sequence) length
        begin_index: The index of the beginning of the chunk. If `None`, chosen
            randomly.
    Returns:
        A Tensor of sequence length `num_frames`, which is a chunk of `tensor_4d`.
    """
    # pad sequence if not long enough
    pad_size = tf.maximum(num_frames - tf.shape(tensor_4d)[1], 0)
    padded_tensor = tf.pad(tensor_4d, ((0, pad_size), (0, 0), (0, 0), (0, 0)))

    # If not given, randomly choose the beginning index of frames
    if not begin_index:
        maxval = tf.shape(padded_tensor)[0] - num_frames + 1
        begin_index = tf.random.uniform([1], minval=0, maxval=maxval, dtype=tf.int32)
        begin_index = tf.stack([begin_index[0], 0, 0, 0], name="begin_index")

    sliced_tensor = tf.slice(
        padded_tensor, begin=begin_index, size=[num_frames, -1, -1, -1]
    )

    return sliced_tensor


def resize_space_axes(tensor_4d, new_row_count, new_col_count):
    """Given a 4-D tensor, resize space axes to have target size.

    Args:
        tensor_4d: A Tensor of shape
                [sequence_size, row_count, col_count, num_channels].
        new_row_count: An integer indicating the target row count.
        new_col_count: An integer indicating the target column count.
    Returns:
        A Tensor of shape [sequence_size, target_row_count, target_col_count].
    """
    resized_images = tf.image.resize_images(
        tensor_4d, size=(new_row_count, new_col_count)
    )
    return resized_images


def print_log(*content):
    """Logging function. (could've also used `import logging`.)"""
    now = datetime.datetime.now().strftime("%y-%m-%d %H:%M:%S")
    print("MODEL INFO: " + str(now) + " ", end="")
    print(*content)


# Models ################################################################
class VariousModels:
    def __init__(
        self,
        output_dim,
        model_name,
        seq_length,
        image_shape,
        config,
        # saved_model=None,
        features_length=2048,
        lr=1e-5,
        dc=1e-6,
    ):
        """
        `model_name` = one of:
            c3d_reduced
        `classes` = the number of classes to predict
        `seq_length` = the length of our video sequences
        """
        # Set defaults.
        self.seq_length = seq_length
        self.load_model = load_model
        self.output_dim = output_dim

        # Set the metrics. Only use top k if there's a need.
        metrics = ["accuracy"]
        # if self.classes >= 10:
        #    metrics.append('top_k_categorical_accuracy')
        # Get the appropriate model.
        # if self.saved_model is not None:
        #    print("Loading model %s" % self.saved_model)
        #    self.model = load_model(self.saved_model)
        if model_name == "c3d_reduced":
            print("Loading C3D")
            self.input_shape = (seq_length, *image_shape)
            self.model = self.c3d_reduced(
                neurons=config["neurons"], dropout=config["dropout"]
            )
        else:
            print("Unknown network.")
            sys.exit()

        # Now compile the network.
        optimizer = Adam(lr=config["lr"], decay=config["weight_decay"])
        self.model.compile(
            loss="categorical_crossentropy", optimizer=optimizer, metrics=metrics
        )

        print(self.model.summary())

    def c3d_reduced(self, neurons=2042, dropout=0.5):
        """
        Build a 3D convolutional network, aka C3D.
            https://arxiv.org/pdf/1412.0767.pdf

        With thanks:
            https://gist.github.com/albertomontesg/d8b21a179c1e6cca0480ebdf292c34d2
        """
        model = Sequential()

        # Conv3D(32, (3, 3, 3), activation="relu", name="conv1", input_shape=(10, 80, 8..., strides=(1, 1, 1), padding="same")
        # 1st layer group
        model.add(
            Conv3D(
                32,
                (3, 3, 3),
                activation="relu",
                padding="same",
                name="conv1",
                input_shape=self.input_shape,
                strides=(1, 1, 1),
            )
        )
        model.add(
            MaxPooling3D(
                pool_size=(4, 4, 4), strides=(2, 2, 2), padding="valid", name="pool1"
            )
        )
        # 2nd layer group
        model.add(
            Conv3D(
                32,
                (3, 3, 3),
                activation="relu",
                padding="same",
                name="conv2",
                strides=(1, 1, 1),
            )
        )
        model.add(
            MaxPooling3D(
                pool_size=(4, 4, 4), strides=(2, 2, 2), padding="valid", name="pool2"
            )
        )
        # 3rd layer group
        # model.add(Conv3D(64, (3, 3, 3), activation="relu", padding="same",
        #                 name="conv3a", strides=(1, 1, 1)))
        # model.add(MaxPooling3D(pool_size=(3, 4, 4), strides=(2, 2, 2),
        #                       padding='valid', name='pool3'))
        # model.add(Conv3D(64, (3, 3, 3), activation="relu", padding="same",
        #                 name="conv3b", strides=(1, 1, 1)))
        # model.add(MaxPooling3D(pool_size=(2, 4, 4), strides=(1, 2, 2),
        #                       padding='valid', name='pool4'))
        model.add(Flatten())

        # FC layers group
        # model.add(Dense(neurons, activation='relu', name='fc7'))
        # model.add(Dropout(dropout))
        model.add(Dense(self.output_dim, activation="softmax"))

        return model


# AUTOML Worker ################################################################
class AUTOMLWorker(Worker):
    """ Worker for BOHB optimizer """

    def __init__(self, *args, **kwargs):
        """
        initializes variables dataset and device
        """
        super().__init__(*args, **kwargs)

        self.warm_start_dir = "/media/human/18e40163-a195-45fc-a594-42ebbc89ff6c/ML/autodl_starting_kit_stable/checkpoints/16.05.2019 01:39"
        self.model_dir = self.warm_start_dir
        self.model_dir = None
        # Get model function from class method below

    def setup(self, *args, **kwargs):
        """ Fix this, to use init, but i don't know how"""
        self.output_dim = kwargs["output_dim"]
        self.num_examples_train = kwargs["num_examples_train"]
        self.dataset_name = kwargs["dataset_name"]
        self.default_shuffle_buffer = kwargs["default_shuffle_buffer"]
        self.model_name = "c3d_reduced"
        self.batch_size = kwargs["batch_size"]
        # Get the data and process it.
        self.data_type = "video"
        self.fixed_sequence_size = kwargs["fixed_sequence_size"]
        self.default_num_frames = kwargs["default_num_frames"]
        self.seq_length = self.default_num_frames
        # Get dimensions out of image size
        self.fixed_matrix_size = kwargs["fixed_matrix_size"]
        self.default_image_size = kwargs["default_image_size"]
        self.image_shape = (*self.default_image_size, 3)
        self.features = (self.default_num_frames, *self.image_shape)
        # Get samples per epoch.
        self.steps_per_epoch = (self.num_examples_train) // self.batch_size
        self.test_val_split = kwargs["test_val_split"]
        # Get generators. TODO: train/val split
        self.dataset = kwargs["dataset"]
        # self.val_generator = data.frame_generator(self.batch_size, 'test', self.data_type)

        # Preprocess dataset
        if kwargs["train"]:
            # dataset = dataset.map(lambda *x: (self.preprocess_tensor_4d(x[0]), x[1]))
            self.train_input_fn = lambda: self.input_function(
                self.dataset, is_training=True, train=True
            )
            self.val_input_fn = lambda: self.input_function(
                self.dataset, is_training=True, train=False
            )
        else:
            self.test_input_fn = lambda: self.input_function(
                self.dataset, is_training=False
            )

    def train(self, config, perc_steps_per_epoch=1.0):

        time_start = time.time()

        model_fn = self.model_fn
        # Classifier using model_fn
        run_config = tf.estimator.RunConfig(
            save_summary_steps=None, save_checkpoints_secs=None
        )
        classifier = tf.estimator.Estimator(
            model_fn=model_fn, params=config, model_dir=self.model_dir, config=run_config
        )
        # calculate maximal number of batches to train times budget
        steps_train = int(
            perc_steps_per_epoch * (self.steps_per_epoch * self.test_val_split)
        )
        logger.info("\n Train for {} steps\n".format(steps_train))
        classifier.train(input_fn=self.train_input_fn, steps=steps_train)
        # evaluate classifier on full eval dataset
        steps_val = int((self.steps_per_epoch * (1 - self.test_val_split)))
        logger.info("\n Evaluate for {} steps\n".format(steps_val))
        result = classifier.evaluate(input_fn=self.val_input_fn, steps=steps_val)

        time_end = time.time()
        logger.info(
            "\n time To train: {}\n result: {}\n".format(time_end - time_start, result)
        )

        return result["accuracy"]

    # Nescessary function for improving bohb #########################################
    def compute(self, config, budget, *args, **kwargs):
        """
        Train a 3D-CNN with parameters config and on budget * epoch
        """
        # Training params
        result = self.train(config=config, perc_steps_per_epoch=budget)
        return {
            "loss": 1 - float(result),  # .history['val_acc'][0],
            "info": {},  # result.history
        }

    def train_single_network(self, config, perc_steps_per_epoch=1.0):

        model_fn = self.model_fn
        self.classifier = tf.estimator.Estimator(
            model_fn=model_fn,
            params=config,
            # model_dir=self.model_dir,
        )
        # Start training
        steps = int(perc_steps_per_epoch * self.steps_per_epoch)
        logger.info("\n train for {} steps\n".format(steps))

        self.classifier.train(input_fn=self.train_input_fn, steps=steps)

    # Config Space ################################################################
    def get_configspace(self):
        """ Define a conditional hyperparameter search-space with parameters from:
            ################################################################
            USED CONFIGSPACE
            ################################################################
            ### TRAINING ###
            lr:                          1e-7 to 0.5; 0.001; (log, float)
            ### ADAM ###
            weight_decay:            5e-7 to 0.05; 0.0005; (cond, log, float)
            ################################################################
            ### ARCHITECTURE ###
            ### FC Layers ###
            neurons:                   200 to 1000; 500; (log, int)
            dropout:            0 to 0.95; 0.5; (float)
        """
        cs = CS.ConfigurationSpace()
        ################################################################
        # TRAINING
        lr = CSH.UniformFloatHyperparameter(
            "lr", 1e-7, 0.5, default_value=0.001, log=True
        )
        weight_decay = CSH.UniformFloatHyperparameter(
            "weight_decay", 5e-7, 0.05, default_value=0.0005
        )
        ### FC Layers ###
        neurons = CSH.UniformIntegerHyperparameter("neurons", 64, 300, log=True)
        dropout = CSH.UniformFloatHyperparameter("dropout", 0, 0.95, default_value=0.5)
        ##########################
        cs.add_hyperparameters(
            [
                lr,
                # weight_decay,
                neurons,
                dropout,
            ]
        )
        return cs

    def model_fn(self, features, labels, mode, params):
        """Auto-Scaling 3D CNN model.

        For more information on how to write a model function, see:
            https://www.tensorflow.org/guide/custom_estimators#write_a_model_function
        """
        input_layer = features
        # Replace missing values by 0
        hidden_layer = tf.where(
            tf.is_nan(input_layer), tf.zeros_like(input_layer), input_layer
        )
        # logger.info("input layer : {}".format(hidden_layer))

        # Repeatedly apply 3D CNN, followed by 3D max pooling
        # until the hidden layer has reasonable number of entries
        REASONABLE_NUM_ENTRIES = 1000
        num_filters = 8  # The number of filters is fixed
        filter_counter = 0
        while True:
            filter_counter += 1
            shape = hidden_layer.shape
            kernel_size = [min(3, shape[1]), min(3, shape[2]), min(3, shape[3])]
            hidden_layer = tf.layers.conv3d(
                inputs=hidden_layer,
                filters=num_filters * filter_counter,
                kernel_size=kernel_size,
            )
            # logger.info("hidden layer before pooling: {}".format(hidden_layer))
            shape = hidden_layer.shape
            kernel_size = [min(3, shape[1]), min(3, shape[2]), min(3, shape[3])]
            hidden_layer = tf.layers.conv3d(
                inputs=hidden_layer,
                filters=num_filters * filter_counter,
                kernel_size=kernel_size,
            )
            # logger.info("hidden layer before pooling: {}".format(hidden_layer))
            if shape[1] > 3:
                pool_size = [min(3, shape[1]), min(4, shape[2]), min(4, shape[3])]
            else:
                pool_size = [1, min(4, shape[2]), min(4, shape[3])]
            hidden_layer = tf.layers.max_pooling3d(
                inputs=hidden_layer,
                pool_size=pool_size,
                strides=[
                    min(2, pool_size[0]),
                    min(3, pool_size[1]),
                    min(3, pool_size[2]),
                ],
                padding="valid",
                data_format="channels_last",
            )
            # logger.info("hidden layer after pooling: {}".format(hidden_layer))
            if (
                filter_counter == 2
            ):  # get_num_entries(hidden_layer) < REASONABLE_NUM_ENTRIES or
                break

        hidden_layer = tf.layers.flatten(hidden_layer)
        hidden_layer = tf.layers.dense(
            inputs=hidden_layer, units=params["neurons"], activation=tf.nn.relu
        )
        hidden_layer = tf.layers.dropout(
            inputs=hidden_layer,
            rate=params["dropout"],
            training=mode == tf.estimator.ModeKeys.TRAIN,
        )

        logits = tf.layers.dense(inputs=hidden_layer, units=self.output_dim)
        sigmoid_tensor = tf.nn.sigmoid(logits, name="sigmoid_tensor")

        predictions = {
            # Generate predictions (for PREDICT and EVAL mode)
            "classes": tf.argmax(input=logits, axis=1),
            # "classes": binary_predictions,
            # Add `sigmoid_tensor` to the graph. It is used for PREDICT and by the
            # `logging_hook`.
            "probabilities": sigmoid_tensor,
        }

        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

        # Calculate Loss (for both TRAIN and EVAL modes)
        # For multi-label classification, a correct loss is sigmoid cross entropy
        loss = sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)
        tf.losses.add_loss(loss, loss_collection=tf.GraphKeys.LOSSES)
        tf.summary.scalar("loss", loss)

        # Configure the Training Op (for TRAIN mode)
        if mode == tf.estimator.ModeKeys.TRAIN:
            optimizer = tf.train.AdamOptimizer(learning_rate=params["lr"])
            train_op = optimizer.minimize(
                loss=loss, global_step=tf.train.get_global_step()
            )
            return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

        # Add evaluation metrics (for EVAL mode)
        assert mode == tf.estimator.ModeKeys.EVAL
        eval_metric_ops = {
            "accuracy": tf.metrics.accuracy(
                labels=tf.argmax(input=labels, axis=1), predictions=predictions["classes"]
            )
        }
        return tf.estimator.EstimatorSpec(
            mode=mode, loss=loss, eval_metric_ops=eval_metric_ops
        )

    def input_function(self, dataset, is_training, train=None):
        """For training, use twice, one time with is_training=True, train=True,
        to return train iterator and with train=False for validation iterator
        for final testing use is_training=False to return full dataset iterator unshuffled
        """
        # Train val split
        if is_training:
            # Convert to RepeatDataset to train for several epochs
            # Set batch size
            train_size = int(self.test_val_split * self.num_examples_train)
            if train:
                # Resize dataset only once, so you have to call first train=True
                train_dataset = dataset.take(train_size)
                # train_dataset = train_dataset.map(lambda *x: (self.preprocess_tensor_4d(x[0]), x[1]))
                # Shuffle input examples
                train_dataset = train_dataset.shuffle(
                    buffer_size=self.default_shuffle_buffer
                )
                train_dataset = train_dataset.repeat(200)
                train_dataset = train_dataset.batch(
                    batch_size=self.batch_size
                )  # , drop_remainder=True)
                train_dataset = train_dataset.prefetch(buffer_size=self.batch_size)
                iterator = train_dataset.make_one_shot_iterator()
                example, labels = iterator.get_next()
                return example, labels
            else:
                val_dataset = dataset.skip(train_size)
                # val_dataset = val_dataset.map(lambda *x: (self.preprocess_tensor_4d(x[0]), x[1]))
                # val_dataset = val_dataset.shuffle(buffer_size=self.default_shuffle_buffer)
                val_dataset = val_dataset.repeat(2)
                val_dataset = val_dataset.batch(
                    batch_size=self.batch_size, drop_remainder=True
                )
                val_dataset = val_dataset.prefetch(buffer_size=self.batch_size)
                iterator = val_dataset.make_one_shot_iterator()
                example, labels = iterator.get_next()
                return example, labels
        else:
            # dataset = dataset.map(lambda *x: (self.preprocess_tensor_4d(x[0]), x[1]))
            dataset = dataset.repeat(1)
            dataset = dataset.batch(batch_size=self.batch_size, drop_remainder=True)
            dataset = dataset.prefetch(buffer_size=self.batch_size)
            iterator = dataset.make_one_shot_iterator()
            example, labels = iterator.get_next()
            return example, labels

        ###########################################################################

    def preprocess_tensor_4d(self, tensor_4d):
        """Preprocess a 4-D tensor (only when some dimensions are `None`, i.e.
        non-fixed). The output tensor wil have fixed, known shape.

        Args:
            tensor_4d: A Tensor of shape
                    [sequence_size, row_count, col_count, num_channels]
                    where some dimensions might be `None`.
        Returns:
            A 4-D Tensor with fixed, known shape.
        """
        tensor_4d_shape = tensor_4d.shape
        logger.info("Tensor shape before preprocessing: {}".format(tensor_4d_shape))

        if tensor_4d_shape[0] > 0 and tensor_4d_shape[0] < 10:
            num_frames = tensor_4d_shape[0]
        else:
            num_frames = self.default_num_frames
        if tensor_4d_shape[1] > 0:
            new_row_count = tensor_4d_shape[1]
        else:
            new_row_count = self.default_image_size[0]
        if tensor_4d_shape[2] > 0:
            new_col_count = tensor_4d_shape[2]
        else:
            new_col_count = self.default_image_size[1]

        if not tensor_4d_shape[0] > 0 or True:
            logger.info(
                "Detected that examples have variable sequence_size, will "
                + "randomly crop a sequence with num_frames = "
                + "{}".format(num_frames)
            )
            tensor_4d = crop_time_axis(tensor_4d, num_frames=num_frames)
        if not tensor_4d_shape[1] > 0 or not tensor_4d_shape[2] > 0:
            logger.info(
                "Detected that examples have variable space size, will "
                + "resize space axes to (new_row_count, new_col_count) = "
                + "{}".format((new_row_count, new_col_count))
            )
            tensor_4d = resize_space_axes(
                tensor_4d, new_row_count=new_row_count, new_col_count=new_col_count
            )
        logger.info("Tensor shape after preprocessing: {}".format(tensor_4d.shape))
        return tensor_4d


def get_logger(verbosity_level):
    """Set logging format to something like:
       2019-04-25 12:52:51,924 INFO model.py: <message>
  """
    logger = logging.getLogger(__file__)
    logging_level = getattr(logging, verbosity_level)
    logger.setLevel(logging_level)
    formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)s %(filename)s: %(message)s"
    )
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging_level)
    stdout_handler.setFormatter(formatter)
    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setLevel(logging.WARNING)
    stderr_handler.setFormatter(formatter)
    logger.addHandler(stdout_handler)
    logger.addHandler(stderr_handler)
    logger.propagate = False
    return logger


logger = get_logger("INFO")
