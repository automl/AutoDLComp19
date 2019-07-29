# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Modified by: Zhengying Liu, Isabelle Guyon

"""An example of code submission for the AutoDL challenge.

It implements 3 compulsory methods ('__init__', 'train' and 'test') and
an attribute 'done_training' for indicating if the model will not proceed more
training due to convergence or limited time budget.

To create a valid submission, zip model.py together with other necessary files
such as Python modules/packages, pre-trained weights, etc. The final zip file
should not exceed 300MB.
"""

import logging
import numpy as np
import os
import torch
import sys
import tensorflow as tf
import time
import subprocess
import torchvision
from opts import parser
from ops.load_dataloader import get_model_for_loader
from ops.load_models import load_loss_criterion, load_model_and_optimizer
from dataset_kakaobrain import TFDataset
from dataloader_kakaobrain import FixedSizeDataLoader
from transforms import (
    GroupCenterCrop, GroupNormalize, GroupResize, GroupScale, IdentityTransform, SelectSamples, Stack,
    ToPilFormat, ToTorchFormatTensor
)


class ParserMock():
    # mock class for handing over the correct arguments
    def __init__(self):
        self._parser_args = parser.parse_known_args()[0]
        setattr(
            self._parser_args, 'finetune_model',
            #'./AutoDL_sample_code_submission/1pretrained_models/Averagenet_RGB_Kinetics_128.pth.tar'
            './input/res/pretrained_models/Averagenet_RGB_Kinetics_128.pth.tar'
        )
        setattr(self._parser_args, 'arch', 'Averagenet')
        setattr(self._parser_args, 'batch_size', 64)
        setattr(self._parser_args, 'num_segments', 8)
        setattr(self._parser_args, 'optimizer', 'Adam')
        setattr(self._parser_args, 'modality', 'RGB')
        setattr(self._parser_args, 'print', True)
        setattr(self._parser_args, 't_diff', 1.0 / 50)

        # setattr(
        #   self._parser_args, 'finetune_model',
        #   './AutoDL_sample_code_submission/pretrained_models/somethingv2_rgb_epoch_16_checkpoint.pth.tar'
        # )
        # setattr(self._parser_args, 'arch', 'TSM')
        # setattr(self._parser_args, 'batch_size', 32)
        # setattr(self._parser_args, 'modality', 'RGB')
        # setattr(self._parser_args, 'classification_type', 'multiclass')
        # setattr(self._parser_args, 'shift', True)
        # setattr(self._parser_args, 'shift_div', 9)
        # setattr(self._parser_args, 'shift_place', 'blockres')
        # setattr(self._parser_args, 'epoch_frac', 0.1)
        # setattr(self._parser_args, 'dense_sample', True)
        # setattr(self._parser_args, 'input_size', 224)
        # setattr(self._parser_args, 'print', True)
        # setattr(self._parser_args, 'time_mult', 10000)

        if torch.cuda.device_count() == 1:
            try:
                from apex import amp
                setattr(self._parser_args, 'apex_available', True)
            except Exception:
                pass
            print('Apex =', self._parser_args.apex_available)

    def set_attr(self, attr, val):
        setattr(self._parser_args, attr, val)

    def parse_args(self):
        return self._parser_args


class Model(object):
    """Trivial example of valid model. Returns all-zero predictions."""

    def __init__(self, metadata):
        """
        Args:
          metadata: an AutoDLMetadata object. Its definition can be found in
              AutoDL_ingestion_program/dataset.py
        """
        print('INIT')
        super().__init__()
        self.time_start = time.time()
        self.done_training = False
        self.metadata = metadata
        self.num_classes = self.metadata.get_output_size()
        self.num_examples_train = self.metadata.size()

        row_count, col_count = self.metadata.get_matrix_size(0)
        channel = self.metadata.get_num_channels(0)
        sequence_size = self.metadata.get_sequence_size()
        print('INPUT SHAPE :', row_count, col_count, channel, sequence_size)

        parser = ParserMock()
        parser.set_attr('num_classes', self.num_classes)

        self.parser_args = parser.parse_args()
        self.model, self.optimizer = load_model_and_optimizer(
            self.parser_args, 0.1, 0.001)
        self.model_for_loader = get_model_for_loader(self.parser_args)
        self.model.cuda()

        self.training_round = 0  # flag indicating if we are in the first round of training
        self.testing_round = 0  # flag indicating if we are in the first round of testing
        self.num_samples_training = None  # number of training samples
        self.num_samples_testing = None  # number of test samples
        self.is_multiclass = None  # multilabel or multiclass dataset?

        self.session = tf.Session()

    def train(self, dataset, remaining_time_budget=None):
        """Train this algorithm on the tensorflow |dataset|.

        This method will be called REPEATEDLY during the whole training/predicting
        process. So your `train` method should be able to handle repeated calls and
        hopefully improve your model performance after each call.

        ****************************************************************************
        ****************************************************************************
        IMPORTANT: the loop of calling `train` and `test` will only run if
            self.done_training = False
          (the corresponding code can be found in ingestion.py, search
          'M.done_training')
          Otherwise, the loop will go on until the time budget is used up. Please
          pay attention to set self.done_training = True when you think the model is
          converged or when there is not enough time for next round of training.
        ****************************************************************************
        ****************************************************************************

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

          remaining_time_budget: a float, time remaining to execute train(). The method
              should keep track of its execution time to avoid exceeding its time
              budget. If remaining_time_budget is None, no time budget is imposed.
        """
        logger.info("TRAINING START: " + str(time.time()))
        logger.info("REMAINING TIME: " + str(remaining_time_budget))

        self.training_round += 1

        t1 = time.time()

        # initial config during first round
        if int(self.training_round) == 1:
            print('TRAINING: FIRST ROUND')
            # show directory structure
            for root, subdirs, files in os.walk(os.getcwd()):
                print(root)
            # get multiclass/multilabel information
            ds_temp = TFDataset(session=self.session, dataset=dataset, num_samples=10)
            info = ds_temp.scan()
            if info['is_multilabel']:
                setattr(self.parser_args, 'classification_type', 'multilabel')
            else:
                setattr(self.parser_args, 'classification_type', 'multiclass')
            self.criterion = load_loss_criterion(self.parser_args)

        t2 = time.time()

        train_augmentation = self.model_for_loader.get_augmentation()
        input_mean = self.model_for_loader.input_mean
        input_std = self.model_for_loader.input_std
        transform = torchvision.transforms.Compose([
            SelectSamples(self.parser_args.num_segments),
            ToPilFormat(),
            train_augmentation,
            Stack(roll=True),
            ToTorchFormatTensor(div=False),
            GroupNormalize(input_mean, input_std)])

        torch.set_grad_enabled(True)
        self.model.train()

        t3 = time.time()

        ds = TFDataset(session=self.session,
                       dataset=dataset,
                       num_samples=10000000,
                       transform=transform)

        dl = FixedSizeDataLoader(ds,
                                 steps=10000000,
                                 batch_size=self.parser_args.batch_size,
                                 shuffle=True,
                                 num_workers=0,
                                 pin_memory=True,
                                 drop_last=False)

        t4 = time.time()

        t_train = time.time()
        brk = False
        while brk == False:
            for i, (data, labels) in enumerate(dl):
                print('training: ' + str(i))
                labels_format = format_labels(labels, self.parser_args)
                labels_format = labels_format.cuda()
                data.cuda()
                labels_var = torch.autograd.Variable(labels_format)
                data_var = torch.autograd.Variable(data)

                output = self.model(data_var)
                loss = self.criterion(output, labels_var)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                if i == 0:
                    subprocess.run(["nvidia-smi"])

                t_cur = time.time()

                # early out
                t_diff = transform_time_abs(t_cur - self.time_start) - \
                         transform_time_abs(t_train - self.time_start)
                if t_diff > self.parser_args.t_diff:
                    brk = True
                    break

        self.training_round += 1
        self.done_training = True

        t5 = time.time()

        logger.info('\nTIMINGS TRAINING: ' + \
                    '\n t2-t1 ' + str(t2 - t1) + \
                    '\n t3-t2 ' + str(t3 - t2) + \
                    '\n t4-t3 ' + str(t4 - t3) + \
                    '\n t5-t4 ' + str(t5 - t4))

        logger.info("TRAINING END: " + str(time.time()))

    def test(self, dataset, remaining_time_budget=None):
        """Make predictions on the test set `dataset` (which is different from that
        of the method `train`).

        Args:
          Same as that of `train` method, except that the labels will be empty
              (all zeros) since this time `dataset` is a test set.
        Returns:
          predictions: A `numpy.ndarray` matrix of shape (num_samples, output_dim).
              here `num_samples` is the number of examples in this dataset as test
              set and `output_dim` is the number of labels to be predicted. The
              values should be binary or in the interval [0,1].
        """
        logger.info("TESTING START: " + str(time.time()))
        logger.info("REMAINING TIME: " + str(remaining_time_budget))

        self.testing_round += 1

        t1 = time.time()

        if int(self.testing_round) == 1:
            print('TESTING: FIRST ROUND')
            ds_temp = TFDataset(session=self.session,
                                dataset=dataset,
                                num_samples=10000000)

            info = ds_temp.scan()
            self.num_samples_testing = info['count']

        t2 = time.time()

        input_mean = self.model_for_loader.input_mean
        input_std = self.model_for_loader.input_std
        transform = torchvision.transforms.Compose([
            SelectSamples(self.parser_args.num_segments),
            ToPilFormat(),
            GroupResize(int(self.model_for_loader.scale_size)),
            GroupCenterCrop(self.model_for_loader.crop_size),
            Stack(roll=True),
            ToTorchFormatTensor(div=False),
            GroupNormalize(input_mean, input_std)])
        self.model.eval()
        self.model.cuda()
        torch.set_grad_enabled(False)
        predictions = None

        t3 = time.time()
        ds = TFDataset(session=self.session,
                       dataset=dataset,
                       num_samples=self.num_samples_testing,
                       transform=transform)

        dl = torch.utils.data.DataLoader(ds,
                                         batch_size=self.parser_args.batch_size,
                                         drop_last=False)

        t4 = time.time()

        for i, (data, _) in enumerate(dl):
            print('testing: ' + str(i))
            data.cuda()
            output = self.model(data)
            if predictions is None:
                predictions = output
            else:
                predictions = torch.cat((predictions, output), 0)

        self.done_training = False

        t5 = time.time()

        logger.info('\nTIMINGS TESTING: ' + \
                    '\n t2-t1 ' + str(t2 - t1) + \
                    '\n t3-t2 ' + str(t3 - t2) + \
                    '\n t4-t3 ' + str(t4 - t3) + \
                    '\n t5-t4 ' + str(t5 - t4))

        logger.info("TESTING END: " + str(time.time()))

        return predictions.cpu().numpy()

    ##############################################################################
    #### Above 3 methods (__init__, train, test) should always be implemented ####
    ##############################################################################


def format_labels(labels, parser_args):
    if parser_args.classification_type == 'multiclass':
        return np.argmax(labels, axis=1)
    else:
        return labels


def transform_time_abs(t_abs):
    '''
    conversion from absolute time 0s-1200s to relative time 0-1
    '''
    return np.log(1 + t_abs / 60.0) / np.log(21)


def transform_time_rel(t_rel):
    '''
    convertsion from relative time 0-1 to absolute time 0s-1200s
    '''
    return 60 * (21 ** t_rel - 1)


def get_logger(verbosity_level):
    """Set logging format to something like:
         2019-04-25 12:52:51,924 INFO model.py: <message>
    """
    logger = logging.getLogger(__file__)
    logging_level = getattr(logging, verbosity_level)
    logger.setLevel(logging_level)
    formatter = logging.Formatter(
        fmt='%(asctime)s %(levelname)s %(filename)s: %(message)s')
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


logger = get_logger('INFO')
