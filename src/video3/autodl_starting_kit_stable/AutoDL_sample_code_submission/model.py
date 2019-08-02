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

"""An example of code submission for the Auto challenge.

It implements 3 compulsory methos ('__init__', 'train' and 'test') and
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
import _pickle as pickle
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
        self.load_manual_parameters()
        self.load_bohb_parameters()
        self.load_apex()

    def load_manual_parameters(self):
        # manually set parameters
        setattr(
            self._parser_args, 'finetune_model',
            './AutoDL_sample_code_submission/pretrained_models/Averagenet_RGB_Kinetics_128.pth.tar'
            # './input/res/pretrained_models/Averagenet_RGB_Kinetics_128.pth.tar'
        )
        setattr(self._parser_args, 'arch', 'Averagenet')
        setattr(self._parser_args, 'batch_size', 20)
        setattr(self._parser_args, 'num_segments', 8)
        setattr(self._parser_args, 'optimizer', 'SGD')
        setattr(self._parser_args, 'modality', 'RGB')
        setattr(self._parser_args, 'print', True)
        setattr(self._parser_args, 't_diff', 1.0 / 50)

    def load_bohb_parameters(self):
        # parameters from bohb_auc
        path = os.path.join(os.getcwd(), 'bohb_config.txt')
        if os.path.isfile(path):
            with open(path, 'rb') as file:
                logger.info('FOUND BOHB CONFIG, OVERRIDING PARAMETERS')
                bohb_cfg = pickle.load(file)
                logger.info('BOHB_CFG: ' + str(bohb_cfg))
                for key, value in bohb_cfg.items():
                    logger.info('OVERRIDING PARAMETER ' + str(key) + ' WITH ' + str(value))
                    setattr(self._parser_args, key, value)
            os.remove(path)

    def load_apex(self):
        # apex
        if torch.cuda.device_count() == 1:
            try:
                from apex import amp
                setattr(self._parser_args, 'apex_available', True)
            except Exception:
                pass
            logger.info('Apex = ' + str(self._parser_args.apex_available))

    def set_attr(self, attr, val):
        setattr(self._parser_args, attr, val)

    def parse_args(self):
        return self._parser_args


class Model(object):
    """Trivial example of valid model. Returns all-zero predictions."""

    def __init__(self, metadata):
        """
        Args:
          metadata: an AutoMetadata object. Its definition can be found in
              Auto_ingestion_program/dataset.py
        """
        logger.info("INIT START: " + str(time.time()))
        super().__init__()
        self.time_start = time.time()
        self.done_training = False
        self.metadata = metadata
        self.num_classes = self.metadata.get_output_size()
        self.num_examples_train = self.metadata.size()

        self.row_count, self.col_count = self.metadata.get_matrix_size(0)
        self.channel = self.metadata.get_num_channels(0)
        self.sequence_size = self.metadata.get_sequence_size()
        print('INPUT SHAPE : ', self.row_count, self.col_count,
              self.channel, self.sequence_size)

        parser = ParserMock()
        parser.set_attr('num_classes', self.num_classes)

        self.parser_args = parser.parse_args()
        #self.model, self.optimizer = load_model_and_optimizer(
        #    self.parser_args, 0.1, 0.001)
        #self.model_for_loader = get_model_for_loader(self.parser_args)
        #self.model.cuda()

        self.training_round = 0  # flag indicating if we are in the first round of training
        self.testing_round = 0  # flag indicating if we are in the first round of testing
        self.num_samples_training = None  # number of training samples
        self.num_samples_testing = None  # number of test samples
        self.is_multiclass = None  # multilabel or multiclass dataset?

        self.session = tf.Session()

    def train(self, dataset, remaining_time_budget=None):
        """Train this algorithm on the tensorflow |dataset|.

        This method will be called REPEATEY during the whole training/predicting
        process. So your `train` method should be able to hane repeated calls and
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
        print(dataset)
        self.training_round += 1

        t1 = time.time()

        # initial config during first round
        if int(self.training_round) == 1:
            logger.info('TRAINING: FIRST ROUND')
            # show directory structure
            # for root, subdirs, files in os.walk(os.getcwd()):
            #     logger.info(root)
            # get multiclass/multilabel information
            temp_dataset = TFDataset(session=self.session, dataset=dataset, num_samples=300)
            info = temp_dataset.scan(with_tensors=False)
            del temp_dataset
            if info['is_multilabel']:
                setattr(self.parser_args, 'classification_type', 'multilabel')
            else:
                setattr(self.parser_args, 'classification_type', 'multiclass')
            #print('shape', info)
            if info['example']['shape'][0] == 1:
                print('old batch_size', self.parser_args.batch_size)
                self.parser_args.batch_size *= (self.parser_args.num_segments - 1)
                print('new batch_size', self.parser_args.batch_size)
                self.parser_args.num_segments = 1
            # load proper criterion for multiclass/multilabel
            self.criterion = load_loss_criterion(self.parser_args)
            self.model, self.optimizer = load_model_and_optimizer(
                                          self.parser_args, 0.0, 0.01)
            #self.model_for_loader = get_model_for_loader(self.parser_args)
            
            #TODO: choose resizing based on avg input
            if ((self.row_count > 160 and self.row_count > 160)
                or  (self.row_count < 130 or self.row_count < 130)):
                preprocessor = get_tf_resize(180, 180)
                self.train_dataset = dataset.map(
                    lambda *x: (preprocessor(x[0]), x[1]),
                    num_parallel_calls=tf.data.experimental.AUTOTUNE
                    )
            else:
                self.train_dataset = dataset

            train_augmentation = self.model.get_augmentation()
            self.input_mean = self.model.input_mean
            self.input_std = self.model.input_std
            self.crop_size = self.model.crop_size
            self.scale_size = self.model.scale_size
            transform = torchvision.transforms.Compose([
                SelectSamples(self.parser_args.num_segments),
                ToPilFormat(),
                train_augmentation,
                Stack(roll=True),
                ToTorchFormatTensor(div=False),
                GroupNormalize(self.input_mean, self.input_std)])


            #self.model.cuda()
            self.train_dataset = TFDataset(session=self.session,
                           dataset=self.train_dataset,
                           num_samples=10000000,
                           transform=transform)

            self.train_dataloader = FixedSizeDataLoader(self.train_dataset,
                           steps=10000000,
                           batch_size=self.parser_args.batch_size,
                           shuffle=True,
                           num_workers=0,
                           pin_memory=True,
                           drop_last=False)

        t2 = time.time()
        torch.set_grad_enabled(True)
        self.model.train()
        t_train = time.time()
        brk = False
        while brk == False:
            for i, (data, labels) in enumerate(self.train_dataloader):
                logger.info('training: ' + str(i))
                labels_format = format_labels(labels, self.parser_args)
                labels_format = labels_format.cuda()
                data.cuda()
                labels_var = torch.autograd.Variable(labels_format)
                data_var = torch.autograd.Variable(data).cuda()

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
                # TODO: check train acc before testiong, otherwise negative score
                if t_diff > self.parser_args.t_diff:
                    brk = True
                    break

        self.training_round += 1
        self.done_training = True

        t3 = time.time()

        logger.info('\nTIMINGS TRAINING: ' + \
                    '\n t2-t1 ' + str(t2 - t1) + \
                    '\n t3-t2 ' + str(t3 - t2))

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
            logger.info('TESTING: FIRST ROUND')
            transform = torchvision.transforms.Compose([
                SelectSamples(self.parser_args.num_segments),
                ToPilFormat(),
                #GroupResize(int(self.model_for_loader.scale_size)),
                GroupCenterCrop(self.crop_size),
                Stack(roll=True),
                ToTorchFormatTensor(div=False),
                GroupNormalize(self.input_mean, self.input_std)])
            self.samples_test = 0
            iterator = dataset.make_one_shot_iterator()
            next_element = iterator.get_next()
            while True:
              try:
                  _ = self.session.run(next_element)
                  self.samples_test += 1
              except tf.errors.OutOfRangeError:
                  break
            preprocessor = get_tf_resize(190, 190)
            self.dataset = dataset.map(
                lambda *x: (preprocessor(x[0]), x[1]),
                num_parallel_calls=tf.data.experimental.AUTOTUNE
                )
            self.test_dataset = TFDataset(session=self.session,
                           dataset=self.dataset,
                           num_samples=self.samples_test,
                           transform=transform)
            # info = self.test_dataset
            # TODO: Multicrop and unbalanced last batch
        self.test_dataset.reset()
        test_dataloader = torch.utils.data.DataLoader(
                      self.test_dataset,
                      batch_size=self.parser_args.batch_size,
                      drop_last=False)

        self.model.eval()
        torch.set_grad_enabled(False)
        predictions = None
        t2 = time.time()
        test_dataloader
        for i, (data, _) in enumerate(test_dataloader):
            logger.info('testing: ' + str(i))
            output = self.model(data.cuda())
            if predictions is None:
                predictions = output
            else:
                predictions = torch.cat((predictions, output), 0)

        self.done_training = False

        t3 = time.time()

        logger.info('\nTIMINGS TESTING: ' + \
                    '\n t2-t1 ' + str(t2 - t1) + \
                    '\n t3-t2 ' + str(t3 - t2))


        logger.info("TESTING END: " + str(time.time()))

        return predictions.cpu().numpy()

    ##############################################################################
    #### Above 3 methos (__init__, train, test) should always be implemented ####
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

def get_tf_resize(height=180, width=180):
    def preprocessor(tensor):
        #tensor = tensor[0]
        _ , in_height, in_width, _ = tensor.shape

        if in_width < 128 or in_height < 128:
            tensor = tf.image.resize_images(tensor,
                                            (height, width),
                                            method=3,  # Area
                                            preserve_aspect_ratio=True,)
        else:
            tensor = tf.image.resize_images(tensor,
            															  (height, width),
                                            method=1,  # NEAREST_NEIGHBOR
            															  preserve_aspect_ratio=True,)

        # tensor = (tensor - 0.5) / 0.25
        return tensor
    return preprocessor


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
