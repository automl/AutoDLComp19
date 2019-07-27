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
import torchvision
from opts import parser
from ops.load_dataloader import get_model_for_loader
from ops.load_models import load_loss_criterion, load_model_and_optimizer
from dataset_kakaobrain import TFDataset
from dataloader_kakaobrain import FixedSizeDataLoader
from transforms import (
    GroupCenterCrop, GroupNormalize, GroupScale, IdentityTransform, SelectSamples, Stack,
    ToPilFormat, ToTorchFormatTensor
)


class ParserMock():
  # mock class for handing over the correct arguments
  def __init__(self):
    self._parser_args = parser.parse_known_args()[0]
    setattr(
      self._parser_args, 'finetune_model',
      './AutoDL_sample_code_submission/pretrained_models/Averagenet_RGB_Kinetics_128.pth.tar'
    )
    setattr(self._parser_args, 'arch', 'Averagenet')
    setattr(self._parser_args, 'batch_size', 32)
    setattr(self._parser_args, 'optimizer', 'SGD')
    setattr(self._parser_args, 'modality', 'RGB')
    setattr(self._parser_args, 'time_mult', 1000)
    setattr(self._parser_args, 'print', True)
    setattr(self._parser_args, 'classification_type', 'multiclass')

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
      self.parser_args, 0.5, 0.001)
    self.criterion = load_loss_criterion(self.parser_args)

    if torch.cuda.is_available():
      self.model.cuda()

    self.first_round_training = True  # flag indicating if we are in the first round of training
    self.first_round_testing  = True  # flag indicating if we are in the first round of testing
    self.num_samples_training = None  # number of training samples
    self.num_samples_testing  = None  # number of test samples
    self.is_multilabel        = None  # multilabel or multiclass dataset?
    self.majority_label       = None  # most representative output label calculated during first round

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
    logger.info("TRAINING START")

    # early out during first round
    if self.first_round_training:
      print('TRAINING: FIRST ROUND')
      self.num_samples_training, self.is_multilabel, self.majority_label = get_dataset_information(dataset, self.num_classes)
      self.first_round_training = False
      self.done_training = True
      return

    model = get_model_for_loader(self.parser_args)
    train_augmentation = model.get_augmentation()
    input_mean = model.input_mean
    input_std = model.input_std
    transform = torchvision.transforms.Compose([
                    SelectSamples(self.parser_args.num_segments),
                    ToPilFormat(),
                    train_augmentation,
                    Stack(roll=True),
                    ToTorchFormatTensor(div=False),
                    GroupNormalize(input_mean, input_std)])

    self.model.train()
    torch.set_grad_enabled(True)

    batch_size = self.parser_args.batch_size

    epoch_frac = get_epoch_frac(self.time_start, time.time(), self.parser_args.time_mult)
    print('EPOCH_FRAC: ' + str(epoch_frac))

    with tf.Session() as sess:
      ds = TFDataset(session=sess,
                     dataset=dataset,
                     num_samples=self.num_samples_training,
                     transform=transform)

      dl = FixedSizeDataLoader(ds,
                               steps=self.num_samples_training,
                               batch_size=batch_size,
                               shuffle=True,
                               num_workers=0,
                               pin_memory=True,
                               drop_last=False)

      for i, (data, labels) in enumerate(dl):
        print('training: ' + str(i))
        labels_int = np.argmax(labels, axis=1)

        labels_int = labels_int.cuda()
        data = data.cuda()
        labels_var = torch.autograd.Variable(labels_int)
        data_var = torch.autograd.Variable(data)

        output = self.model(data_var)
        loss = self.criterion(output, labels_var)
        loss.backward()

        # early out

        if i*batch_size > epoch_frac:
          print('early out')
          return

    self.done_training = True

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
    logger.info("TESTING START")

    if self.first_round_testing is True:
      print('TESTING: FIRST ROUND')
      self.num_samples_testing, _, _ = get_dataset_information(dataset, self.num_classes)
      self.first_round_testing = False
      self.done_training = False
      return np.tile(self.majority_label, [self.num_samples_testing, 1])

    model = get_model_for_loader(self.parser_args)
    train_augmentation = model.get_augmentation()
    input_mean = model.input_mean
    input_std = model.input_std
    transform = torchvision.transforms.Compose([
      SelectSamples(self.parser_args.num_segments),
      ToPilFormat(),
      train_augmentation,
      Stack(roll=True),
      ToTorchFormatTensor(div=False),
      GroupNormalize(input_mean, input_std)])

    self.model.eval()
    torch.set_grad_enabled(False)
    predictions = None

    with tf.Session() as sess:
      ds = TFDataset(session=sess,
                     dataset=dataset,
                     num_samples=self.num_samples_testing,
                     transform=transform)

      dl = torch.utils.data.DataLoader(ds,
                                       batch_size=self.parser_args.batch_size,
                                       drop_last=False)

      for i, (data, _) in enumerate(dl):
        print('testing: ' + str(i))

        output = self.model(data)

        if predictions is None:
          predictions = output
        else:
          predictions = torch.cat((predictions, output), 0)

    self.done_training = False
    return predictions.cpu().numpy()

  ##############################################################################
  #### Above 3 methods (__init__, train, test) should always be implemented ####
  ##############################################################################


def get_dataset_information(dataset, num_classes):
  '''
  extract information about the dataset:
  - num_samples: number of samples withing the dataset
  - is_multilabel: flag indicating if the dataset is multilabel/multiclass
  - label_rep: representative label, used for first round
  '''
  num_samples = 0
  is_multilabel = False
  label_sum = np.zeros([num_classes])

  iterator = dataset.make_one_shot_iterator()
  example, labels = iterator.get_next()

  # how the fuck can you access elements within a TF session?
  with tf.Session() as sess:
    while True:
      try:
        sess.run(labels)
        num_samples += 1
      except tf.errors.OutOfRangeError:
        break

  # because of that we have to iterate a second time...
  with tf.Session() as sess:
    ds = TFDataset(session=sess,
                   dataset=dataset,
                   num_samples=num_samples)

    dl = torch.utils.data.DataLoader(ds,
                                     batch_size=1,
                                     drop_last=False)

    for i, (_, label) in enumerate(dl):
      is_multilabel = is_multilabel or check_multilabel(label)
      label_sum += label.cpu().numpy().flatten()

  if is_multilabel is True:  # multilabel: get representative output -> could be dangerous
    label_maj = label_sum / sum(label_sum)
  else:  # multiclass: get most occuring class label
    label_maj_tmp = (label_sum == max(label_sum))
    # the two following lines are needed if the most occurring label is not unique
    label_maj = np.zeros([num_classes])
    label_maj[np.argmax(label_maj_tmp)] = 1

  print('NUM SAMPLES: ' + str(num_samples))

  return num_samples, is_multilabel, label_maj


def check_multilabel(labels):
  if labels.dim() == 1:   # single label
    return check_multilabel_single(labels)

  elif labels.dim() == 2: # batch of labels
    for label in labels:
      if check_multilabel_single(label) == True:
        return True
    return False

  else:                   # something else
    raise ValueError('label dimension not treated: ' + str(labels.dim()))


def check_multilabel_single(label):
  '''
  check if the given label is multiclass (a single label with 1) or multilabel (all other cases)
  '''
  if sum(label > 1e-9) > 1:
    return True
  return False



def get_epoch_frac(t_start, t_cur, t_mult):
  return transform_time(t_cur-t_start)*t_mult


def transform_time(dt):
  return np.log(1+dt/60.0) / np.log(1+20)


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
