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
import tensorflow as tf
import time
import subprocess
import torchvision
import _pickle as pickle
from opts import parser
from ops.load_models import load_loss_criterion, load_model_and_optimizer
from transforms import SelectSamples, RandomCropPad
from dataset_kakaobrain import TFDataset
from wrapper_net import WrapperNet
from torch.optim.lr_scheduler import StepLR
from utils import LOGGER



class ParserMock():
    # mock class for handing over the correct arguments
    def __init__(self):
        self._parser_args = parser.parse_known_args()[0]
        self.load_manual_parameters()
        self.load_bohb_parameters()

    def load_manual_parameters(self):
        # manually set parameters
        rootpath = os.path.dirname(__file__)
        LOGGER.setLevel(logging.DEBUG)
        setattr(self._parser_args, 'finetune_model', os.path.join(rootpath, 'pretrained_models/'))
        setattr(self._parser_args, 'arch', 'ECOfull_efficient_py') # Averagenet or bninception
        setattr(self._parser_args, 'bn_prod_limit', 64)    # limit of batch_size * num_segments
        setattr(self._parser_args, 'batch_size_train', 16)
        setattr(self._parser_args, 'num_segments_test', 4)
        setattr(self._parser_args, 'num_segments_step', 4000)
        setattr(self._parser_args, 'optimizer', 'SGD')
        setattr(self._parser_args, 'modality', 'RGB')
        setattr(self._parser_args, 'dropout_diff', 1e-3)
        setattr(self._parser_args, 't_diff', 1.0 / 50)
        setattr(self._parser_args, 'lr', 0.005)
        setattr(self._parser_args, 'lr_gamma', 0.01)
        setattr(self._parser_args, 'lr_step', 10)
        setattr(self._parser_args, 'print', True)
        setattr(self._parser_args, 'fast_augment', True)
        setattr(self._parser_args, 'early_stop', 600)

    def load_bohb_parameters(self):
        # parameters from bohb_auc
        path = os.path.join(os.getcwd(), 'bohb_config.txt')
        if os.path.isfile(path):
            with open(path, 'rb') as file:
                LOGGER.info('FOUND BOHB CONFIG, OVERRIDING PARAMETERS')
                bohb_cfg = pickle.load(file)
                LOGGER.info('BOHB_CFG: ' + str(bohb_cfg))
                for key, value in bohb_cfg.items():
                    LOGGER.info('OVERRIDING PARAMETER ' + str(key) + ' WITH ' + str(value))
                    setattr(self._parser_args, key, value)
            os.remove(path)

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
        LOGGER.info("INIT START: " + str(time.time()))
        super().__init__()

        # This flag allows you to enable the inbuilt cudnn auto-tuner to find the best
        # algorithm to use for your hardware. Benchmark mode is good whenever your input sizes
        # for your network do not vary
        # https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do/5936
        #torch.backends.cudnn.benchmark = True

        self.time_start = time.time()
        self.train_time = []
        self.test_time = []

        self.done_training = False

        self.metadata = metadata
        self.num_classes = self.metadata.get_output_size()
        self.num_examples_train = self.metadata.size()

        row_count, col_count = self.metadata.get_matrix_size(0)
        channel = self.metadata.get_num_channels(0)
        sequence_size = self.metadata.get_sequence_size()
        print('INPUT SHAPE : ', row_count, col_count, channel, sequence_size)
        parser = ParserMock()
        parser.set_attr('num_classes', self.num_classes)

        self.parser_args = parser.parse_args()
        self.select_fast_augment()

        self.training_round = 0  # flag indicating if we are in the first round of training
        self.testing_round = 0
        self.num_samples_testing = None
        self.train_counter = 0

        self.session = tf.Session()
        LOGGER.info("INIT END: " + str(time.time()))


    def train(self, dataset, remaining_time_budget=None):
        LOGGER.info("TRAINING START: " + str(time.time()))
        LOGGER.info("REMAINING TIME: " + str(remaining_time_budget))

        self.training_round += 1

        t1 = time.time()

        # initial config during first round
        if int(self.training_round) == 1:
            self.late_init(dataset)

        t2 = time.time()

        num_segments = self.set_num_segments(is_training=True)
        batch_size = self.set_batch_size(num_segments, is_training=True)
        dl_train = self.get_dataloader_train(dataset, num_segments, batch_size)
        torch.set_grad_enabled(True)
        self.model.train()

        t3 = time.time()

        t_train = time.time()
        finish_loop = False
        while not finish_loop:
            # Set train mode before we go into the train loop over an epoch
            for i, (data, labels) in enumerate(dl_train):
                if (
                    hasattr(self.parser_args, 'early_stop')
                    and time.time() - self.time_start > self.parser_args.early_stop
                ):
                    finish_loop = True
                    break

                self.optimizer.zero_grad()

                output = self.model(data.cuda())
                labels = format_labels(labels, self.parser_args).cuda()

                loss = self.criterion(output, labels)
                loss.backward()
                self.optimizer.step()
                self.lr_scheduler.step()
                self.set_dropout()
                self.train_counter += batch_size

                LOGGER.info('TRAIN BATCH #{0}:\t{1}'.format(i, loss))

                if int(self.training_round) == 1:
                    if i > 20:
                        finish_loop = True
                        break
                else:
                    t_diff = (transform_to_time_rel(time.time() - self.time_start)
                        - transform_to_time_rel(t_train - self.time_start))

                    if t_diff > self.parser_args.t_diff:
                        finish_loop = True
                        break

            subprocess.run(['nvidia-smi'])
            self.training_round += 1

        t4 = time.time()
        LOGGER.info(
            '\nTIMINGS TRAINING: ' +
            '\n t2-t1 ' + str(t2 - t1) +
            '\n t3-t2 ' + str(t3 - t2) +
            '\n t4-t3 ' + str(t4 - t3))

        LOGGER.info('LR: ')
        for param_group in self.optimizer.param_groups:
            LOGGER.info(param_group['lr'])
        LOGGER.info('DROPOUT: ' + str(self.model.model.dropout))
        LOGGER.info("TRAINING FRAMES PER SEC: " + str(self.train_counter/(time.time()-self.time_start)))
        LOGGER.info("TRAINING COUNTER: " + str(self.train_counter))
        LOGGER.info("TRAINING END: " + str(time.time()))
        self.train_time.append(t4 - t1)


    def late_init(self, dataset):
        LOGGER.info('TRAINING: FIRST ROUND')
        # show directory structure
        # for root, subdirs, files in os.walk(os.getcwd()):
        #     LOGGER.info(root)

        # get multiclass/multilabel information based on a small subset of videos/images

        t1 = time.time()

        ds_temp = TFDataset(session=self.session, dataset=dataset, num_samples=self.num_examples_train)

        t2 = time.time()

        self.info = ds_temp.scan2(50)

        t3 = time.time()

        LOGGER.info('AVG SHAPE: ' + str(self.info['avg_shape']))

        if self.info['is_multilabel']:
            setattr(self.parser_args, 'classification_type', 'multilabel')
        else:
            setattr(self.parser_args, 'classification_type', 'multiclass')

        t4 = time.time()

        self.select_model()

        t5 = time.time()

        self.model_main, self.optimizer = load_model_and_optimizer(self.parser_args)
        self.model_main.partialBN(False)  # bninception default behaviour is to freeze
                                          # which gets apply after the first .train() call

        t6 = time.time()

        self.model = WrapperNet(self.model_main, self.parser_args.fast_augment)
        self.model.cuda()
        self.lr_scheduler = StepLR(self.optimizer,
                                   self.parser_args.lr_step,
                                   1-self.parser_args.lr_gamma)
        self.set_dropout(first_round=True)

        t7 = time.time()

        # load proper criterion for multiclass/multilabel
        self.criterion = load_loss_criterion(self.parser_args)

        t8 = time.time()

        LOGGER.info(
            '\nTIMINGS FIRST ROUND: ' +
            '\n t2-t1 ' + str(t2 - t1) +
            '\n t3-t2 ' + str(t3 - t2) +
            '\n t4-t3 ' + str(t4 - t3) +
            '\n t5-t4 ' + str(t5 - t4) +
            '\n t6-t5 ' + str(t6 - t5) +
            '\n t7-t6 ' + str(t7 - t6) +
            '\n t8-t7 ' + str(t8 - t7))

    def test(self, dataset, remaining_time_budget=None):
        if (
            hasattr(self.parser_args, 'early_stop')
            and time.time() - self.time_start > self.parser_args.early_stop
        ):
            self.done_training = True
            return None
        LOGGER.info("TESTING START: " + str(time.time()))
        LOGGER.info("REMAINING TIME: " + str(remaining_time_budget))

        t1 = time.time()

        self.testing_round += 1

        if int(self.testing_round) == 1:
            scan_start = time.time()
            ds_temp = TFDataset(session=self.session, dataset=dataset, num_samples=10000000)
            info = ds_temp.scan2()
            self.num_samples_testing = info['num_samples']
            LOGGER.info('SCAN TIME: ' + str(time.time() - scan_start))
            LOGGER.info('TESTING: FIRST ROUND')

        t2 = time.time()

        num_segments = self.set_num_segments(is_training=False)
        batch_size = self.set_batch_size(num_segments, is_training=False)
        dl = self.get_dataloader_test(dataset, num_segments, batch_size)
        torch.set_grad_enabled(False)
        self.model.eval()

        t3 = time.time()

        prediction_list = []
        for i, (data, _) in enumerate(dl):
            LOGGER.info('TEST: ' + str(i))
            prediction_list.append(self.model(data.cuda()).cpu())
        predictions = np.vstack(prediction_list)

        t4 = time.time()

        LOGGER.info(
            '\nTIMINGS TESTING: ' +
            '\n t2-t1 ' + str(t2 - t1) +
            '\n t3-t2 ' + str(t3 - t2) +
            '\n t4-t3 ' + str(t4 - t3)
        )

        LOGGER.info("TESTING END: " + str(time.time()))
        self.test_time.append(t3 - t1)
        return predictions


    def select_fast_augment(self):
        '''
        if all input videos/images have the same width/height, we can do faster data augmentation on the GPU
        '''
        if (
            hasattr(self.parser_args, 'fast_augment')
            and not self.parser_args.fast_augment
        ):
            return
        row_count, col_count = self.metadata.get_matrix_size(0)
        if row_count > 0 and col_count > 0:
            setattr(self.parser_args, 'fast_augment', True)
        else:
            setattr(self.parser_args, 'fast_augment', False)
        LOGGER.info('FAST AUGMENT: ' + str(self.parser_args.fast_augment))


    def select_model(self):
        '''
        select proper model based on information from the dataset (image/video, etc.)
        '''
        avg_shape = self.info['avg_shape']
        if self.metadata.get_sequence_size() == 1:  # image network
            num_pixel = avg_shape[1]*avg_shape[2]
            if num_pixel < 10000:                   # select network based on average number of pixels in the dataset
                self.parser_args.finetune_model = self.parser_args.finetune_model + 'Efficientnet_Image_Input_64_non_final.pth.tar'
            else:
                self.parser_args.finetune_model = self.parser_args.finetune_model + 'Efficientnet_Image_Input_128_non_final.pth.tar'
        else:                                       # video network
            self.parser_args.finetune_model = self.parser_args.finetune_model + 'Efficientnet_Image_Input_128_non_final.pth.tar'
        LOGGER.info('USE MODEL: ' + str(self.parser_args.finetune_model))


    def set_dropout(self, first_round=False):
        '''
        linearly increase dropout over number of processed batches. Also ensures that dropout is never larger than 0.9
        '''
        if first_round:
            self.model.model.dropout = 0
        else:
            self.model.model.dropout = self.model.model.dropout + self.parser_args.dropout_diff
            self.model.model.dropout = min(self.model.model.dropout, 0.9)
        self.model.model.alphadrop = torch.nn.AlphaDropout(p=self.model.model.dropout)


    def set_num_segments(self, is_training):
        '''
        increase number of segments after a given time by a factor of 2
        '''

        if self.metadata.get_sequence_size() == 1:
            # image dataset
            num_segments = 1
        else:
            # video dataset
            if is_training:
                num_segments = 2**int(self.train_counter/self.parser_args.num_segments_step+1)
                avg_frames = self.info['avg_shape'][0]
                if avg_frames > 64:
                    upper_limit = 16
                else:
                    upper_limit = 8
                num_segments = min(max(num_segments, 2), upper_limit)
            else:
                num_segments = self.parser_args.num_segments_test

        LOGGER.info('TRAIN COUNTER: ' + str(self.train_counter))
        LOGGER.info('SET NUM SEGMENTS: ' + str(num_segments))
        self.model.model.num_segments = num_segments
        return num_segments


    def set_batch_size(self, num_segments, is_training):
        '''
        calculate resulting batch size based on desired batch size and specified upper limit due to GPU memory
        '''
        if is_training:
            bn_prod_des = self.parser_args.batch_size_train*num_segments
            if bn_prod_des <= self.parser_args.bn_prod_limit:
                batch_size = self.parser_args.batch_size_train
            else:
                batch_size = int(self.parser_args.bn_prod_limit / num_segments)
        else:
            batch_size = int(self.parser_args.bn_prod_limit / num_segments)*4

        LOGGER.info('SET BATCH SIZE: ' + str(batch_size))

        return batch_size


    def get_transform(self, num_segments):
        if self.parser_args.fast_augment:
            return torchvision.transforms.Compose([
                SelectSamples(num_segments)])
        else:
            return torchvision.transforms.Compose([
                SelectSamples(num_segments),
                RandomCropPad(self.model_main.input_size)])


    def get_dataloader_train(self, dataset, num_segments, batch_size):
        transform = self.get_transform(num_segments)

        ds = TFDataset(
            session=self.session,
            dataset=dataset,
            num_samples=int(10000000),
            transform=transform
        )

        dl = torch.utils.data.DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False
        )

        return dl


    def get_dataloader_test(self, dataset, num_segments, batch_size):
        transform = self.get_transform(num_segments)

        ds = TFDataset(
            session=self.session,
            dataset=dataset,
            num_samples=self.num_samples_testing,
            transform=transform
        )

        dl = torch.utils.data.DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=False
        )

        return dl


    ##############################################################################
    #### Above 3 methods (__init__, train, test) should always be implemented ####
    ##############################################################################


def format_labels(labels, parser_args):
    if parser_args.classification_type == 'multiclass':
        return np.argmax(labels, axis=1)
    else:
        return labels


def transform_to_time_rel(t_abs):
    '''
    conversion from absolute time 0s-1200s to relative time 0-1
    '''
    return np.log(1 + t_abs / 60.0) / np.log(21)


def transform_to_time_abs(t_rel):
    '''
    convertsion from relative time 0-1 to absolute time 0s-1200s
    '''
    return 60 * (21 ** t_rel - 1)
