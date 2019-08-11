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
import copy
import os
import torch
import sys
import tensorflow as tf
import time
import subprocess
import torchvision
import _pickle as pickle
from functools import partial
from opts import parser
from ops.load_models import load_loss_criterion, load_model_and_optimizer
from transforms import SelectSamples, RandomCropPad
from dataset_kakaobrain import TFDataset
from wrapper_net import WrapperNet
from torch.optim.lr_scheduler import StepLR



class ParserMock():
    # mock class for handing over the correct arguments
    def __init__(self):
        self._parser_args = parser.parse_known_args()[0]
        self.load_manual_parameters()
        self.load_bohb_parameters()
        self.load_apex()

    def load_manual_parameters(self):
        # manually set parameters
        rootpath = os.path.dirname(__file__)
        logger.info('ROOT PATH: ' + str(rootpath))
        setattr(self._parser_args, 'finetune_model', os.path.join(rootpath, 'pretrained_models/'))
        setattr(self._parser_args, 'arch', 'bninception') # Averagenet or bninception
        setattr(self._parser_args, 'bn_prod_limit', 256)    # limit of batch_size * num_segments
        setattr(self._parser_args, 'batch_size_train', 128)
        setattr(self._parser_args, 'num_segments_test', 4)
        setattr(self._parser_args, 'num_segments_step', 50000)
        setattr(self._parser_args, 'optimizer', 'SGD')
        setattr(self._parser_args, 'modality', 'RGB')
        setattr(self._parser_args, 'dropout_diff', 1e-4)
        setattr(self._parser_args, 't_diff', 1.0 / 50)
        setattr(self._parser_args, 'lr', 0.002)
        setattr(self._parser_args, 'lr_gamma', 0.01)
        setattr(self._parser_args, 'lr_step', 20)
        setattr(self._parser_args, 'print', True)
        setattr(self._parser_args, 'fast_augment', True)

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
          metadata: an AutoDLMetadata object. Its definition can be found in
              AutoDL_ingestion_program/dataset.py
        """
        logger.info("INIT START: " + str(time.time()))
        super().__init__()

        # This flag allows you to enable the inbuilt cudnn auto-tuner to find the best
        # algorithm to use for your hardware. Benchmark mode is good whenever your input sizes
        # for your network do not vary
        # https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do/5936
        torch.backends.cudnn.benchmark = True

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
        self.testing_round = 0  # flag indicating if we are in the first round of testing
        self.train_counter = 0
        self.num_samples_testing = None  # number of test samples

        self.session = tf.Session()
        logger.info("INIT END: " + str(time.time()))


    def train(self, dataset, remaining_time_budget=None):
        logger.info("TRAINING START: " + str(time.time()))
        logger.info("REMAINING TIME: " + str(remaining_time_budget))

        self.training_round += 1

        t1 = time.time()

        # initial config during first round
        if int(self.training_round) == 1:
            self.late_init(dataset)

        t2 = time.time()

        num_segments = self.set_num_segments(is_training=True)
        batch_size = self.set_batch_size(num_segments, is_training=True)
        dl_train = self.get_dataloader_train(dataset, num_segments, batch_size)
        self.model.train()

        t3 = time.time()

        t_train = time.time()
        self.finish_loop = False
        while not self.finish_loop:
            # Set train mode before we go into the train loop over an epoch
            for i, (data, labels) in enumerate(dl_train):
                self.optimizer.zero_grad()

                output = self.model(data.cuda())
                labels = format_labels(labels, self.parser_args).cuda()

                loss = self.criterion(output, labels)
                loss.backward()
                self.optimizer.step()
                self.lr_scheduler.step()
                self.set_dropout()
                self.train_counter += num_segments*batch_size

                logger.info('TRAIN BATCH #{0}:\t{1}'.format(i, loss))

                t_diff = (transform_to_time_rel(time.time() - self.time_start)
                        - transform_to_time_rel(t_train - self.time_start))

                if t_diff > self.parser_args.t_diff:
                    self.finish_loop = True
                    break

            subprocess.run(['nvidia-smi'])
            self.training_round += 1

        t4 = time.time()

        logger.info(
            '\nTIMINGS TRAINING: ' +
            '\n t2-t1 ' + str(t2 - t1) +
            '\n t3-t2 ' + str(t3 - t2) +
            '\n t4-t3 ' + str(t4 - t3))

        logger.info('LR: ')
        for param_group in self.optimizer.param_groups:
            logger.info(param_group['lr'])
        logger.info('DROPOUT: ' + str(self.model.model.dropout))
        logger.info("TRAINING FRAMES PER SEC: " + str(self.train_counter/(time.time()-self.time_start)))
        logger.info("TRAINING COUNTER: " + str(self.train_counter))
        logger.info("TRAINING END: " + str(time.time()))
        self.train_time.append(t4 - t1)


    def late_init(self, dataset):
        logger.info('TRAINING: FIRST ROUND')
        # show directory structure
        # for root, subdirs, files in os.walk(os.getcwd()):
        #     logger.info(root)

        # get multiclass/multilabel information based on a small subset of videos/images

        ds_temp = TFDataset(session=self.session, dataset=dataset, num_samples=self.num_examples_train)
        scan_start = time.time()
        self.info = ds_temp.scan2(50)
        logger.info('TRAIN SCAN TIME: {0}'.format(time.time() - scan_start))
        logger.info('AVG SHAPE: ' + str(self.info['avg_shape']))

        if self.info['is_multilabel']:
            setattr(self.parser_args, 'classification_type', 'multilabel')
        else:
            setattr(self.parser_args, 'classification_type', 'multiclass')

        self.select_model()
        self.model_main, self.optimizer = load_model_and_optimizer(self.parser_args)
        self.model = WrapperNet(self.model_main, self.parser_args)
        self.model.cuda()
        self.lr_scheduler = StepLR(self.optimizer,
                                   self.parser_args.lr_step,
                                   1-self.parser_args.lr_gamma)
        self.set_dropout(first_round=True)

        # load proper criterion for multiclass/multilabel
        self.criterion = load_loss_criterion(self.parser_args)
        if self.parser_args.apex_available:
            from apex import amp

            def scaled_loss_helper(loss, optimizer):
                with amp.scale_loss(loss, optimizer) as scale_loss:
                    scale_loss.backward()

            def amp_loss(predictions, labels, loss_fn, optimizer):
                loss = loss_fn(predictions, labels)
                if hasattr(optimizer, '_amp_stash'):
                    loss.backward = partial(scaled_loss_helper, loss=loss, optimizer=optimizer)
                return loss

            self.criterion = partial(
                amp_loss, loss_fn=self.criterion, optimizer=self.optimizer
            )


    def test(self, dataset, remaining_time_budget=None):
        logger.info("TESTING START: " + str(time.time()))
        logger.info("REMAINING TIME: " + str(remaining_time_budget))

        self.testing_round += 1

        t1 = time.time()

        if int(self.testing_round) == 1:
            scan_start = time.time()
            ds_temp = TFDataset(session=self.session, dataset=dataset, num_samples=10000000)
            info = ds_temp.scan2()
            self.num_samples_testing = info['num_samples']
            logger.info('SCAN TIME: {0}'.format(time.time() - scan_start))
            logger.info('TESTING: FIRST ROUND')

        t2 = time.time()

        num_segments = self.set_num_segments(is_training=False)
        batch_size = self.set_batch_size(num_segments, is_training=False)
        dl = self.get_dataloader_test(dataset, num_segments, batch_size)
        self.model.eval()

        t3 = time.time()

        predictions = torch.zeros([self.num_samples_testing, self.num_classes])
        idx = 0
        with torch.no_grad():
            for i, (data, _) in enumerate(dl):
                logger.info('test: ' + str(i))
                data = data.cuda()
                output = self.model(data).cpu()
                predictions[idx:idx+output.shape[0],:] = output
                idx += output.shape[0]

        # remove if needed: Only train for 10 mins in order to save time on the submissions
        # if remaining_time_budget < 600:
        #     self.done_training = True
        #     return None

        t4 = time.time()

        logger.info(
            '\nTIMINGS TESTING: ' +
            '\n t2-t1 ' + str(t2 - t1) +
            '\n t3-t2 ' + str(t3 - t2) +
            '\n t4-t3 ' + str(t4 - t3)
        )

        logger.info("TESTING END: " + str(time.time()))
        self.test_time.append(t4 - t1)
        return predictions.cpu().numpy()


    def select_fast_augment(self):
        '''
        if all input videos/images have the same width/height, we can do faster data augmentation on the GPU
        '''
        row_count, col_count = self.metadata.get_matrix_size(0)
        if row_count > 0 and col_count > 0:
            setattr(self.parser_args, 'fast_augment', True)
        else:
            setattr(self.parser_args, 'fast_augment', False)
        logger.info('FAST AUGMENT: ' + str(self.parser_args.fast_augment))


    def select_model(self):
        '''
        select proper model based on information from the dataset (image/video, etc.)
        '''
        avg_shape = self.info['avg_shape']
        if self.metadata.get_sequence_size() == 1:  # image network
            num_pixel = avg_shape[1]*avg_shape[2]
            if num_pixel < 10000:                   # select network based on average number of pixels in the dataset
                self.parser_args.finetune_model = self.parser_args.finetune_model + 'BnT_Image_Input_64.pth.tar'
            else:
                self.parser_args.finetune_model = self.parser_args.finetune_model + 'BnT_Image_Input_128.tar'
        else:                                       # video network
            self.parser_args.finetune_model = self.parser_args.finetune_model + 'BnT_Video_input_128.pth.tar'
        logger.info('USE MODEL: ' + str(self.parser_args.finetune_model))


    def set_dropout(self, first_round=False):
        '''
        linearly increase dropout over number of processed batches. Also ensures that dropout is never larger than 0.9
        '''
        if first_round:
            self.model.model.dropout = 0
        else:
            self.model.model.dropout = self.model.model.dropout + self.parser_args.dropout_diff
            self.model.model.dropout = min(self.model.model.dropout, 0.9)


    def set_num_segments(self, is_training):
        '''
        increase number of segments after a given time by a factor of 2
        '''
        print('train counter: ' + str(self.train_counter))
        print('num segments step: ' + str(self.parser_args.num_segments_step))

        if self.metadata.get_sequence_size() == 1:
            # image dataset
            num_segments = 1
        else:
            # video dataset
            if is_training:
                #num_segments = 2**int(self.train_counter/self.parser_args.num_segments_step+1)
                num_segments = 8
                avg_frames = self.info['avg_shape'][0]
                if avg_frames > 64:
                    upper_limit = 16
                else:
                    upper_limit = 8
                num_segments = min(max(num_segments, 2), upper_limit)
            else:
                num_segments = self.parser_args.num_segments_test

        logger.info('SET NUM SEGMENTS: ' + str(num_segments))
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
            batch_size = int(self.parser_args.bn_prod_limit / num_segments)

        logger.info('SET BATCH SIZE: ' + str(batch_size))

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
            pin_memory=True,
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
            shuffle=False,
            pin_memory=True,
            drop_last=False
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
