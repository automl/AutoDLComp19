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
from common.utils import *
from AutoDL_scoring_program.score import get_solution, accuracy, is_multiclass, autodl_auc


class Model(object):
    """Trivial example of valid model. Returns all-zero predictions."""

    def __init__(self, metadata, cfg, config):
        LOGGER.info("INIT START: " + str(time.time()))
        super().__init__()

        self.time_start = time.time()

        self.metadata = metadata
        self.num_classes = self.metadata.get_output_size()

        parser = ParserMock()
        parser.load_manual_parameters()
        parser.load_dict_parameters(cfg)
        parser.load_dict_parameters(config)

        parser.set_attr('num_classes', self.num_classes)

        self.parser_args = parser.parse_args()

        self.train_round = 0  # flag indicating if we are in the first round of training
        self.test_round = 0
        self.num_samples_testing = None
        self.train_counter = 0
        self.train_batches = 0
        self.dl_train = None
        self.batch_size_train = self.parser_args.batch_size_train
        self.batch_size_test = self.parser_args.batch_size_test

        self.session = tf.Session()
        LOGGER.info("INIT END: " + str(time.time()))


    def train(self, dataset, desired_batches=None):
        LOGGER.info("TRAINING START: " + str(time.time()))
        LOGGER.info("NUM SAMPLES: " + str(desired_batches))

        self.train_round += 1

        # initial config during first round
        if int(self.train_round) == 1:
            self.late_init(dataset)

        torch.set_grad_enabled(True)
        self.model.train()
        self.model.eval()
        self.model.train()

        finish_loop = False

        if self.train_batches >= desired_batches:
            return self.train_batches

        LOGGER.info('TRAIN BATCH START')
        while not finish_loop:
            # Set train mode before we go into the train loop over an epoch
            for i, (data, labels) in enumerate(self.dl_train):
                self.optimizer.zero_grad()
                output = self.model(data.cuda())
                labels = format_labels(labels, self.parser_args).cuda()
                loss = self.criterion(output, labels)
                print(loss)
                loss.backward()
                self.optimizer.step()
                self.train_counter += self.batch_size_train
                self.train_batches += 1
                #LOGGER.info('TRAIN BATCH #{0}:\t{1}'.format(i, loss))

                if self.train_batches > desired_batches:
                    finish_loop = True
                    break

            #subprocess.run(['nvidia-smi'])
            LOGGER.info("TRAIN COUNTER: " + str(self.train_counter))
            self.train_round += 1

        LOGGER.info('TRAIN BATCH END')
        LOGGER.info('LR: ')
        for param_group in self.optimizer.param_groups:
            LOGGER.info(param_group['lr'])
        LOGGER.info("TRAINING FRAMES PER SEC: " + str(self.train_counter/(time.time()-self.time_start)))
        LOGGER.info("TRAINING COUNTER: " + str(self.train_counter))
        LOGGER.info("TRAINING BATCHES: " + str(self.train_batches))
        LOGGER.info("TRAINING END: " + str(time.time()))

        return self.train_batches


    def late_init(self, dataset):
        LOGGER.info('INIT')

        ds_temp = TFDataset(session=self.session, dataset=dataset)
        self.info = ds_temp.scan(50)

        LOGGER.info('AVG SHAPE: ' + str(self.info['avg_shape']))

        if self.info['is_multilabel']:
            setattr(self.parser_args, 'classification_type', 'multilabel')
        else:
            setattr(self.parser_args, 'classification_type', 'multiclass')

        self.model = get_model(parser_args=self.parser_args,
                               num_classes=self.num_classes)
        self.input_size = get_input_size(parser_args=self.parser_args)
        self.optimizer = get_optimizer(model=self.model,
                                       parser_args=self.parser_args)
        self.criterion = get_loss_criterion(parser_args=self.parser_args)

        self.dl_train, self.batch_size_train = get_dataloader(model=self.model, dataset=dataset, session=self.session,
                                                   is_training=True, first_round=(int(self.train_round) == 1),
                                                   batch_size=self.batch_size_train, input_size=self.input_size,
                                                   num_samples=int(10000000))


    def test(self, dataset):
        LOGGER.info("TESTING START: " + str(time.time()))

        self.test_round += 1

        if int(self.test_round) == 1:
            scan_start = time.time()
            ds_temp = TFDataset(session=self.session, dataset=dataset, num_samples=10000000)
            info = ds_temp.scan()
            self.num_samples_test = info['num_samples']
            LOGGER.info('SCAN TIME: ' + str(time.time() - scan_start))
            LOGGER.info('TESTING: FIRST ROUND')

        torch.set_grad_enabled(False)
        self.model.eval()
        dl, self.batch_size_test = get_dataloader(model=self.model, dataset=dataset, session=self.session,
                                                  is_training=False, first_round=(int(self.test_round) == 1),
                                                  batch_size=self.batch_size_test, input_size=self.input_size,
                                                  num_samples=self.num_samples_test)

        LOGGER.info('TEST BATCH START')
        prediction_list = []
        for i, (data, _) in enumerate(dl):
            LOGGER.info('TEST: ' + str(i))
            prediction_list.append(self.model(data.cuda()).cpu())
        predictions = np.vstack(prediction_list)
        LOGGER.info('TEST BATCH END')

        LOGGER.info("TESTING END: " + str(time.time()))
        return predictions

