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
import hpbandster.core.result as hpres
import hpbandster.visualization as hpvis
import subprocess
import json
from common.utils import *
from bohb_classification_normal import WrapperModel_dl, load_transform, Identity


datasets = ['binary_alpha_digits', 'caltech101', 'caltech_birds2010', 'caltech_birds2011',
            'cats_vs_dogs', 'cifar10', 'cifar100', 'coil100',
            'colorectal_histology', 'deep_weeds', 'eurosat', 'emnist',
            'fashion_mnist', 'horses_or_humans', 'kmnist', 'mnist',
            'oxford_flowers102', 'oxford_iiit_pet', 'patch_camelyon', 'rock_paper_scissors',
            'smallnorb', 'svhn_cropped', 'tf_flowers', 'uc_merced',
            'Hmdb51', 'Ucf101', 'SMv2']

def load_precalculated_results(log_dir):
    result_dict = json.load(open(os.path.join(log_dir,"result_dict.json"),"r"))
    best_models = json.load(open(os.path.join(log_dir,"best_models.json"),"r"))

    return result_dict, best_models


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


class Model(object):
    """Trivial example of valid model. Returns all-zero predictions."""

    def __init__(self, metadata):
        LOGGER.info("INIT START: " + str(time.time()))
        super().__init__()

        self.time_start = time.time()

        self.metadata = metadata
        self.num_classes = self.metadata.get_output_size()

        parser = ParserMock()
        parser.set_attr('momentum', 0.9)
        parser.set_attr('weight_decay', 1e-6)
        parser.set_attr('nesterov', True)
        parser.set_attr('num_classes', self.num_classes)
        parser.set_attr('t_diff', 1.0/50)
        parser.set_attr('log_dir', os.getcwd())
        parser.set_attr('file_dir', os.path.join(os.getcwd(), 'common', 'files'))
        parser.set_attr('bohb_sample_size', 32)
        parser.set_attr('bohb_log_dir', os.path.join(os.getcwd(), 'dl_logs', str(parser._parser_args.bohb_sample_size)))
        parser.set_attr('bohb_config_id', (6,0,2))
        self.parser_args = parser.parse_args()

        self.done_training = False
        self.train_counter = 0
        self.train_batches = 0
        self.dl_train = None
        self.batch_size_train = 64
        self.batch_size_test = 512

        self.train_round = 0
        self.test_round = 0
        self.num_samples_testing = None

        self.session = tf.Session()
        LOGGER.info("INIT END: " + str(time.time()))


    def train(self, dataset, remaining_time_budget=None):
        LOGGER.info("TRAINING START: " + str(time.time()))

        self.train_round += 1

        # initial config during first round
        if int(self.train_round) == 1:
            self.late_init(dataset)

        torch.set_grad_enabled(True)
        self.model.train()

        t_train = time.time()
        brk = False

        while brk == False:
            for i, (data, labels) in enumerate(self.dl_train):
                self.optimizer.zero_grad()

                output = self.model(data.cuda())
                labels = format_labels(labels, self.parser_args).cuda()

                loss = self.criterion(output, labels)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                #if i == 0:
                #    subprocess.run(["nvidia-smi"])

                t_cur = time.time()

                t_diff = transform_time_abs(t_cur - self.time_start) - \
                         transform_time_abs(t_train - self.time_start)

                if t_diff > self.parser_args.t_diff:
                    brk = True
                    break

        LOGGER.info("TRAINING END: " + str(time.time()))

        return self.train_batches



    def find_optimized_parameters(self, dataset):
        # load config parameters
        result = hpres.logged_results_to_HBS_result(self.parser_args.bohb_log_dir)
        id2conf = result.get_id2config_mapping()
        bohb_conf = id2conf[self.parser_args.bohb_config_id]['config']
        class_conf = {'bohb_log_dir': self.parser_args.bohb_log_dir}
        class_conf.update(bohb_conf)

        trans_conf = {'model_input_size': 128, 'transform_scale': 0.7, 'transform_ratio': 0.75}
        transform = load_transform(trans_conf, is_training=False)
        ds_temp = TFDataset(session=self.session, dataset=dataset, num_samples=100000, transform=transform)
        dl_temp = torch.utils.data.DataLoader(
            ds_temp,
            batch_size=self.parser_args.bohb_sample_size,
            shuffle=class_conf['shuffle_data'],
            drop_last=False
        )
        model_temp = torchvision.models.resnet18(pretrained=True)
        model_temp.fc = Identity()

        class_temp = WrapperModel_dl(config_id=self.parser_args.bohb_config_id, num_classes=27, cfg=class_conf)
        out1 = model_temp(next(iter(dl_temp))[0])
        out2 = class_temp(out1)
        dataset = datasets[np.argmax(out2.data)]

        result_dict, best_models = load_precalculated_results(self.parser_args.log_dir)

        result_list = result_dict[dataset]
        best_perf = 0
        best_tup = None
        for tup in result_list:
            if tup[0] in best_models:
                if tup[1] > best_perf:
                    best_perf = tup[1]
                    best_tup = tup

        best_model  = best_tup[0]
        best_params = eval(best_tup[2])

        print('------------')
        print(dataset)
        print(best_model)
        print(best_params)
        print('------------')

        return best_model, best_params


    def late_init(self, dataset):
        LOGGER.info('INIT')

        ds_temp = TFDataset(session=self.session, dataset=dataset, num_samples=100000)
        self.info = ds_temp.scan(10)

        LOGGER.info('AVG SHAPE: ' + str(self.info['avg_shape']))

        if self.info['is_multilabel']:
            setattr(self.parser_args, 'classification_type', 'multilabel')
        else:
            setattr(self.parser_args, 'classification_type', 'multiclass')

        best_model, best_params = self.find_optimized_parameters(dataset)
        setattr(self.parser_args, 'model', best_model)
        setattr(self.parser_args, 'optimizer', best_params['optimizer'])
        setattr(self.parser_args, 'dropout', best_params['dropout'])
        setattr(self.parser_args, 'lr', best_params['lr'])

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


    def test(self, dataset, remaining_time_budget=None):
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



