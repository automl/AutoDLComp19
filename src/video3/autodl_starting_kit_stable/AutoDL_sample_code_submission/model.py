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
import argparse
import _pickle as pickle
from transforms import *
from dataset_kakaobrain import TFDataset
from utils import LOGGER

parser = argparse.ArgumentParser()

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
        setattr(self._parser_args, 'save_file_dir', os.path.join(rootpath, 'pretrained_models/'))
        setattr(self._parser_args, 'arch', 'squeezenet_64')
        setattr(self._parser_args, 'batch_size_test', 256)
        setattr(self._parser_args, 'optimizer', 'Adam')
        setattr(self._parser_args, 'dropout', 1e-3)
        setattr(self._parser_args, 't_diff', 1.0 / 50)
        setattr(self._parser_args, 'lr', 0.005)
        setattr(self._parser_args, 'momentum', 0.9)
        setattr(self._parser_args, 'weight_decay', 1e-6)
        setattr(self._parser_args, 'nesterov', True)

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
        LOGGER.info("INIT START: " + str(time.time()))
        super().__init__()

        self.time_start = time.time()
        self.train_time = []
        self.test_time = []

        self.done_training = False

        self.metadata = metadata
        self.num_classes = self.metadata.get_output_size()

        parser = ParserMock()
        parser.set_attr('num_classes', self.num_classes)

        self.parser_args = parser.parse_args()

        self.training_round = 0  # flag indicating if we are in the first round of training
        self.testing_round = 0
        self.num_samples_testing = None
        self.train_counter = 0
        self.batch_size_train = self.parser_args.batch_size_train
        self.batch_size_test = self.parser_args.batch_size_test

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

        torch.set_grad_enabled(True)
        self.model.train()
        if int(self.training_round) == 1:
            dl = self.get_dataloader(dataset, is_training=True, first_round=True)
        else:
            dl = self.get_dataloader(dataset, is_training=True, first_round=False)

        t3 = time.time()

        t_train = time.time()
        finish_loop = False

        LOGGER.info('TRAIN BATCH START')
        while not finish_loop:
            # Set train mode before we go into the train loop over an epoch
            for i, (data, labels) in enumerate(dl):
                self.optimizer.zero_grad()
                output = self.model(data.cuda())
                labels = format_labels(labels, self.parser_args).cuda()
                loss = self.criterion(output, labels)
                loss.backward()
                self.optimizer.step()
                self.train_counter += self.batch_size_train

                LOGGER.info('TRAIN BATCH #{0}:\t{1}'.format(i, loss))

                t_diff = (transform_to_time_rel(time.time() - self.time_start)
                        - transform_to_time_rel(t_train - self.time_start))

                if t_diff > self.parser_args.t_diff:
                    finish_loop = True
                    break

            subprocess.run(['nvidia-smi'])
            self.training_round += 1
        LOGGER.info('TRAIN BATCH END')

        t4 = time.time()
        LOGGER.info(
            '\nTIMINGS TRAINING: ' +
            '\n t2-t1 ' + str(t2 - t1) +
            '\n t3-t2 ' + str(t3 - t2) +
            '\n t4-t3 ' + str(t4 - t3))

        LOGGER.info('LR: ')
        for param_group in self.optimizer.param_groups:
            LOGGER.info(param_group['lr'])
        LOGGER.info("TRAINING FRAMES PER SEC: " + str(self.train_counter/(time.time()-self.time_start)))
        LOGGER.info("TRAINING COUNTER: " + str(self.train_counter))
        LOGGER.info("TRAINING END: " + str(time.time()))
        self.train_time.append(t4 - t1)


    def late_init(self, dataset):
        LOGGER.info('INIT')
        # show directory structure
        # for root, subdirs, files in os.walk(os.getcwd()):
        #     LOGGER.info(root)

        # get multiclass/multilabel information based on a small subset of videos/images

        t1 = time.time()

        ds_temp = TFDataset(session=self.session, dataset=dataset)
        self.info = ds_temp.scan(50)

        LOGGER.info('AVG SHAPE: ' + str(self.info['avg_shape']))

        if self.info['is_multilabel']:
            setattr(self.parser_args, 'classification_type', 'multilabel')
        else:
            setattr(self.parser_args, 'classification_type', 'multiclass')

        self.model = self.get_model()
        self.input_size = self.get_input_size()
        self.optimizer = self.get_optimizer(self.model)
        self.criterion = self.get_loss_criterion()

        t2 = time.time()

        LOGGER.info(
            '\nTIMINGS FIRST ROUND: ' +
            '\n t2-t1 ' + str(t2 - t1))


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
            info = ds_temp.scan()
            self.num_samples_testing = info['num_samples']
            LOGGER.info('SCAN TIME: ' + str(time.time() - scan_start))
            LOGGER.info('TESTING: FIRST ROUND')

        t2 = time.time()

        torch.set_grad_enabled(False)
        self.model.eval()
        if int(self.testing_round) == 1:
            dl = self.get_dataloader(dataset, is_training=False, first_round=True)
        else:
            dl = self.get_dataloader(dataset, is_training=False, first_round=False)

        t3 = time.time()

        LOGGER.info('TEST BATCH START')
        prediction_list = []
        for i, (data, _) in enumerate(dl):
            LOGGER.info('TEST: ' + str(i))
            prediction_list.append(self.model(data.cuda()).cpu())
        predictions = np.vstack(prediction_list)
        LOGGER.info('TEST BATCH END')

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


    def get_model(self):
        '''
        select proper model based on information from the dataset (image/video, etc.)
        '''
        print('+++++++++++++ ARCH ++++++++++++++')
        print(self.parser_args.arch)

        # model = self.load_model(model, os.path.join(self.parser_args.save_file_path, save_name))
        # optimizer = self.load_opimizer(model, self.parser_args)

        if 'squeezenet' in self.parser_args.arch:
            from torchvision.models import squeezenet1_1
            model = squeezenet1_1(pretrained=False,
                                  num_classes=self.num_classes).cuda()
            if '64' in self.parser_args.arch:
                save_file = 'imagenet_squeezenet_epochs_87_input_64_bs_256_SGD_ACC_33_7.pth'
            elif '128' in self.parser_args.arch:
                save_file = 'imagenet_squeezenet_epochs_128_input_128_bs_256_SGD_ACC_51.pth'
            elif '224' in self.parser_args.arch:
                save_file = 'imagenet_squeezenet_epochs_0_input_224_bs_256_SGD_ACC_58_3.pth'

        elif 'shufflenet05' in self.parser_args.arch:
            from torchvision.models import shufflenet_v2_x0_5
            model = shufflenet_v2_x0_5(pretrained=False,
                                            num_classes=self.num_classes).cuda()
            if '64' in self.parser_args.arch:
                save_file = 'imagenet_shufflenet05_epochs_48_input_64_bs_512_SGD_ACC_25_8.pth'
            elif '128' in self.parser_args.arch:
                save_file = 'imagenet_shufflenet05_epochs_111_input_128_bs_512_SGD_ACC_47_3.pth'
            elif '224' in self.parser_args.arch:
                save_file = 'imagenet_shufflenet05_epochs_0_input_224_bs_512_SGD_ACC_48_9.pth'

        elif 'shufflenet10' in self.parser_args.arch:
            from torchvision.models import shufflenet_v2_x1_0
            model = shufflenet_v2_x1_0(pretrained=False,
                                            num_classes=self.num_classes).cuda()
            if '64' in self.parser_args.arch:
                save_file = 'imagenet_shufflenet10_epochs_5_input_64_bs_512_SGD_ACC_30_5.pth'
            elif '128' in self.parser_args.arch:
                save_file = 'imagenet_shufflenet10_epochs_10_input_128_bs_512_SGD_ACC_54_8.pth'
            elif '224' in self.parser_args.arch:
                save_file = 'imagenet_shufflenet10_epochs_0_input_224_bs_512_SGD_ACC_63_4.pth'

        elif 'shufflenet20' in self.parser_args.arch:
            from torchvision.models import shufflenet_v2_x2_0
            model = shufflenet_v2_x2_0(pretrained=False,
                                            num_classes=self.num_classes).cuda()
            if '64' in self.parser_args.arch:
                save_file = 'imagenet_shufflenet20_epochs_114_input_64_bs_512_SGD_ACC_46_8.pth'
            elif '128' in self.parser_args.arch:
                save_file = 'imagenet_shufflenet20_epochs_115_input_128_bs_512_SGD_ACC_61_5.pth'
            elif '224' in self.parser_args.arch:
                save_file = 'imagenet_shufflenet20_epochs_110_input_224_bs_512_SGD_ACC_68.pth'

        elif 'resnet18' in self.parser_args.arch:
            from torchvision.models import resnet18
            model = resnet18(pretrained=False,
                             num_classes=self.num_classes).cuda()
            if '64' in self.parser_args.arch:
                save_file = 'imagenet_resnet18_epochs_88_input_64_bs_256_SGD_ACC_41_4.pth'
            elif '128' in self.parser_args.arch:
                save_file = 'imagenet_resnet18_epochs_67_input_128_bs_256_SGD_ACC_63.pth'
            elif '224' in self.parser_args.arch:
                save_file = 'imagenet_resnet18_epochs_0_input_224_bs_256_SGD_ACC_68_8.pth'

        elif 'mobilenetv2_64' in self.parser_args.arch:
            from torchvision.models import mobilenet_v2
            model = mobilenet_v2(pretrained=False,
                                width_mult=0.25).cuda()
            model.classifier[1] = torch.nn.Linear(1280, self.num_classes)
            save_file = 'imagenet_mobilenetv2_epochs_83_input_64_bs_256_SGD_ACC_24_8.pth'

        elif 'efficientnet' in self.parser_args.arch:
            if 'b07' in self.parser_args.arch:
                scale = 0.7
                if '64' in self.parser_args.arch:
                    save_file = 'imagenet_efficientnet_b07_epochs_104_input_64_bs_256_SGD_ACC_33_8.pth'
                elif '128' in self.parser_args.arch:
                    save_file = 'imagenet_efficientnet_b07_epochs_129_input_128_bs_256_SGD_ACC_53_3.pth'
                elif '224' in self.parser_args.arch:
                    save_file = 'imagenet_efficientnet_b07_epochs_128_input_224_bs_256_SGD_ACC_62_6.pth'
            elif 'b05' in self.parser_args.arch:
                scale = 0.5
                if '64' in self.parser_args.arch:
                    save_file = 'imagenet_efficientnet_b05_epochs_129_input_64_bs_256_SGD_ACC_24.pth'
                elif '128' in self.parser_args.arch:
                    save_file = 'imagenet_efficientnet_b05_epochs_125_input_128_bs_256_SGD_ACC43_8_.pth'
                elif '224' in self.parser_args.arch:
                    save_file = 'imagenet_efficientnet_b05_epochs_126_input_224_bs_256_SGD_ACC_54_2.pth'
            elif 'b03' in self.parser_args.arch:
                scale = 0.3
                if '64' in self.parser_args.arch:
                    save_file = 'imagenet_efficientnet_b03_epochs_128_input_64_bs_256_SGD_ACC_15.pth'
                elif '128' in self.parser_args.arch:
                    save_file = 'imagenet_efficientnet_b03_epochs_127_input_128_bs_256_SGD_ACC_28_5.pth'
                elif '224' in self.parser_args.arch:
                    save_file = 'imagenet_efficientnet_b03_epochs_129_input_224_bs_256_SGD_ACC_38_2.pth'
            elif 'b0' in self.parser_args.arch:
                scale = 1
                if '64' in self.parser_args.arch:
                    save_file = 'imagenet_efficientnet_b0_epochs_27_input_64_bs_256_SGD_ACC_41_5.pth'
                elif '128' in self.parser_args.arch:
                    save_file = 'imagenet_efficientnet_b0_epochs_115_input_128_bs_256_SGD_ACC_52.pth'
                elif '224' in self.parser_args.arch:
                    save_file = 'imagenet_efficientnet_b0_epochs_185_input_224_bs_256_SGD_ACC_67_5.pth'

            from models_efficientnet import EfficientNet
            model = EfficientNet(num_classes=self.num_classes, width_coef=scale,
                                 depth_coef=scale, scale=scale, dropout_ratio=self.parser_args.dropout,
                                 pl=0.2, arch='fullEfficientnet').cuda()

        elif 'densenet05' in self.parser_args.arch:
            from torchvision.models import DenseNet
            model = DenseNet(growth_rate=16, block_config=(3, 6, 12, 8),
                             num_init_features=64, bn_size=2, drop_rate=self.parser_args.dropout,
                             num_classes=self.num_classes).cuda()
            if '64' in self.parser_args.arch:
                save_file = 'imagenet_densenet05_epochs_117_input_64_bs_256_SGD_ACC_24.pth'
            elif '128' in self.parser_args.arch:
                save_file = 'imagenet_densenet05_epochs_124_input_128_bs_256_SGD_ACC_44_7.pth'
            elif '224' in self.parser_args.arch:
                save_file = 'imagenet_densenet05_epochs_122_input_224_bs_256_SGD_ACC_50.pth'

        elif 'densenet025' in self.parser_args.arch:
            from torchvision.models import DenseNet
            model = DenseNet(growth_rate=8, block_config=(2, 4, 8, 4),
                             num_init_features=32, bn_size=2, drop_rate=self.parser_args.dropout,
                             num_classes=self.num_classes).cuda()
            if '64' in self.parser_args.arch:
                save_file = 'imagenet_densenet025_epochs_113_input_64_bs_256_SGD_ACC_17_3.pth'
            elif '128' in self.parser_args.arch:
                save_file = 'imagenet_densenet025_epochs_133_input_128_bs_256_SGD_ACC_23_9.pth'

        elif 'densenet' in self.parser_args.arch:
            from torchvision.models import densenet121
            model = densenet121(pretrained=False,
                                num_classes=self.num_classes,
                                drop_rate=self.parser_args.dropout).cuda()
            if '64' in self.parser_args.arch:
                save_file = 'imagenet_densenet_epochs_139_input_64_bs_256_SGD_ACC_52_8.pth'
            elif '128' in self.parser_args.arch:
                save_file = 'imagenet_densenet_epochs_90_input_128_bs_256_SGD_ACC_63_8.pth'
            elif '224' in self.parser_args.arch:
                save_file = 'imagenet_densenet_epochs_0_input_224_bs_256_SGD_ACC_72_7.pth'

        else:
            raise TypeError('Unknown model type')

        model = self.load_model(model, os.path.join(self.parser_args.save_file_dir, save_file)).cuda()

        return model


    def get_input_size(self):
        if '64' in self.parser_args.arch:
            return 64
        elif '128' in self.parser_args.arch:
            return 128
        elif '224' in self.parser_args.arch:
            return 224


    def get_loss_criterion(self):
        if self.parser_args.classification_type == 'multiclass':
            return torch.nn.CrossEntropyLoss().cuda()
        elif self.parser_args.classification_type == 'multilabel':
            return torch.nn.BCEWithLogitsLoss().cuda()
        else:
            raise ValueError("Unknown loss type")


    def get_optimizer(self, model):
        if self.parser_args.optimizer == 'SGD':
            return torch.optim.SGD(model.parameters(),
                                    self.parser_args.lr,
                                    momentum = self.parser_args.momentum,
                                    weight_decay = self.parser_args.weight_decay,
                                    nesterov = self.parser_args.nesterov)
        elif self.parser_args.optimizer == 'Adam':
            return torch.optim.Adam(model.parameters(),
                                         self.parser_args.lr)
        else:
            raise ValueError("Unknown optimizer type")


    def load_model(self, model, save_file):
        #################################################################################################
        pretrained_dict = torch.load(save_file)
        model_dict = model.state_dict()
        # 1. filter out unnecessary keys and if different classes
        new_state_dict = {}
        for k1, v in pretrained_dict.items():
            for k2 in model_dict.keys():
                k = k1.replace('module.', '')
                if k2 in k and (v.size() == model_dict[k2].size()):
                    new_state_dict[k2] = v
        # If explicitely delete last fully connected layed(like finetuning with new
        # dataset that has equal amount of classes
        # if remove_last_fc:
        #     temp_keys = []
        #     for k in new_state_dict.keys():
        #         if 'last' in k:
        #             temp_keys.append(k)
        #     for k in temp_keys:
        #         del new_state_dict[k]

        un_init_dict_keys = [k for k in model_dict.keys() if k
                             not in new_state_dict]

        print("un_init_dict_keys: ", un_init_dict_keys)
        print("\n------------------------------------")

        for k in un_init_dict_keys:
            new_state_dict[k] = torch.DoubleTensor(
                model_dict[k].size()).zero_()
            if 'weight' in k:
                if 'bn' in k:
                    print("{} init as: 1".format(k))
                    torch.nn.init.constant_(new_state_dict[k], 1)
                else:

                    print("{} init as: xavier".format(k))
                    try:
                        torch.nn.init.xavier_uniform_(new_state_dict[k])
                    except Exception:
                        torch.nn.init.constant_(new_state_dict[k], 1)
            elif 'bias' in k:

                print("{} init as: 0".format(k))
                torch.nn.init.constant_(new_state_dict[k], 0)
            print("------------------------------------")
        model.load_state_dict(new_state_dict)
        print('loaded model')

        return model

    def get_transform(self, is_training):
        if is_training:
            return torchvision.transforms.Compose([
                SelectSample(),
                AlignAxes(),
                FormatChannels(channels_des = 3),
                ToPilFormat(),
                #SaveImage(save_dir = self.parser_args.save_file_dir, suffix='_2'),
                torchvision.transforms.RandomResizedCrop(size = self.input_size),
                #SaveImage(save_dir=self.parser_args.save_file_dir, suffix='_3'),
                torchvision.transforms.RandomHorizontalFlip(),
                #SaveImage(save_dir=self.parser_args.save_file_dir, suffix='_4'),
                torchvision.transforms.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.01),
                #SaveImage(save_dir=self.parser_args.save_file_dir, suffix='_5'),
                ToTorchFormat(),
                torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
                #SaveImage(save_dir=self.parser_args.save_file_dir, suffix='_6')])
        else:
            return torchvision.transforms.Compose([
                SelectSample(),
                AlignAxes(),
                FormatChannels(3),
                Normalize(),
                ToPilFormat(),
                torchvision.transforms.RandomResizedCrop(size=self.input_size),
                torchvision.transforms.Resize(int(self.input_size*1.1)),
                torchvision.transforms.CenterCrop(self.input_size),
                ToTorchFormat(),
                torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


    def get_dataloader(self, dataset, is_training, first_round):
        if is_training:
            batch_size = self.batch_size_train
        else:
            batch_size = self.batch_size_test

        transform = self.get_transform(is_training)

        num_samples = int(10000000) if is_training else self.num_samples_testing
        shuffle = True if is_training else False
        drop_last = False

        ds = TFDataset(
            session=self.session,
            dataset=dataset,
            num_samples=num_samples,
            transform=transform
        )

        if first_round:
            batch_size_ok = False

            while not batch_size_ok and batch_size > 1:
                ds.reset()
                try:
                    dl = torch.utils.data.DataLoader(
                        ds,
                        batch_size=batch_size,
                        shuffle=shuffle,
                        drop_last=drop_last
                    )

                    data, labels = next(iter(dl))
                    self.model(data.cuda())

                    batch_size_ok = True

                except RuntimeError:
                    batch_size = int(batch_size/2)
                    if is_training:
                        LOGGER.info('REDUCING BATCH SIZE FOR TRAINING TO: ' + str(batch_size))
                    else:
                        LOGGER.info('REDUCING BATCH SIZE FOR TESTING TO: ' + str(batch_size))

            if is_training:
                self.batch_size_train = batch_size
            else:
                self.batch_size_test = batch_size

        LOGGER.info('USING BATCH SIZE: ' + str(batch_size))
        ds.reset()
        dl = torch.utils.data.DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last
        )

        return dl


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
