import os
import sys

sys.path.append(os.path.join(os.getcwd(), 'AutoDL_ingestion_program'))
sys.path.append(os.path.join(os.getcwd(), 'AutoDL_scoring_program'))

import pickle
import xgboost as xgb
import tensorflow as tf
import random
import numpy as np
import time
import torchvision
import torch
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
import hpbandster.core.nameserver as hpns
import hpbandster.core.result as hpres
from hpbandster.core.worker import Worker
from hpbandster.core.master import Master
from hpbandster.optimizers.config_generators.bohb import BOHB as BOHB
from hpbandster.optimizers.iterations import SuccessiveHalving
from common.dataset_kakaobrain import TFDataset
from AutoDL_ingestion_program.dataset import AutoDLDataset
from common.utils import ToTorchFormat, SaveImage, Stats, SelectSample, AlignAxes, FormatChannels, ToPilFormat

print(sys.path)

def split_datasets(datasets, fraction):
    '''
    split list of datasets into training/testing datasets
    '''
    train_datasets = []
    test_datasets = []
    for i in range(len(datasets)):
        if i%fraction == 0:
            test_datasets.append(datasets[i])
        else:
            train_datasets.append(datasets[i])
    return train_datasets, test_datasets


def get_configspace():
    cs = CS.ConfigurationSpace()
    # fc classifier
    cs.add_hyperparameter(CSH.CategoricalHyperparameter(name='nn_train_batches', choices=[1,2,4,8]))
    cs.add_hyperparameter(CSH.CategoricalHyperparameter(name='nn_train_batch_size', choices = [2,4,8,16,32,64,128,256,512,1024]))
    cs.add_hyperparameter(CSH.CategoricalHyperparameter(name='nn_test_batch_size', choices = [32]))
    #cs.add_hyperparameter(CSH.CategoricalHyperparameter(name='nn_width', choices = [1024, 200, 50]))
    #cs.add_hyperparameter(CSH.CategoricalHyperparameter(name='nn_num_hidden_layers', choices=[0,1,2]))
    cs.add_hyperparameter(CSH.UniformFloatHyperparameter(name='nn_lr', lower=1e-5, upper=1e-3, log=True))
    cs.add_hyperparameter(CSH.CategoricalHyperparameter(name='nn_use_med', choices=[False, True]))
    cs.add_hyperparameter(CSH.CategoricalHyperparameter(name='nn_use_mean', choices=[False, True]))
    cs.add_hyperparameter(CSH.CategoricalHyperparameter(name='nn_use_var', choices=[False, True]))
    cs.add_hyperparameter(CSH.CategoricalHyperparameter(name='nn_use_std', choices=[False, True]))
    cs.add_hyperparameter(CSH.CategoricalHyperparameter(name='nn_use_skew', choices=[False, True]))
    cs.add_hyperparameter(CSH.CategoricalHyperparameter(name='nn_use_kurt', choices=[False, True]))
    cs.add_hyperparameter(CSH.UniformFloatHyperparameter(name='nn_cut_perc', lower=0, upper=0.4, log=False))


    # xgb classifier
    #cs.add_hyperparameter(CSH.UniformFloatHyperparameter(name='xgb_eta', lower=0, upper=1, log=False))
    #cs.add_hyperparameter(CSH.UniformIntegerHyperparameter(name='xgb_max_depth', lower=3, upper=10))
    #cs.add_hyperparameter(CSH.UniformIntegerHyperparameter(name='xgb_train_steps', lower=2, upper=20))
    #cs.add_hyperparameter(CSH.CategoricalHyperparameter(name='xgb_train_batch_size', choices = [32,64,128]))
    #cs.add_hyperparameter(CSH.CategoricalHyperparameter(name='xgb_test_batch_size', choices = [32,64,128]))

    # common parameters
    #cs.add_hyperparameter(CSH.UniformFloatHyperparameter(name='transform_scale', lower=0.3, upper=1, log=False))
    #cs.add_hyperparameter(CSH.UniformFloatHyperparameter(name='transform_ratio', lower=0.3, upper=1, log=False))
    cs.add_hyperparameter(CSH.CategoricalHyperparameter(name='shuffle_data', choices=[True, False]))

    return cs


def get_configuration(log_subfolder=None):
    cfg = {}
    cluster_mode = False
    if cluster_mode:
        cfg["code_dir"] = '/home/nierhoff/AutoDLComp19/src/video3/autodl_starting_kit_stable/AutoDL_sample_code_submission'
        cfg['proc_dataset_dir'] = '/data/aad/image_datasets/thomas_processed_datasets/1e5'
        cfg['des_num_samples'] = int(1e5)
        cfg["image_dir"] = '/data/aad/image_datasets/challenge'
        cfg["video_dir"] = '/data/aad/video_datasets/challenge'
        cfg["bohb_interface"] = 'eth0'
        cfg["bohb_workers"] = 10
    else:
        cfg["code_dir"] = '/home/dingsda/autodl/AutoDLComp19/src/video3/autodl_starting_kit_stable'
        cfg['proc_dataset_dir'] = '/home/dingsda/data/datasets/processed_datasets/1e5'
        cfg['des_num_samples'] = int(1e5)
        cfg["image_dir"] = '/home/dingsda/data/datasets/challenge/image'
        cfg["video_dir"] = '/home/dingsda/data/datasets/challenge/video'
        cfg["bohb_interface"] = 'lo'
        cfg["bohb_workers"] = 3

    log_folder = "dl_logs"
    if log_subfolder == None:
        cfg["bohb_log_dir"] = os.path.join(os.getcwd(), log_folder, str(int(time.time())))
    else:
        cfg["bohb_log_dir"] = os.path.join(os.getcwd(), log_folder, log_subfolder)

    cfg["use_model"] = 'dl'    # xgb or dl

    if cfg["use_model"] == 'dl':
        cfg["bohb_min_budget"] = 100
        cfg["bohb_max_budget"] = 1000
    else:
        cfg["bohb_min_budget"] = 1000
        cfg["bohb_max_budget"] = 10000
    cfg["bohb_iterations"] = 10
    cfg["bohb_run_id"] = '123'
    cfg['model_input_size'] = 128
    cfg['transform_scale'] = 0.7
    cfg['transform_ratio'] = 0.75
    cfg['nn_lr'] = 1e-4
    cfg['nn_train_batches'] = 1
    cfg['nn_train_batch_size'] = 128
    cfg['nn_num_hidden_layers'] = 0
    cfg['nn_use_last_resnet_layer'] = False
    cfg['xgb_eta'] = 0.3
    cfg['xgb_max_depth'] = 6
    cfg['xgb_train_steps'] = 10
    cfg['xgb_train_batch_size'] = 10

    datasets = ['binary_alpha_digits', 'caltech101', 'caltech_birds2010',
               'cats_vs_dogs', 'cifar10', 'cifar100', 'coil100',
               'colorectal_histology', 'deep_weeds', 'eurosat',
               'fashion_mnist', 'horses_or_humans', 'kmnist', 'mnist',
               'oxford_iiit_pet', 'patch_camelyon', 'rock_paper_scissors',
               'smallnorb', 'svhn_cropped', 'tf_flowers', 'uc_merced']
    # without 'Munster', 'Kraut', 'Chucky', 'Decal', 'Hammer', 'Katze', 'Kreatur', 'Pedro', 'Hmdb51', 'Ucf101', 'SMv2' 'oxford_flowers102', 'emnist', 'caltech_birds2011',

    #train_datasets, test_datasets = split_datasets(datasets, 4)
    train_datasets = datasets
    test_datasets = []
    cfg["train_datasets"] = train_datasets
    cfg["test_datasets"] = test_datasets

    return cfg


def copy_config_to_cfg(cfg, config):
    for key, value in config.items():
        cfg[key] = value
    return cfg


class BohbWrapper(Master):
    def __init__(self, configspace=None,
                 eta=3, min_budget=0.01, max_budget=1,
                 min_points_in_model=None, top_n_percent=15,
                 num_samples=64, random_fraction=1 / 3, bandwidth_factor=3,
                 min_bandwidth=1e-3,
                 **kwargs):

        # TODO: Proper check for ConfigSpace object!
        if configspace is None:
            raise ValueError("You have to provide a valid CofigSpace object")

        cg = BOHB(configspace=configspace,
                  min_points_in_model=min_points_in_model,
                  top_n_percent=top_n_percent,
                  num_samples=num_samples,
                  random_fraction=random_fraction,
                  bandwidth_factor=bandwidth_factor,
                  min_bandwidth=min_bandwidth
                  )

        super().__init__(config_generator=cg, **kwargs)

        # Hyperband related stuff
        self.eta = eta
        self.min_budget = min_budget
        self.max_budget = max_budget

        # precompute some HB stuff
        self.max_SH_iter = -int(np.log(min_budget / max_budget) / np.log(eta)) + 1
        self.budgets = max_budget * np.power(eta, -np.linspace(self.max_SH_iter - 1, 0, self.max_SH_iter))

        self.config.update({
            'eta': eta,
            'min_budget': min_budget,
            'max_budget': max_budget,
            'budgets': self.budgets,
            'max_SH_iter': self.max_SH_iter,
            'min_points_in_model': min_points_in_model,
            'top_n_percent': top_n_percent,
            'num_samples': num_samples,
            'random_fraction': random_fraction,
            'bandwidth_factor': bandwidth_factor,
            'min_bandwidth': min_bandwidth
        })

    def get_next_iteration(self, iteration, iteration_kwargs={}):

        # number of 'SH rungs'
        s = self.max_SH_iter - 1
        # number of configurations in that bracket
        n0 = int(np.floor((self.max_SH_iter) / (s + 1)) * self.eta ** s)
        ns = [max(int(n0 * (self.eta ** (-i))), 1) for i in range(s + 1)]

        return (SuccessiveHalving(HPB_iter=iteration, num_configs=ns, budgets=self.budgets[(-s - 1):],
                                  config_sampler=self.config_generator.get_config, **iteration_kwargs))


class ProcessedDataset(torch.utils.data.Dataset):
    """
    Data loader for the ucf 101 dataset. It assumes that in the top-level folder there is another
    folder (aaa_recognition in our case) with the files for the test/train split
    """

    def __init__(self, data_path, class_index):
        with open(data_path, 'rb') as fh:
            self.dataset = torch.tensor(pickle.load(fh))
            self.class_index = torch.tensor(class_index)

    def get_dataset(self):
        # for compatibility
        return self

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx], self.class_index


def load_datasets_processed(cfg, datasets):
    '''
    load preprocessed datasets from a list, return train/test datasets, dataset index and dataset name
    '''
    dataset_list = []
    dataset_dir = cfg['proc_dataset_dir']
    class_index = 0

    for dataset_name in datasets:
        dataset_train_path = os.path.join(cfg['proc_dataset_dir'], dataset_name + '_train')
        dataset_test_path = os.path.join(cfg['proc_dataset_dir'], dataset_name + '_test')

        try:
            dataset_train = ProcessedDataset(dataset_train_path, class_index)
            dataset_test = ProcessedDataset(dataset_test_path, class_index)
        except Exception as e:
            print(e)
            continue

        dataset_list.append((dataset_train, dataset_test, dataset_name, class_index))
        class_index += 1

    return dataset_list


def load_datasets_raw(cfg, datasets):
    '''
    load datasets from a list, return train/test datasets, dataset index and dataset name
    '''

    dataset_list = []
    class_index = 0

    for dataset_name in datasets:
        challenge_image_dataset = os.path.join(cfg["image_dir"], dataset_name)
        challenge_video_dataset = os.path.join(cfg["video_dir"], dataset_name)

        if os.path.isdir(challenge_image_dataset):
            dataset_dir = challenge_image_dataset
        elif os.path.isdir(challenge_video_dataset):
            dataset_dir = challenge_video_dataset
        else:
            raise ValueError('unknown dataset: ' + str(dataset_name))

        lower_path = os.path.join(dataset_dir, (dataset_name+'.data').lower())
        capital_path = os.path.join(dataset_dir, dataset_name+'.data')
        try:
            if os.path.exists(lower_path):
                dataset_train = AutoDLDataset(os.path.join(lower_path, 'train'))
                dataset_test = AutoDLDataset(os.path.join(lower_path, 'test'))
            else:
                dataset_train = AutoDLDataset(os.path.join(capital_path, 'train'))
                dataset_test = AutoDLDataset(os.path.join(capital_path, 'test'))
        except Exception as e:
            print(e)
            continue
        dataset_list.append((dataset_train, dataset_test, dataset_name, class_index))
        class_index += 1

    return dataset_list


def load_transform(cfg, is_training):
    size = cfg["model_input_size"]
    scale = cfg["transform_scale"]
    ratio = cfg["transform_ratio"]

    if is_training:
        return torchvision.transforms.Compose([
            SelectSample(),
            AlignAxes(),
            FormatChannels(channels_des=3),
            ToPilFormat(),
            torchvision.transforms.RandomResizedCrop(size = size, scale=(scale, 1.0), ratio=(ratio, 1/ratio)),
            torchvision.transforms.RandomHorizontalFlip(),
            ToTorchFormat()])
    else:
        return torchvision.transforms.Compose([
            SelectSample(),
            AlignAxes(),
            FormatChannels(channels_des=3),
            ToPilFormat(),
            torchvision.transforms.Resize(size=(size, size)),
            ToTorchFormat()])

def load_dataloader(cfg, train_dataloader_dict, session, dataset, dataset_name, num_samples, is_training):
    print(dataset_name)

    transform = load_transform(cfg, is_training=is_training)

    ds = TFDataset(
        session=session,
        dataset=dataset,
        num_samples=num_samples,
        transform=transform
    )

    if is_training:
        batch_size = cfg['nn_train_batch_size']
    else:
        batch_size = cfg["nn_train_batch_size"] * 2

    if dataset_name not in train_dataloader_dict or is_training is False:
        # # reduce batch size until it fits into memory
        # batch_size_ok = False
        #
        # while not batch_size_ok and batch_size > 1:
        #     ds.reset()
        #     try:
        #         dl = torch.utils.data.DataLoader(
        #             ds,
        #             batch_size=batch_size,
        #             shuffle=False,
        #             drop_last=False
        #         )
        #
        #         data, labels = next(iter(dl))
        #         self.model(data.cuda())
        #
        #         batch_size_ok = True
        #
        #     except Exception as e:
        #         print(str(e))
        #         batch_size = int(batch_size / 2)
        #         print('REDUCING BATCH SIZE TO: ' + str(batch_size))
        # ds.reset()

        dl = torch.utils.data.DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False
        )
        train_dataloader_dict[dataset_name] = dl

    else:
        dl = train_dataloader_dict[dataset_name]

    return dl


class Identity(torch.nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class WrapperModel_dl(torch.nn.Module):
    def __init__(self, config_id, num_classes, cfg):
        super().__init__()
        self.cfg = cfg
        self.filename = self.cfg["bohb_log_dir"] + '/' + str(config_id[0]) + '_' + str(config_id[1]) + '_' + str(config_id[2]) + '_model' + '.pt'

        self.mode = 'train'
        self.timer_cum = 0
        self.timer_runs = 0

        mult = 0

        if self.cfg['nn_use_med']:
            mult += 1
        if self.cfg['nn_use_mean']:
            mult += 1
        if self.cfg['nn_use_std']:
            mult += 1
        if self.cfg['nn_use_var']:
            mult += 1
        if self.cfg['nn_use_skew']:
            mult += 1
        if self.cfg['nn_use_kurt']:
            mult += 1

        self.fc = torch.nn.Linear(512*mult, num_classes)

        if os.path.isfile(self.filename):
            self.load_state_dict(torch.load(self.filename))

    def eval(self):
        self.mode = 'eval'

    def train(self):
        self.mode = 'train'

    def forward(self, x):
        if self.mode == 'eval':
            t1 = time.time()

        nb_samples = x.shape[0]
        nb_cut = int(self.cfg['nn_cut_perc'] * nb_samples)
        x = x.sort(dim=0)[0]
        x = x[nb_cut:nb_samples-nb_cut]

        med  = x.median(dim=0)[0]
        mean = torch.mean(x, dim=0)
        var  = torch.var(x, dim=0)
        std  = torch.pow(var, 0.5)
        diff = x - mean
        tmp  = diff/(std+0.1)
        skew = torch.mean(torch.pow(tmp, 3.0), dim=0)
        kurt = torch.mean(torch.pow(tmp, 4.0), dim=0)

        x = None
        if self.cfg['nn_use_med']:
            x = med if x is None else torch.cat((x, med), 0)
        if self.cfg['nn_use_mean']:
            x = mean if x is None else torch.cat((x, mean), 0)
        if self.cfg['nn_use_std']:
            x = std if x is None else torch.cat((x, std), 0)
        if self.cfg['nn_use_var']:
            x = var if x is None else torch.cat((x, var), 0)
        if self.cfg['nn_use_skew']:
            x = skew if x is None else torch.cat((x, skew), 0)
        if self.cfg['nn_use_kurt']:
            x = kurt if x is None else torch.cat((x, kurt), 0)

        x = self.fc(x)
        x = x.unsqueeze(0)

        if self.mode == 'eval':
            t2 = time.time()
            self.timer_cum += t2-t1
            self.timer_runs += 1

        return x

    def save(self):
        torch.save(self.state_dict(), self.filename)

    def get_avg_time(self):
        return self.timer_cum / self.timer_runs


class WrapperModel_xgb(torch.nn.Module):
    def __init__(self, processed=False):
        super().__init__()
        self.model = torchvision.models.resnet18(pretrained=True)
        self.processed = processed

    def eval(self):
        self.model.eval()

    def train(self):
        self.model.train()

    def forward(self, x):
        if self.processed:
            x = self.model(x)
        x = torch.cat((torch.mean(x, dim=0), torch.var(x, dim=0)), 0)
        x = x.unsqueeze(0)
        return x


class WrapperOptimizer(object):
    def __init__(self, config_id, model, cfg):
        super().__init__()

        self.filename = cfg["bohb_log_dir"] + '/' + str(config_id[0]) + '_' + str(config_id[1]) + '_' + str(config_id[2]) + '_optimizer' + '.pt'

        self.optimizer =  torch.optim.Adam(model.parameters(),
                                           cfg['nn_lr'])

        if os.path.isfile(self.filename):
            self.optimizer.load_state_dict(torch.load(self.filename))

    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self):
        self.optimizer.step()

    def save(self):
        torch.save(self.optimizer.state_dict(), self.filename)


def calc_accuracy(prediction, class_index):
    max_val, max_idx = torch.max(prediction, 1)
    acc = float(torch.sum(max_idx == int(class_index))) / float(len(prediction))
    return acc


def execute_run_xgb(config_id, cfg, budget, dataset_list, session):
    num_classes = len(dataset_list)

    m = Model_xgb(config_id = config_id,
                  num_classes = num_classes,
                  cfg = cfg,
                  session = session)

    for i in range(num_classes):
        selected_class = i % num_classes
        dataset_train = dataset_list[selected_class][0]
        dataset_name  = dataset_list[selected_class][2]
        class_index   = dataset_list[selected_class][3]

        print(dataset_name)

        if class_index != selected_class:
            raise ValueError("class index mismatch: " + str(class_index) + ' ' + str(selected_class))

        m.collect_samples_train(dataset = dataset_train.get_dataset(),
                                class_index = class_index,
                                desired_batches = budget)

    m.train()

    acc_list = []
    for i in range(num_classes):
        selected_class = i % num_classes
        dataset_test = dataset_list[selected_class][1]
        dataset_name = dataset_list[selected_class][2]
        class_index  = dataset_list[selected_class][3]

        if class_index != selected_class:
            raise ValueError("class index mismatch: " + str(class_index) + ' ' + str(selected_class))

        acc = m.test(dataset=dataset_test.get_dataset(),
                     dataset_name=dataset_name,
                     class_index=class_index)
        acc_list.append(acc)

    avg_acc = sum(acc_list) / len(acc_list)

    return avg_acc, 0, 0


class Model_xgb(object):
    def __init__(self, config_id, num_classes, cfg, session):
        super().__init__()
        self.cfg = cfg
        self.train_dataloader_dict = {}
        self.session = session
        self.model = WrapperModel_xgb(processed = False)
        self.model.cuda()
        self.num_classes = num_classes
        self.X = []
        self.y = []

    def collect_samples_train(self, dataset, class_index, desired_batches):
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.cfg['xgb_train_batch_size'],
            shuffle=cfg['shuffle_data'],
            drop_last=False
        )

        torch.set_grad_enabled(False)
        self.model.eval()

        finish_loop = False
        train_batches = 0

        while not finish_loop:
            # Set train mode before we go into the train loop over an epoch
            for data, _ in dataloader:
                output = self.model(data.cuda()).cpu()
                self.X.append(output.numpy())
                self.y.append(np.array([class_index]))
                # if self.X is None:
                #     self.X = output
                # else:
                #     self.X = torch.cat((self.X, output), dim=0)
                #
                #
                # if self.y is None:
                #     self.y = torch.tensor([class_index])
                # else:
                #     self.y = torch.cat((self.y, torch.tensor([class_index])), dim=0)

                train_batches += 1
                if train_batches > desired_batches:
                    finish_loop = True
                    break


    def train(self):
        X = np.squeeze(np.array(self.X))
        y = np.squeeze(np.array(self.y))

        param = {
            'eta': self.cfg['xgb_eta'],
            'max_depth': self.cfg['xgb_max_depth'],
            'objective': 'multi:softprob',
            'num_class': self.num_classes}

        D_train = xgb.DMatrix(X, y)
        self.xgb = xgb.train(param, D_train, self.cfg['xgb_train_steps'])
        self.X = None
        self.y = None


    def test(self, dataset, dataset_name, class_index):
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.cfg['xgb_test_batch_size'],
            shuffle=False,
            drop_last=False
        )

        torch.set_grad_enabled(False)
        self.model.eval()

        output_list = []

        for data, _ in dataloader:
            output = self.model(data.cuda()).cpu()
            output_list.append(output.numpy())
        output = np.squeeze(np.array(output_list))
        output = xgb.DMatrix(output)
        prediction = torch.from_numpy(self.xgb.predict(output))

        acc = calc_accuracy(prediction, class_index)
        print("ACCURACY: " + str(dataset_name) + ' ' + str(acc))  # + ' ' + str(time.time() - t1))

        return acc



def execute_run_dl(config_id, cfg, budget, dataset_list, session):
    num_classes = len(dataset_list)

    m = Model_dl(config_id = config_id,
                 num_classes = num_classes,
                 cfg = cfg,
                 session = session)

    loss_list = []

    for i in range(int(budget)):
        # load training dataset
        selected_class = i % num_classes
        dataset_train  = dataset_list[selected_class][0]
        dataset_name   = dataset_list[selected_class][2]
        class_index    = dataset_list[selected_class][3]

        if i%10 == 0:
            print(str(i), end=' ')

        if class_index != selected_class:
            raise ValueError("class index mismatch: " + str(class_index) + ' ' + str(selected_class))

        #print('TRAINING ' + dataset_name + ' ' + str(class_index))

        loss = m.train(dataset = dataset_train.get_dataset(),
                       dataset_name = dataset_name,
                       class_index = class_index,
                       desired_batches = cfg["nn_train_batches"])

        loss_list.append(loss)

    m.model.save()
    m.optimizer.save()

    avg_loss = sum(loss_list) / len(loss_list)

    print(' ')
    acc_list = []
    time_list = []
    for i in range(num_classes):
        selected_class = i % num_classes
        dataset_test = dataset_list[selected_class][1]
        dataset_name = dataset_list[selected_class][2]
        class_index  = dataset_list[selected_class][3]

        #print('TESTING ' + dataset_name + ' ' + str(class_index))

        if class_index != selected_class:
            raise ValueError("class index mismatch: " + str(class_index) + ' ' + str(selected_class))

        acc, tim = m.test(dataset = dataset_test.get_dataset(),
                     dataset_name = dataset_name,
                     class_index = class_index)
        acc_list.append(acc)
        time_list.append(tim)

    avg_acc = sum(acc_list) / len(acc_list)
    avg_time = sum(time_list) / len(time_list)
    print(avg_acc)

    return avg_acc, avg_loss, avg_time


class Model_dl(object):
    def __init__(self, config_id, num_classes, cfg, session):
        super().__init__()
        self.cfg = cfg
        self.train_dataloader_dict = {}
        self.session = session
        self.model = WrapperModel_dl(config_id = config_id, num_classes = num_classes, cfg = cfg)
        self.model.cuda()
        self.criterion = torch.nn.CrossEntropyLoss().cuda()
        self.optimizer = WrapperOptimizer(config_id = config_id, model=self.model, cfg=self.cfg)

    def train(self, dataset, dataset_name, class_index, desired_batches):
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.cfg['nn_train_batch_size'],
            shuffle=self.cfg['shuffle_data'],
            drop_last=False
        )

        torch.set_grad_enabled(True)
        self.model.train()

        finish_loop = False
        train_batches = 0
        loss_list = []

        while not finish_loop:
            # Set train mode before we go into the train loop over an epoch
            for data, _ in dataloader:
                #im = data[0].cpu().permute(1,2,0).numpy()
                #matplotlib.pyplot.imsave(dataset_name + '_' + str(iteration) + '.jpeg', im)
                self.optimizer.zero_grad()
                output = self.model(data.cuda())
                labels = torch.LongTensor([class_index]).cuda()
                loss = self.criterion(output, labels)
                #print('LOSS: ' + str(loss))
                loss.backward()
                self.optimizer.step()

                loss_list.append(loss.cpu().detach().numpy())

                train_batches += 1
                if train_batches > desired_batches:
                    finish_loop = True
                    break

        return sum(loss_list) / len(loss_list)


    def test(self, dataset, dataset_name, class_index):
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.cfg['nn_test_batch_size'],
            shuffle=False,
            drop_last=False
        )

        torch.set_grad_enabled(False)
        self.model.eval()

        prediction_list = []

        for data, _ in dataloader:
            prediction_list.append(self.model(data.cuda()).cpu())
        prediction = torch.cat(prediction_list, dim=0)


        acc = calc_accuracy(prediction, class_index)
        time = self.model.get_avg_time()
        print("ACCURACY: " + str(dataset_name) + ' ' + str(acc))# + ' ' + str(time.time() - t1))

        #print('TEST DL END')

        return acc, time



class BOHBWorker(Worker):
    def __init__(self, cfg, *args, **kwargs):
        super(BOHBWorker, self).__init__(*args, **kwargs)
        self.cfg = cfg
        self.dataset_list = load_datasets_processed(cfg, cfg["train_datasets"])
        self.session = tf.Session()

        print(cfg)

    def compute(self, config_id, config, budget, working_directory, *args, **kwargs):
        print("START BOHB ITERATION")
        print('CONFIG: ' + str(config))
        print('BUDGET: ' + str(budget))

        info = {}

        score = 0
        avg_loss = 0

        cfg = copy_config_to_cfg(self.cfg, config)

        #try:
        if cfg['use_model'] == 'xgb':
            score, avg_loss, avg_time = execute_run_xgb(config_id = config_id,
                                                        cfg = cfg,
                                                        budget = budget,
                                                        dataset_list = self.dataset_list,
                                                        session=self.session)
        elif cfg['use_model'] == 'dl':
            score, avg_loss, avg_time = execute_run_dl(config_id = config_id,
                                                       cfg=cfg,
                                                       budget=budget,
                                                       dataset_list=self.dataset_list,
                                                       session=self.session)
        # except Exception as e:
        #     status = str(e)
        #     print(status)

        #info[cfg["dataset"]] = score
        info['avg_loss'] = avg_loss
        info['avg_time'] = avg_time
        info['config'] = str(config)
        info['cfg'] = str(cfg)

        print('----------------------------')
        print('FINAL SCORE: ' + str(score))
        print('----------------------------')
        print("END BOHB ITERATION")

        return {
            "loss": -score,
            "info": info
        }


def runBohbSerial(cfg):
    # assign random port in the 30000-40000 range to avoid using a blocked port because of a previous improper bohb shutdown
    run_id = '0'
    port = int(30000 + random.random() * 10000)

    ns = hpns.NameServer(run_id=run_id, host="127.0.0.1", port=port)
    ns.start()

    w = BOHBWorker(cfg=cfg,
                   nameserver="127.0.0.1",
                   run_id=run_id,
                   nameserver_port=port)
    w.run(background=True)

    result_logger = hpres.json_result_logger(
        directory=cfg["bohb_log_dir"], overwrite=True
    )

    bohb = BohbWrapper(
        configspace=get_configspace(),
        run_id=run_id,
        min_budget=cfg["bohb_min_budget"],
        max_budget=cfg["bohb_max_budget"],
        nameserver="127.0.0.1",
        nameserver_port=port,
        result_logger=result_logger,
    )

    res = bohb.run(n_iterations=cfg["bohb_iterations"])
    bohb.shutdown(shutdown_workers=True)
    ns.shutdown()

    return res


def runBohbParallel(cfg, id):
    # every process has to lookup the hostname
    host = hpns.nic_name_to_host(cfg["bohb_interface"])
    # assign random port in the 30000-40000 range to avoid using a blocked port because of a previous improper bohb shutdown
    # port = int(30000 + random.random() * 10000)

    os.makedirs(cfg["bohb_log_dir"], exist_ok=True)

    if int(id) > 0:
        time.sleep(15)
        w = BOHBWorker(cfg=cfg,
                       timeout=1,
                       host=host,
                       run_id=cfg["bohb_run_id"])
        w.load_nameserver_credentials(working_directory=cfg["bohb_log_dir"])
        w.run(background=False)
        exit(0)

    ns = hpns.NameServer(run_id=cfg["bohb_run_id"],
                         host=host,
                         port=0,
                         working_directory=cfg["bohb_log_dir"])
    ns_host, ns_port = ns.start()

    # w = BOHBWorker(sleep_interval=0.5,
    #                run_id=run_id,
    #                host=host,
    #                nameserver=ns_host,
    #                nameserver_port=ns_port)
    # w.run(background=True)

    result_logger = hpres.json_result_logger(directory=cfg["bohb_log_dir"],
                                             overwrite=True)

    bohb = BohbWrapper(configspace=get_configspace(),
                       run_id=cfg["bohb_run_id"],
                       host=host,
                       nameserver=ns_host,
                       nameserver_port=ns_port,
                       min_budget=cfg["bohb_min_budget"],
                       max_budget=cfg["bohb_max_budget"],
                       result_logger=result_logger)

    res = bohb.run(n_iterations=cfg["bohb_iterations"],
                   min_n_workers=cfg["bohb_workers"])

    bohb.shutdown(shutdown_workers=True)
    ns.shutdown()

    return res


def continuous_training(cfg):
    print(cfg)

    os.makedirs(cfg['bohb_log_dir'], exist_ok=True)

    dataset_list = load_datasets_raw(cfg, cfg["train_datasets"])
    num_classes = len(dataset_list)

    m = Model_dl(config_id=(0,0,0), num_classes=num_classes, cfg=cfg)

    i = 1
    while True:
        # load training dataset
        selected_class = i % num_classes
        dataset_train = dataset_list[selected_class][0]
        dataset_name = dataset_list[selected_class][2]
        class_index = dataset_list[selected_class][3]

        if i % 100 == 0:
            print(str(i), end=' ', flush = True)

        if class_index != selected_class:
            raise ValueError("class index mismatch: " + str(class_index) + ' ' + str(selected_class))

        m.train(dataset=dataset_train.get_dataset(),
                dataset_name=dataset_name,
                class_index=class_index,
                desired_batches=cfg["train_batches"],
                iteration=i,
                save_iteration=1000)

        if i % 10000 == 0:
            print(' ', flush = True)

            acc_list = []

            for i in range(num_classes):
                selected_class = i % num_classes
                dataset_test = dataset_list[selected_class][1]
                dataset_name = dataset_list[selected_class][2]
                class_index = dataset_list[selected_class][3]

                print('TESTING ' + dataset_name + ' ' + str(class_index))

                if class_index != selected_class:
                    raise ValueError("class index mismatch: " + str(class_index) + ' ' + str(selected_class))

                acc = m.test(dataset=dataset_test.get_dataset(),
                             dataset_name=dataset_name,
                             class_index=class_index)
                acc_list.append(acc)

            avg_acc = sum(acc_list) / len(acc_list)
            print(avg_acc, flush = True)

        i += 1

    return avg_acc


def generate_samples(cfg, idx=1, idx_total=1):
    model = torchvision.models.resnet18(pretrained=True)
    model.cuda()
    model.fc = Identity()
    session = tf.Session()
    batch_size = 512

    dataset_list = load_datasets_raw(cfg, cfg["train_datasets"] + cfg["test_datasets"])
    print(dataset_list)
    des_num_samples = cfg['des_num_samples']

    os.makedirs(cfg["proc_dataset_dir"], exist_ok=True)

    for i in range(len(dataset_list)):
        dataset_train = dataset_list[i][0].get_dataset()
        dataset_test  = dataset_list[i][1].get_dataset()
        dataset_name  = dataset_list[i][2]

        print(dataset_name)

        if (i - idx) % idx_total != 0:
            print('skip')
            continue

        transform_train = load_transform(cfg, is_training=True)
        transform_test  = load_transform(cfg, is_training=False)
        ds_temp = TFDataset(session=session, dataset=dataset_test, num_samples=int(1e9))
        info = ds_temp.scan()

        ds_train = TFDataset(session=session, dataset=dataset_train, num_samples=int(1e9), transform=transform_train)
        ds_test = TFDataset(session=session, dataset=dataset_test, num_samples=info['num_samples'], transform=transform_test)
        dl_train = torch.utils.data.DataLoader(ds_train, batch_size=batch_size, shuffle=False, drop_last=False)
        dl_test = torch.utils.data.DataLoader(ds_test, batch_size=batch_size, shuffle=False, drop_last=False)

        torch.set_grad_enabled(False)
        model.eval()

        num_samples = 0
        output_list_train = []
        while num_samples < des_num_samples:
            for data, _ in dl_train:
                output = model(data.cuda())
                output_list_train.append(output.cpu().numpy())

                num_samples += batch_size
                if num_samples > des_num_samples:
                    break

                if num_samples/batch_size % 2 == 0:
                    print('Done ' + str(num_samples) + ' out of ' + str(des_num_samples) + ' samples')

        output_list_test = []
        for data, _ in dl_test:
            output = model(data.cuda())
            output_list_test.append(output.cpu().numpy())

        output_train = np.concatenate(output_list_train, axis=0)
        output_test = np.concatenate(output_list_test, axis=0)

        file_train = os.path.join(cfg["proc_dataset_dir"], dataset_name + '_train')
        file_test = os.path.join(cfg["proc_dataset_dir"], dataset_name + '_test')
        with open(file_train, "wb") as fh_train:
            pickle.dump(output_train, fh_train)
        with open(file_test, "wb") as fh_test:
            pickle.dump(output_test, fh_test)


def verify_data(cfg):
    # load data with different data loaders and verify its output

    model = torchvision.models.resnet18(pretrained=True)
    model.cuda()
    model.fc = Identity()
    session = tf.Session()
    batch_size = 1

    dataset_list_raw = load_datasets_raw(cfg, cfg["train_datasets"] + cfg["test_datasets"])
    dataset_list_pro = load_datasets_processed(cfg, cfg["train_datasets"] + cfg["test_datasets"])

    for i in range(len(dataset_list_raw)):
        dataset_test_raw  = dataset_list_raw[i][1].get_dataset()
        dataset_name_raw  = dataset_list_raw[i][2]

        dataset_test_pro  = dataset_list_pro[i][1].get_dataset()
        dataset_name_pro  = dataset_list_pro[i][2]

        print(dataset_name_pro)

        if dataset_name_pro != dataset_name_raw:
            raise ValueError("dataset mismatch: " + str(dataset_name_pro) + ' ' + str(dataset_name_raw))

        transform_test  = load_transform(cfg, is_training=False)

        ds_temp = TFDataset(session=session, dataset=dataset_test_raw, num_samples=int(1e9))
        info = ds_temp.scan()

        ds_test = TFDataset(session=session, dataset=dataset_test_raw, num_samples=info['num_samples'], transform=transform_test)
        dl_test_raw = torch.utils.data.DataLoader(ds_test, batch_size=batch_size, shuffle=False, drop_last=False)
        dl_test_pro = torch.utils.data.DataLoader(dataset_test_pro, batch_size=batch_size, shuffle=False, drop_last=False)
        torch.set_grad_enabled(False)
        model.eval()

        output_list_raw = []
        for data_raw,_ in dl_test_raw:
            output_raw = model(data_raw.cuda()).cpu()
            output_list_raw.append(output_raw)

        output_list_pro = []
        for data_pro,_ in dl_test_pro:
            output_list_pro.append(data_pro)


        for i in range(len(output_list_raw)):
            if torch.sum(output_list_pro[i]-output_list_raw[i]) / torch.sum(torch.abs(output_list_pro[i])) > 0.001:
                print('error too large')


def verify_data_histogram(cfg):
    # plot historgram of different datasets

    batch_size = 128
    dataset_list_pro = load_datasets_processed(cfg, cfg["train_datasets"] + cfg["test_datasets"])

    for i in range(len(dataset_list_pro)):
        dataset_test_pro  = dataset_list_pro[i][1].get_dataset()
        dataset_name_pro  = dataset_list_pro[i][2]

        dl_test_pro = torch.utils.data.DataLoader(dataset_test_pro, batch_size=batch_size, shuffle=False, drop_last=False)

        sum_pro = np.zeros([512])
        for data_pro,_ in dl_test_pro:
            sum_pro += np.sum(data_pro.data.numpy(), axis=0)

        plt.plot(sum_pro)
        plt.xlabel(dataset_name_pro)
        plt.savefig(dataset_name_pro + '.png')
        plt.close()



if __name__ == "__main__":
    # datasets = ['binary_alpha_digits', 'caltech101', 'caltech_birds2010', 'caltech_birds2011', # 4
    #            'cats_vs_dogs', 'cifar10', 'cifar100', 'coil100', 'colorectal_histology',       # 5
    #            'deep_weeds', 'emnist', 'eurosat', 'fashion_mnist', 'food101',                  # 5
    #            'horses_or_humans', 'kmnist', 'mnist', 'omniglot',                              # 4
    #            'oxford_flowers102', 'oxford_iiit_pet', 'patch_camelyon', 'rock_paper_scissors',# 4
    #            'smallnorb', 'stanford_dogs', 'svhn_cropped', 'tf_flowers', 'uc_merced',        # 5
    #            'Chucky', 'Decal', 'Hammer', 'Hmdb51', 'Katze', 'Kraut', 'Kreatur', 'miniciao', # 8
    #            'Monkeys', 'Munster', 'Pedro', 'SMv2', 'Ucf101']                                # 5

    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            print(arg)
        cfg = get_configuration(sys.argv[2])
        res = runBohbParallel(cfg, sys.argv[1])
    else:
        cfg = get_configuration('blubb')
        res = runBohbSerial(cfg)

    # cfg = get_configuration()
    # res = verify_data(cfg)

    # cfg = get_configuration()
    # generate_samples(cfg)

    # res = continuous_training(cfg)
