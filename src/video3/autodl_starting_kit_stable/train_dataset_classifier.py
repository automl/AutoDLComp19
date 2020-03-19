import os
import sys

sys.path.append(os.path.join(os.getcwd(), 'AutoDL_ingestion_program'))
sys.path.append(os.path.join(os.getcwd(), 'AutoDL_scoring_program'))

import matplotlib.pyplot as plt
import pickle
import xgboost as xgb
import sklearn.cluster as cluster
import sklearn.ensemble as ensemble
import sklearn.neighbors as neighbors
from sklearn.metrics import accuracy_score
import tensorflow as tf
import random
import numpy as np
import time
import torchvision
import torch
import json
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

#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.WARN)
torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
#torch.backends.cudnn.enabled = False

print(sys.path)

def get_data_type(dataset_dir):
    for elem in ['meta', 'resnet', 'combined']:
        if elem in dataset_dir:
            return elem

    raise ValueError('Unknown data type')


def get_configspace():
    cs = CS.ConfigurationSpace()
    # fc classifier
    cs.add_hyperparameter(CSH.CategoricalHyperparameter(name='nn_train_batch_size', choices = [2,4,8,16,32,64,128,256,512,1024]))
    #cs.add_hyperparameter(CSH.CategoricalHyperparameter(name='nn_test_batch_size', choices = [4,8,16,32,64,128,256,512,1024]))
    cs.add_hyperparameter(CSH.UniformFloatHyperparameter(name='nn_lr', lower=1e-5, upper=1e-2, log=True))
    cs.add_hyperparameter(CSH.CategoricalHyperparameter(name='nn_neurons', choices=[16,32,64,128,256,512,1024]))
    cs.add_hyperparameter(CSH.UniformFloatHyperparameter(name='nn_dropout', lower=0.0, upper=0.8))

    # xgb classifier
    #cs.add_hyperparameter(CSH.UniformFloatHyperparameter(name='xgb_eta', lower=0, upper=1, log=False))
    #cs.add_hyperparameter(CSH.UniformIntegerHyperparameter(name='xgb_max_depth', lower=1, upper=10))
    #cs.add_hyperparameter(CSH.UniformIntegerHyperparameter(name='xgb_train_steps', lower=1, upper=20))
    #  cs.add_hyperparameter(CSH.CategoricalHyperparameter(name='xgb_train_batch_size', choices = [2,4,8,16,32,64,128,256,512,1024]))
    #cs.add_hyperparameter(CSH.CategoricalHyperparameter(name='xgb_test_batch_size', choices = [4]))
    # xgb2 classifier
    #cs.add_hyperparameter(CSH.UniformFloatHyperparameter(name='xgb_lr', lower=0.001, upper=0.5))
    #cs.add_hyperparameter(CSH.UniformIntegerHyperparameter(name='xgb_n_estimators', lower=1, upper=20))
    #cs.add_hyperparameter(CSH.UniformFloatHyperparameter(name='xgb_subsample', lower=0.2, upper=1.0))
    #cs.add_hyperparameter(CSH.CategoricalHyperparameter(name='xgb_criterion', choices = ['friedman_mse', 'mae', 'mse']))
    #cs.add_hyperparameter(CSH.UniformIntegerHyperparameter(name='xgb_min_samples_split', lower=2, upper=10))
    #cs.add_hyperparameter(CSH.UniformIntegerHyperparameter(name='xgb_min_samples_leaf', lower=1, upper=10))
    #cs.add_hyperparameter(CSH.UniformIntegerHyperparameter(name='xgb_max_depth', lower=1, upper=10))

    # kmeans classifier
    #cs.add_hyperparameter(CSH.UniformIntegerHyperparameter(name='kmeans_n_clusters', lower=5, upper=40, ))  # default=8
    #cs.add_hyperparameter(CSH.UniformIntegerHyperparameter(name='kmeans_n_init', lower=10, upper=30, ))  #  default=10
    #cs.add_hyperparameter(CSH.UniformIntegerHyperparameter(name='kmeans_max_iter', lower=100, upper=500, ))  # default=300
    #cs.add_hyperparameter(CSH.UniformIntegerHyperparameter(name='kmeans_random_state', lower=1, upper=20000))

    # nearest centroid
    cs.add_hyperparameter(CSH.UniformIntegerHyperparameter(name='cluster_samples', lower=10, upper=15000, ))  #  default=10


    # DBSCAN Classifier
    #cs.add_hyperparameter(CSH.UniformFloatHyperparameter(name='dbscan_eps', lower=0.1, upper=5))  # , default=0.5
    #cs.add_hyperparameter(CSH.UniformIntegerHyperparameter(name='dbscan_min_samples', lower=3, upper=20))  # , default=5
    #cs.add_hyperparameter(CSH.CategoricalHyperparameter(name='dbscan_algo', choices=['auto', 'ball_tree', 'kd_tree', 'brute']))  # , default='auto')

    # Optics classifier
    #cs.add_hyperparameter(CSH.UniformIntegerHyperparameter(name='optics_min_samples', lower=3, upper=20, ))  # default=5
    #cs.add_hyperparameter(CSH.UniformFloatHyperparameter(name='optics_maxeps', lower=0.1, upper=1e16, ))  # default=0.5
    #cs.add_hyperparameter(CSH.CategoricalHyperparameter(name='optics_algo', choices=['auto', 'ball_tree', 'kd_tree', 'brute'],))  #  default='auto'


    # common parameters
    #cs.add_hyperparameter(CSH.UniformFloatHyperparameter(name='transform_scale', lower=0.3, upper=1, log=False))
    #cs.add_hyperparameter(CSH.UniformFloatHyperparameter(name='transform_ratio', lower=0.3, upper=1, log=False))
    cs.add_hyperparameter(CSH.CategoricalHyperparameter(name='use_med', choices=[False, True]))
    cs.add_hyperparameter(CSH.CategoricalHyperparameter(name='use_mean', choices=[False, True]))
    cs.add_hyperparameter(CSH.CategoricalHyperparameter(name='use_var', choices=[False, True]))
    cs.add_hyperparameter(CSH.CategoricalHyperparameter(name='use_std', choices=[False, True]))
    cs.add_hyperparameter(CSH.CategoricalHyperparameter(name='use_skew', choices=[False, True]))
    cs.add_hyperparameter(CSH.CategoricalHyperparameter(name='use_kurt', choices=[False, True]))
    cs.add_hyperparameter(CSH.UniformFloatHyperparameter(name='cut_perc', lower=0, upper=0.4, log=False))

    cs.add_hyperparameter(CSH.CategoricalHyperparameter(name='shuffle_data', choices=[True, False]))

    return cs


def get_configuration(log_subfolder=None, test_batch_size=32):
    cfg = {}
    cluster_mode = False
    if cluster_mode:
        cfg["code_dir"] = '/home/nierhoff/AutoDLComp19/src/video3/autodl_starting_kit_stable/AutoDL_sample_code_submission'
        cfg['proc_dataset_dir'] = '/data/aad/image_datasets/thomas_processed_datasets/1e4_resnet'
        cfg['des_num_samples'] = int(1e4)
        cfg["image_dir"] = '/data/aad/image_datasets/challenge'
        cfg["video_dir"] = '/data/aad/video_datasets/challenge'
        cfg["bohb_interface"] = 'eth0'
        cfg["bohb_workers"] = 10
    else:
        cfg["code_dir"] = '/home/dingsda/autodl/AutoDLComp19/src/video3/autodl_starting_kit_stable'
        cfg['proc_dataset_dir'] = '/home/dingsda/data/datasets/processed_datasets/1e4_resnet'
        cfg['des_num_samples'] = int(1e4)
        cfg["image_dir"] = '/home/dingsda/data/datasets/challenge/image'
        cfg["video_dir"] = '/home/dingsda/data/datasets/challenge/video'
        cfg["bohb_interface"] = 'lo'
        cfg["bohb_workers"] = 3

    # log_folder = "dl_logs"
    # if log_subfolder is None:
    #     cfg["bohb_log_dir"] = os.path.join(os.getcwd(), log_folder, str(int(time.time())))
    # else:
    #     cfg["bohb_log_dir"] = os.path.join(os.getcwd(), log_folder, log_subfolder)

    # for final evaluation
    cfg['final_evaluated_config_dir'] = '/home/dingsda/logs/evaluated_configs'
    cfg['bohb_log_dir'] = os.path.join('/home/dingsda/logs/dl_logs_new', log_subfolder)

    cfg["use_model"] = 'dl'    # xgb, dl, dl2 or cluster

    if cfg["use_model"] == 'dl':
        cfg["bohb_min_budget"] = 100
        cfg["bohb_max_budget"] = 1000
    elif cfg["use_model"] == 'dl2':
        cfg["bohb_min_budget"] = 100
        cfg["bohb_max_budget"] = 5000
    elif cfg["use_model"] == 'cluster':
        cfg["bohb_min_budget"] = 1000
        cfg["bohb_max_budget"] = 1000
    else:
        cfg["bohb_min_budget"] = 100
        cfg["bohb_max_budget"] = 1000

    cfg['data_type'] = get_data_type(cfg['proc_dataset_dir'])
    cfg["data_gain"] = 0.5

    cfg["bohb_iterations"] = 20
    cfg["bohb_run_id"] = '123'
    cfg['model_input_size'] = 128
    cfg['transform_scale'] = 0.7
    cfg['transform_ratio'] = 0.75
    cfg['nn_lr'] = 1e-4
    cfg['nn_train_batches'] = 1
    # cfg['nn_train_batch_size'] = 128
    # DL2

    cfg['nn_dropout'] = 0.13170661832419298
    cfg['nn_lr'] = 0.0007190982887197122
    cfg['nn_neurons'] = 1024
    cfg['nn_train_batch_size'] = 16
	cfg['nn_test_batch_size'] = test_batch_size
    cfg['gamma'] = 0.5
    cfg['epochs'] = 3

    cfg['xgb_eta'] = 0.3
    cfg['xgb_max_depth'] = 6
    cfg['xgb_train_steps'] = 10
    cfg['xgb_train_batch_size'] = 10

    cfg['cluster_test_batch_size'] = 10

    # without 'Munster', 'Kraut', 'Chucky', 'Decal', 'Hammer', 'Katze', 'Kreatur', 'Pedro', 'Hmdb51', 'Ucf101', 'SMv2', 'oxford_flowers102', 'emnist', 'caltech_birds2011',

    train_datasets = ['binary_alpha_digits', 'caltech101', 'caltech_birds2010',
                      'cats_vs_dogs', 'cifar10', 'cifar100', 'coil100',
                      'colorectal_histology', 'deep_weeds', 'eurosat',
                      'fashion_mnist', 'horses_or_humans', 'kmnist', 'mnist',
                      'oxford_iiit_pet', 'patch_camelyon', 'rock_paper_scissors',
                      'smallnorb', 'svhn_cropped', 'tf_flowers', 'uc_merced']
    test_datasets = ['Chucky', 'Decal', 'Hammer', 'Munster', 'Pedro', 'Kraut', 'Katze', 'Kreatur']
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
    def __init__(self, data_path, class_index):
        with open(data_path, 'rb') as fh:
            self.dataset = torch.tensor(pickle.load(fh)).float()
            self.class_index = torch.tensor(class_index).float()

    def get_dataset(self):
        # for compatibility
        return self

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx], self.class_index


def load_datasets_processed(cfg, datasets, dataset_dir=None):
    '''
    load preprocessed datasets from a list, return train/test datasets, dataset index and dataset name
    '''
    if dataset_dir is None:
        dataset_dir = cfg['proc_dataset_dir']
    dataset_list = []
    class_index = 0

    for dataset_name in datasets:
        dataset_train_path = os.path.join(dataset_dir, dataset_name + '_train')
        dataset_test_path = os.path.join(dataset_dir, dataset_name + '_test')

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

        mult = 0

        if self.cfg['use_med']:
            mult += 1
        if self.cfg['use_mean']:
            mult += 1
        if self.cfg['use_std']:
            mult += 1
        if self.cfg['use_var']:
            mult += 1
        if self.cfg['use_skew']:
            mult += 1
        if self.cfg['use_kurt']:
            mult += 1

        if cfg['data_type'] == 'meta':
            input_size = 18
        elif cfg['data_type'] == 'resnet':
            input_size = 512
        elif cfg['data_type'] == 'combined':
            input_size = 530

        self.fc1 = torch.nn.Linear(input_size*mult, self.cfg["nn_neurons"])
        self.fc2 = torch.nn.Linear(self.cfg["nn_neurons"], num_classes)
        self.a = Swish()
        self.d = torch.nn.Dropout(cfg["nn_dropout"])

        if os.path.isfile(self.filename):
            print('load model')
            self.load_state_dict(torch.load(self.filename))

    def forward(self, x):
        nb_samples = x.shape[0]
        nb_cut = int(self.cfg['cut_perc'] * nb_samples)
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
        if self.cfg['use_med']:
            x = med if x is None else torch.cat((x, med), 0)
        if self.cfg['use_mean']:
            x = mean if x is None else torch.cat((x, mean), 0)
        if self.cfg['use_std']:
            x = std if x is None else torch.cat((x, std), 0)
        if self.cfg['use_var']:
            x = var if x is None else torch.cat((x, var), 0)
        if self.cfg['use_skew']:
            x = skew if x is None else torch.cat((x, skew), 0)
        if self.cfg['use_kurt']:
            x = kurt if x is None else torch.cat((x, kurt), 0)

        x = self.fc2(self.d(self.a(self.fc1(x))))
        x = x.unsqueeze(0)

        return x

    def save(self):
        torch.save(self.state_dict(), self.filename)

class Swish(torch.nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class WrapperModel_dl2(torch.nn.Module):
    def __init__(self, config_id, num_classes, cfg):
        super().__init__()
        self.cfg = cfg
        self.filename = self.cfg["code_dir"] + '/test_bs_' + str(self.cfg["nn_test_batch_size"]) + '_model.pt'
        self.fc1 = torch.nn.Linear(512, self.cfg["nn_neurons"])
        self.fc2 = torch.nn.Linear(self.cfg["nn_neurons"], num_classes)
        self.a = Swish()
        self.d = torch.nn.Dropout(cfg["nn_dropout"])

        if os.path.isfile(self.filename):
            self.load_state_dict(torch.load(self.filename))

    def forward(self, x):
        x = self.fc2(self.d(self.a(self.fc1(x))))
        return x

    def save(self):
        torch.save(self.state_dict(), self.filename)

class WrapperModel_xgb(torch.nn.Module):
    def __init__(self, config_id, cfg):
        super().__init__()
        self.cfg = cfg
        self.filename = self.cfg["bohb_log_dir"] + '/' + str(config_id[0]) + '_' + str(config_id[1]) + '_' + str(config_id[2]) + '_model' + '.pt'

        mult = 0

        if self.cfg['use_med']:
            mult += 1
        if self.cfg['use_mean']:
            mult += 1
        if self.cfg['use_std']:
            mult += 1
        if self.cfg['use_var']:
            mult += 1
        if self.cfg['use_skew']:
            mult += 1
        if self.cfg['use_kurt']:
            mult += 1

        if os.path.isfile(self.filename):
            self.load_state_dict(torch.load(self.filename))

    def forward(self, x):
        nb_samples = x.shape[0]
        nb_cut = int(self.cfg['cut_perc'] * nb_samples)
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
        if self.cfg['use_med']:
            x = med if x is None else torch.cat((x, med), 0)
        if self.cfg['use_mean']:
            x = mean if x is None else torch.cat((x, mean), 0)
        if self.cfg['use_std']:
            x = std if x is None else torch.cat((x, std), 0)
        if self.cfg['use_var']:
            x = var if x is None else torch.cat((x, var), 0)
        if self.cfg['use_skew']:
            x = skew if x is None else torch.cat((x, skew), 0)
        if self.cfg['use_kurt']:
            x = kurt if x is None else torch.cat((x, kurt), 0)

        x = x.unsqueeze(0)

        return x

    def save(self):
        torch.save(self.state_dict(), self.filename)

class WrapperModel_cluster(torch.nn.Module):
    def __init__(self, config_id, cfg, algorithm='kmeans'):
        super().__init__()
        self.cfg = cfg
        self.filename = self.cfg["bohb_log_dir"] + '/' + str(config_id[0]) + '_' + str(config_id[1]) + '_' + str(config_id[2]) + '_model' + '.pt'

        if os.path.isfile(self.filename):
            self.load_state_dict(torch.load(self.filename))

    def forward(self, x):
        return x

    def save(self):
        torch.save(self.state_dict(), self.filename)

class WrapperModel_xgb2(torch.nn.Module):
    def __init__(self, config_id, cfg):
        super().__init__()
        self.cfg = cfg
        self.filename = self.cfg["bohb_log_dir"] + '/' + str(config_id[0]) + '_' + str(config_id[1]) + '_' + str(config_id[2]) + '_model' + '.pt'

        if os.path.isfile(self.filename):
            self.load_state_dict(torch.load(self.filename))

    def forward(self, x):
        # x = x.unsqueeze(0)

        return x

    def save(self):
        torch.save(self.state_dict(), self.filename)

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

def execute_run_cluster(config_id, cfg, budget, dataset_list, session):
    num_classes = len(dataset_list)
    m = Model_cluster(config_id = config_id,
                 num_classes = num_classes,
                 cfg = cfg,
                 session = session,)
                 #algorithm='kmeans')

    for i in range(num_classes):
        selected_class = i % num_classes
        dataset_train = dataset_list[selected_class][0]
        dataset_name  = dataset_list[selected_class][2]
        class_index   = dataset_list[selected_class][3]

        if class_index != selected_class:
            raise ValueError("class index mismatch: " + str(class_index) + ' ' + str(selected_class))
        m.collect_samples_train(dataset = dataset_train.get_dataset(),
                                class_index = class_index,
                                desired_batches = cfg['cluster_samples'])
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

    return avg_acc, 0

class Model_cluster(object):
    def __init__(self, config_id, num_classes, cfg, session, algorithm='nc'):
        super().__init__()
        self.cfg = cfg
        self.train_dataloader_dict = {}
        self.session = session
        self.algorithm = algorithm
        if algorithm == 'kmeans':
            self.model = cluster.KMeans(n_clusters=cfg['kmeans_n_clusters'],
                                        init='k-means++',
                                        n_init=cfg['kmeans_n_init'],
                                        max_iter=cfg['kmeans_max_iter'],
                                        tol=0.0001, precompute_distances='auto',
                                        verbose=0,
                                        random_state=cfg['kmeans_random_state'],
                                        copy_x=True, n_jobs=-1,
                                        algorithm='auto')
        elif algorithm == 'nc':
            self.model = neighbors.NearestCentroid(metric='manhattan')
        elif algorithm == 'dbscan':
            self.model = cluster.DBSCAN(eps=cfg['dbscan_eps'],
                                        min_samples=cfg['dbscan_min_samples'],
                                        metric='euclidean',
                                        metric_params=None,
                                        algorithm=cfg['dbscan_algo'],
                                        leaf_size=30,
                                        p=None,
                                        n_jobs=-1)
        elif algorithm == 'optics':
            self.model = cluster.OPTICS(min_samples=cfg['optics_min_samples'],
                                        max_eps=cfg['optics_maxeps'],
                                        metric='minkowski', p=2,
                                        metric_params=None,
                                        cluster_method='xi',
                                        eps=None,
                                        xi=0.05,
                                        predecessor_correction=True,
                                        min_cluster_size=None,
                                        algorithm=cfg['optics_algo'],
                                        leaf_size=30,
                                        n_jobs=-1)

        self.X = []
        self.y = []
        self.num_classes = num_classes

    def collect_samples_train(self, dataset, class_index, desired_batches):
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            drop_last=False
        )
        finish_loop = False
        train_batches = 0
        while not finish_loop:
    		# Set train mode before we go into the train loop over an epoch
            for data, _ in dataloader:
                self.X.append(np.squeeze(data.cpu().numpy()))
                self.y.append(np.array([class_index]))
                train_batches += 1
                if train_batches > desired_batches:
                    finish_loop = True
                    break

    def train(self):
        X = np.asarray(self.X)
        y = np.asarray(self.y)

        param = {
            'eta': self.cfg['xgb_eta'],
            'max_depth': self.cfg['xgb_max_depth'],
            'objective': 'multi:softprob',
            'num_class': self.num_classes}
        #print('x shape: ', X.shape)
        #print('y shape: ', y.shape)
        self.model.fit(X, y)
        # D_train = xgb.DMatrix(X, y)
        # self.xgb = xgb.train(param, D_train, self.cfg['xgb_train_steps'])
        self.X = None
        self.y = None

    def test(self, dataset, dataset_name, class_index):
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=cfg['cluster_test_batch_size'],
            shuffle=False,
            drop_last=False
        )
        if False:
            X = []
            y = []
            for data, _ in dataloader:
                X.append(np.squeeze(data.cpu().numpy()))
                y.append(np.array([class_index]))
            prediction = self.model.predict(X)
            acc = accuracy_score(prediction, y)
            print("ACCURACY: " + str(dataset_name) + ' ' + str(acc))  # + ' ' + str(time.time() - t1))

        else:
            prediction_list = []
            for data, _ in dataloader:
                # prediction_list.append(self.model(data.cuda()).cpu())#
                if data.shape[0] == 1:
                    data = torch.cat((data, data), 0)
                X = np.squeeze(data.cpu().numpy())
                predictions = self.model.predict(X)
                labels, counts = np.unique(predictions, return_counts=True)
                prediction_list.append(labels[np.argmax(counts)])
            # prediction = torch.cat(prediction_list)
            prediction = np.array(prediction_list)
            # print(prediction)
            # print(prediction.shape, counterr)
            #+print(prediction)
            # print(sum(prediction == class_index) / len(prediction))
            acc = sum(prediction == class_index) / len(prediction)

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

    return avg_acc, 0

def execute_run_xgb2(config_id, cfg, budget, dataset_list, session):
    num_classes = len(dataset_list)

    m = Model_xgb2(config_id = config_id,
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

    return avg_acc, 0

class Model_xgb(object):
    def __init__(self, config_id, num_classes, cfg, session):
        super().__init__()
        self.cfg = cfg
        self.train_dataloader_dict = {}
        self.session = session
        self.model = WrapperModel_xgb(config_id, cfg)
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
                data_preproc = preprocess_meta_data(data.cuda(), self.cfg)
                output = self.model(data_preproc).cpu()
                self.X.append(output.numpy())
                self.y.append(np.array([class_index]))

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


class Model_xgb2(object):
    def __init__(self, config_id, num_classes, cfg, session, algorithm='nc'):
        super().__init__()
        self.cfg = cfg
        self.train_dataloader_dict = {}
        self.session = session
        self.algorithm = algorithm
        self.model = ensemble.GradientBoostingClassifier(loss='deviance', 
                        learning_rate=cfg['xgb_lr'], n_estimators=cfg['xgb_n_estimators'], subsample=cfg['xgb_subsample'], 
                        criterion=cfg['xgb_criterion'], min_samples_split=cfg['xgb_min_samples_split'],
                        min_samples_leaf=cfg['xgb_min_samples_leaf'], min_weight_fraction_leaf=0.0,
                        max_depth=cfg['xgb_max_depth'], min_impurity_decrease=0.0, min_impurity_split=None,
                        init=None, random_state=42, max_features=None,
                        verbose=0, max_leaf_nodes=None, warm_start=False,
                        validation_fraction=0.1, n_iter_no_change=None, tol=0.0001)

        self.X = []
        self.y = []
        self.num_classes = num_classes

    def collect_samples_train(self, dataset, class_index, desired_batches):
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            drop_last=False
        )
        finish_loop = False
        train_batches = 0
        while not finish_loop:
            # Set train mode before we go into the train loop over an epoch
            for data, _ in dataloader:
                self.X.append(np.squeeze(data.cpu().numpy()))
                self.y.append(np.array([class_index]))
                train_batches += 1
                if train_batches > desired_batches:
                    finish_loop = True
                    break

    def train(self):
        X = np.asarray(self.X)
        y = np.asarray(self.y)
        self.model.fit(X, y)
        self.X = None
        self.y = None

    def test(self, dataset, dataset_name, class_index):
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=cfg['xgb_test_batch_size'],
            shuffle=False,
            drop_last=False
        )
        if True:
            X = []
            y = []
            for data, _ in dataloader:
                X.append(np.squeeze(data.cpu().numpy()))
                y.append(np.array([class_index]))
            prediction = self.model.predict(X)
            acc = accuracy_score(prediction, y)
            print("ACCURACY: " + str(dataset_name) + ' ' + str(acc))  # + ' ' + str(time.time() - t1))

        else:
            prediction_list = []
            for data, _ in dataloader:
                # prediction_list.append(self.model(data.cuda()).cpu())#
                if data.shape[0] == 1:
                    data = torch.cat((data, data), 0)
                X = np.squeeze(data.cpu().numpy())
                predictions = self.model.predict(X)
                labels, counts = np.unique(predictions, return_counts=True)
                prediction_list.append(labels[np.argmax(counts)])
            # prediction = torch.cat(prediction_list)
            prediction = np.array(prediction_list)
            # print(prediction)
            # print(prediction.shape, counterr)
            #+print(prediction)
            # print(sum(prediction == class_index) / len(prediction))
            acc = sum(prediction == class_index) / len(prediction)

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
    for i in range(num_classes):
        selected_class = i % num_classes
        dataset_test = dataset_list[selected_class][1]
        dataset_name = dataset_list[selected_class][2]
        class_index  = dataset_list[selected_class][3]

        #print('TESTING ' + dataset_name + ' ' + str(class_index))

        if class_index != selected_class:
            raise ValueError("class index mismatch: " + str(class_index) + ' ' + str(selected_class))

        acc = m.test(dataset = dataset_test.get_dataset(),
                     dataset_name = dataset_name,
                     class_index = class_index)
        acc_list.append(acc)

    avg_acc = sum(acc_list) / len(acc_list)
    print(avg_acc)

    return avg_acc, avg_loss

def execute_run_dl2(config_id, cfg, budget, dataset_list, session):
    num_classes = len(dataset_list)

    m = Model_dl2(config_id = config_id,
                 num_classes = num_classes,
                 cfg = cfg,
                 session = session)

    loss_list = []

    for i in range(num_classes):
        selected_class = i % num_classes
        dataset_train = dataset_list[selected_class][0]
        dataset_name  = dataset_list[selected_class][2]
        class_index   = dataset_list[selected_class][3]

        if class_index != selected_class:
            raise ValueError("class index mismatch: " + str(class_index) + ' ' + str(selected_class))
        m.collect_samples_train(dataset = dataset_train.get_dataset(),
                                class_index = class_index,
                                desired_batches = budget)
    for e in range(cfg['epochs']):
        loss = m.train(budget * num_classes)
        m.scheduler.step()
        loss_list.append(loss)

    #m.model.save()
    #m.optimizer.save()

    print(' ')
    acc_list = []
    for i in range(num_classes):
        selected_class = i % num_classes
        dataset_test = dataset_list[selected_class][1]
        dataset_name = dataset_list[selected_class][2]
        class_index  = dataset_list[selected_class][3]

        #print('TESTING ' + dataset_name + ' ' + str(class_index))

        if class_index != selected_class:
            raise ValueError("class index mismatch: " + str(class_index) + ' ' + str(selected_class))

        acc = m.test(dataset = dataset_test.get_dataset(),
                     dataset_name = dataset_name,
                     class_index = class_index)
        acc_list.append(acc)

    avg_acc = sum(acc_list) / len(acc_list)
    loss = sum(loss_list) / len(loss_list)
    # print(avg_acc)

    return avg_acc, loss

def preprocess_meta_data(data, cfg):
    if data.shape[1] == 512:
        return data
    else:
        a = 1-cfg["data_gain"]
        b = 1+cfg["data_gain"]
        for i in [4]:
            data[:,i] = data[1,i] * (a + (b-a)*np.random.random())
        for i in [5,6,7,8,9,10,11,12,13,14,15,16]:
            data[:,i] = 0

    return data


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
                self.optimizer.zero_grad()
                data_preproc = preprocess_meta_data(data.cuda(), self.cfg)
                output = self.model(data_preproc)
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
        print("ACCURACY: " + str(dataset_name) + ' ' + str(acc))

        #print('TEST DL END')

        return acc

class Model_dl2(object):
    def __init__(self, config_id, num_classes, cfg, session):
        super().__init__()
        self.cfg = cfg
        self.train_dataloader_dict = {}
        self.session = session
        self.model = WrapperModel_dl2(config_id = config_id, num_classes = num_classes, cfg = cfg)
        self.model.cuda()
        self.criterion = torch.nn.CrossEntropyLoss().cuda()
        self.optimizer = torch.optim.Adam(self.model.parameters(), cfg['nn_lr'])
        self.X = []
        self.y = []
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 
                                            step_size=1, 
                                            gamma=cfg['gamma'],
                                            last_epoch=-1)

    def collect_samples_train(self, dataset, class_index, desired_batches):
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            shuffle=True,
            drop_last=False
        )
        finish_loop = False
        train_batches = 0

        while not finish_loop:
            # Set train mode before we go into the train loop over an epoch
            for data, _ in dataloader:
                self.X.append(np.squeeze(data.cpu().numpy()))
                self.y.append(class_index)
                train_batches += 1
                # print(train_batches)
                if train_batches > desired_batches:
                    finish_loop = True
                    break


    def train(self, budget):
        train_dataset =  torch.utils.data.TensorDataset(
                                torch.Tensor(self.X),
                                torch.Tensor(self.y).long())

        dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.cfg['nn_train_batch_size'],
            shuffle=True,
            drop_last=False
        )

        torch.set_grad_enabled(True)
        self.model.train()

        finish_loop = False
        train_batches = 0
        loss_list = []

        # Set train mode before we go into the train loop over an epoch
        for data, labels in dataloader:
            #im = data[0].cpu().permute(1,2,0).numpy()
            #matplotlib.pyplot.imsave(dataset_name + '_' + str(iteration) + '.jpeg', im)
            self.optimizer.zero_grad()
            #data_preproc = preprocess_meta_data(data.cuda(), self.cfg)
            output = self.model(data.cuda())
            labels = labels.cuda()
            loss = self.criterion(output, labels)
            #print('LOSS: ' + str(loss))
            loss.backward()
            self.optimizer.step()
            loss_list.append(loss.cpu().detach().numpy())
            train_batches += 1

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
            # prediction_list.append(self.model(data.cuda()).cpu())#
            _ , label = torch.max(self.model(data.cuda()).cpu(), 1)
            labels, counts = torch.unique(label, sorted=True, return_inverse=False, return_counts=True, dim=None)
            # print(labels[torch.argmax(counts)])
            prediction_list.append(labels[np.argmax(counts)])
        # prediction = torch.cat(prediction_list)
        prediction = np.array(prediction_list)
        # print(prediction.shape, counterr)
        #+print(prediction)
        # print(sum(prediction == class_index) / len(prediction))
        acc = sum(prediction == class_index) / len(prediction)
        # acc = np.sum(prediction_list[prediction_list == class_index])
        #acc = calc_accuracy(prediction, class_index)
        #print(acc1)
        #print(acc)
        # print("ACCURACY: " + str(dataset_name) + ' ' + str(acc))

        #print('TEST DL END')

        return acc

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
            score, avg_loss = execute_run_xgb(config_id = config_id,
                                              cfg = cfg,
                                              budget = budget,
                                              dataset_list = self.dataset_list,
                                              session=self.session)
        elif cfg['use_model'] == 'xgb2':
            score, avg_loss = execute_run_xgb2(config_id = config_id,
                                              cfg = cfg,
                                              budget = budget,
                                              dataset_list = self.dataset_list,
                                              session=self.session)
        elif cfg['use_model'] == 'dl':
            score, avg_loss = execute_run_dl(config_id = config_id,
                                             cfg=cfg,
                                             budget=budget,
                                             dataset_list=self.dataset_list,
                                             session=self.session)
        elif cfg['use_model'] == 'dl2':
            score, avg_loss = execute_run_dl2(config_id = config_id,
                                             cfg=cfg,
                                             budget=budget,
                                             dataset_list=self.dataset_list,
                                             session=self.session)
        elif cfg['use_model'] == 'cluster':
            score, avg_loss = execute_run_cluster(config_id = config_id,
                                             cfg=cfg,
                                             budget=budget,
                                             dataset_list=self.dataset_list,
                                             session=self.session)
        # except Exception as e:
        #     status = str(e)
        #     print(status)

        #info[cfg["dataset"]] = score
        info['avg_loss'] = avg_loss
        info['config'] = str(config)
        info['cfg'] = str(cfg)

        print('----------------------------')
        print('FINAL SCORE: ' + str(score))
        print('----------------------------')
        print("END BOHB ITERATION", flush=True)

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


def continuous_training2(cfg):
    print(cfg)

    os.makedirs(cfg['bohb_log_dir'], exist_ok=True)
    dataset_list = load_datasets_processed(cfg, cfg["train_datasets"])
    num_classes = len(dataset_list)
    session = tf.Session()
    budget = cfg['budget']
    epochs = cfg['epochs']
    m = Model_dl2(config_id=(0,0,0),
                 num_classes=num_classes,
                 cfg=cfg,
                 session=session)

    loss_list = []


    for i in range(num_classes):
        selected_class = i % num_classes
        dataset_train = dataset_list[selected_class][0]
        dataset_name  = dataset_list[selected_class][2]
        class_index   = dataset_list[selected_class][3]

        if class_index != selected_class:
            raise ValueError("class index mismatch: " + str(class_index) + ' ' + str(selected_class))
        m.collect_samples_train(dataset = dataset_train.get_dataset(),
                                class_index = class_index,
                                desired_batches = budget)
    best_acc = 0
    for i in range(epochs):
        loss = m.train(budget*num_classes)
        m.scheduler.step()

        print(' ')
        acc_list = []
        for i in range(num_classes):
            selected_class = i % num_classes
            dataset_test = dataset_list[selected_class][1]
            dataset_name = dataset_list[selected_class][2]
            class_index  = dataset_list[selected_class][3]

            #print('TESTING ' + dataset_name + ' ' + str(class_index))

            if class_index != selected_class:
                raise ValueError("class index mismatch: " + str(class_index) + ' ' + str(selected_class))

            acc = m.test(dataset = dataset_test.get_dataset(),
                         dataset_name = dataset_name,
                         class_index = class_index)
            acc_list.append(acc)
        temp_acc = sum(acc_list) / len(acc_list)
        if temp_acc >= cfg['best_acc']:
            cfg['best_acc'] = temp_acc
            m.model.save()
            print('saved model with best accuracy: {}'.format( cfg['best_acc']))
        print(temp_acc, flush = True)


    avg_acc = sum(acc_list) / len(acc_list)
    print(avg_acc, flush = True)
    return avg_acc


def generate_samples_resnet(cfg, idx=1, idx_total=1):
    model = torchvision.models.resnet18(pretrained=True)
    model.cuda()
    model.fc = Identity()
    session = tf.Session()
    batch_size = 512

    dataset_list = load_datasets_raw(cfg, cfg["train_datasets"])
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


def generate_meta_output_vector(data, general_vector):
    length = data.shape[-4]
    width = data.shape[-3]
    height = data.shape[-2]
    channels = data.shape[1]

    output_vector = np.array([[length, width, height, channels]])
    output_vector = np.concatenate([output_vector, general_vector], axis=1)

    return output_vector


def generate_samples_meta(cfg, idx=1, idx_total=1):
    session = tf.Session()
    batch_size = 1

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

        ds_train = TFDataset(session=session, dataset=dataset_train, num_samples=int(1e9))
        ds_test = TFDataset(session=session, dataset=dataset_test, num_samples=int(1e9))
        info_train = ds_train.scan()
        info_test = ds_test.scan()
        print(info_train)
        print(info_test)

        dl_train = torch.utils.data.DataLoader(ds_train, batch_size=batch_size, shuffle=False, drop_last=False)
        dl_test = torch.utils.data.DataLoader(ds_test, batch_size=batch_size, shuffle=False, drop_last=False)

        torch.set_grad_enabled(False)

        num_samples = 0
        output_list_train = []
        for data, labels in dl_train:
            n_classes = labels.shape[1]
            channels = data.shape[1]


        # general_vector = np.array([info["num_samples"]])
        general_vector_train = np.array([n_classes])
        general_vector_train = np.concatenate([general_vector_train, [channels]])
        general_vector_train = np.concatenate([general_vector_train, info_train["min_shape"]])
        general_vector_train = np.concatenate([general_vector_train, info_train["max_shape"]])
        general_vector_train = np.concatenate([general_vector_train, info_train["avg_shape"]])
        general_vector_train = np.concatenate([general_vector_train, np.array([1 if info_train["is_multilabel"] else 0])])
        general_vector_train = np.array([general_vector_train])
        #print(general_vector_train)
        # general_vector = np.array([info["num_samples"]])
        general_vector_test = np.array([n_classes])
        general_vector_test = np.concatenate([general_vector_test, [channels]])
        general_vector_test = np.concatenate([general_vector_test, info_test["min_shape"]])
        general_vector_test = np.concatenate([general_vector_test, info_test["max_shape"]])
        general_vector_test = np.concatenate([general_vector_test, info_test["avg_shape"]])
        general_vector_test = np.concatenate([general_vector_test, np.array([1 if info_test["is_multilabel"] else 0])])
        general_vector_test = np.array([general_vector_test])
        #print(general_vector_test)

        #while num_samples < des_num_samples:
        #    for data, _ in dl_train:
        #        output_vector = generate_meta_output_vector(data, general_vector)
        #        output_list_train.append(output_vector)

        #        num_samples += batch_size
        #        if num_samples >= des_num_samples:
        #            break

        #output_list_test = []
        #for data, _ in dl_test:
        #    output_vector = generate_meta_output_vector(data, general_vector)
        #    output_list_test.append(output_vector)

        #output_train = np.concatenate(output_list_train, axis=0)
        #output_test = np.concatenate(output_list_test, axis=0)

        print(general_vector_train.shape)
        print(general_vector_test.shape)

        file_train = os.path.join(cfg["proc_dataset_dir"], dataset_name + '_train')
        file_test = os.path.join(cfg["proc_dataset_dir"], dataset_name + '_test')
        with open(file_train, "wb") as fh_train:
            pickle.dump(general_vector_train, fh_train)
        with open(file_test, "wb") as fh_test:
            pickle.dump(general_vector_test, fh_test)

def generate_samples_combined(cfg):
    dataset_resnet_path = '/home/dingsda/data/datasets/processed_datasets/1e4_resnet'
    dataset_meta_path = '/home/dingsda/data/datasets/processed_datasets/1e4_meta'
    dataset_combined_path = '/home/dingsda/data/datasets/processed_datasets/1e4_combined'

    dataset_list = cfg["train_datasets"] + cfg["test_datasets"]
    print(dataset_list)

    os.makedirs(dataset_combined_path, exist_ok=True)

    for suffix in ['train', 'test']:
        for dataset_name in dataset_list:
            dataset_resnet_file = os.path.join(dataset_resnet_path, dataset_name + '_' + suffix)
            dataset_meta_file = os.path.join(dataset_meta_path, dataset_name + '_' + suffix)

            with open(dataset_resnet_file, 'rb') as fh:
                dataset_resnet = np.array(pickle.load(fh))
            with open(dataset_meta_file, 'rb') as fh:
                dataset_meta = np.array(pickle.load(fh))

            min_len = min(len(dataset_resnet), len(dataset_meta))
            dataset_combined = np.concatenate([dataset_meta[:min_len], dataset_resnet[:min_len]], axis=1)
            dataset_combined = np.float32(dataset_combined)

            file_combined = os.path.join(dataset_combined_path, dataset_name + '_' + suffix)
            with open(file_combined, "wb") as fh:
                pickle.dump(dataset_combined, fh)


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


def load_eval_configs(path):
    with open(path) as config_file:
        eval_config_list = json.load(config_file)
    return eval_config_list


def find_eval_config_name(eval_config_list, name):
    for elem in eval_config_list:
        if elem[0] == name:
            return elem


def plot_dataset_scores(dataset_name, test_batch_sizes, kakaobrain_scores, best_scores, same_scores, simi_scores, wrst_scores):
    plt.figure(figsize=(5,3))
    #sns.boxplot(x=data, y=name_list, order=name_list_sorted)
    plt.plot(kakaobrain_scores, lw=2, label='kakaob. ALC.')
    plt.plot(best_scores, lw=2, label='best ALC.')
    plt.plot(simi_scores, lw=2, label='DL ALC.')
    plt.plot(wrst_scores, lw=2, label='worst ALC.')
    plt.legend(loc='right')
    #ax.set_xticklabels(x_data)
    plt.xlabel('batch size')
    plt.ylabel('ALC')
    plt.title(dataset_name)
    plt.xticks(np.arange(9), test_batch_sizes)
    plt.show()


def evaluate_dl_on_datasets():
    cfg = get_configuration()
    session = tf.Session()
    train_datasets = cfg["train_datasets"]
    dataset_list_raw = load_datasets_raw(cfg, cfg["test_datasets"])

    # two-layer MLP
    best_config_id_dict = {4: (14, 0, 5),
                           8: (7, 0, 2),
                           16: (6, 0, 0),
                           32: (18, 0, 7),
                           64: (13, 0, 5),
                           128: (12, 0, 1),
                           256: (16, 0, 7),
                           512: (4, 0, 0),
                           1024: (9, 0, 8)}

    # single-layer MLP
    # best_config_id_dict = {4: (1, 0, 6),
    #                        8: (1, 0, 5),
    #                        16: (1, 0, 6),
    #                        32: (6, 0, 2),
    #                        64: (1, 0, 6),
    #                        128: (0, 0, 6),
    #                        256: (4, 0, 6),
    #                        512: (2, 0, 4),
    #                        1024: (7, 0, 2)}


    for i in range(len(dataset_list_raw)):
        dataset_test = dataset_list_raw[i][1].get_dataset()
        dataset_name = dataset_list_raw[i][2]

        transform_test = load_transform(cfg, is_training=False)
        ds_temp = TFDataset(session=session, dataset=dataset_test, num_samples=int(1e9))
        info = ds_temp.scan()
        ds_test = TFDataset(session=session, dataset=dataset_test, num_samples=info['num_samples'],
                            transform=transform_test)

        test_batch_sizes = [4, 8, 16, 32, 64, 128, 256, 512, 1024]

        best_scores = []
        same_scores = []
        simi_scores = []
        wrst_scores = []

        if dataset_name == 'Chucky':
            kakaobrain_scores = [0.8082]*len(test_batch_sizes)
        elif dataset_name == 'Decal':
            kakaobrain_scores = [0.8647]*len(test_batch_sizes)
        elif dataset_name == 'Hammer':
            kakaobrain_scores = [0.8147]*len(test_batch_sizes)
        elif dataset_name == 'Munster':
            kakaobrain_scores = [0.9421]*len(test_batch_sizes)
        elif dataset_name == 'Pedro':
            kakaobrain_scores = [0.7948]*len(test_batch_sizes)
        elif dataset_name == 'Kreatur':
            kakaobrain_scores = [0.8677]*len(test_batch_sizes)
        elif dataset_name == 'Katze':
            kakaobrain_scores = [0.8613]*len(test_batch_sizes)
        elif dataset_name == 'Kraut':
            kakaobrain_scores = [0.6678]*len(test_batch_sizes)

        for test_batch_size in test_batch_sizes:
            cfg = get_configuration(log_subfolder=str(test_batch_size))

            best_config_id = best_config_id_dict[test_batch_size]
            final_log_dir = cfg['bohb_log_dir']
            result = hpres.logged_results_to_HBS_result(final_log_dir)
            id2conf = result.get_id2config_mapping()
            bohb_conf = id2conf[best_config_id]['config']

            ds_test.reset()
            dl_test = torch.utils.data.DataLoader(ds_test,
                                                  batch_size=test_batch_size,
                                                  shuffle=False,
                                                  drop_last=False)

            resnet = torchvision.models.resnet18(pretrained=True)
            resnet.fc = Identity()
            class_conf = {'bohb_log_dir': final_log_dir}
            class_conf.update(cfg)
            class_conf.update(bohb_conf)
            model = WrapperModel_dl(config_id=best_config_id, num_classes=len(train_datasets), cfg=class_conf)
            torch.set_grad_enabled(False)
            resnet.cuda()
            model.cuda()
            resnet.eval()
            model.eval()

            input_data = next(iter(dl_test))[0].cuda()
            output = model(resnet(input_data))

            similar_dataset = train_datasets[np.argmax(output.cpu().data)]

            print('---------------')
            print('Dataset: ' + dataset_name)
            print('Similar: ' + similar_dataset)

            eval_config_path = os.path.join(cfg['final_evaluated_config_dir'], dataset_name + '.json')
            eval_config_list = load_eval_configs(eval_config_path)
            eval_config_list = sorted(eval_config_list, key = lambda x: -x[1])

            best_config = eval_config_list[0]
            simi_config = find_eval_config_name(eval_config_list, similar_dataset)
            same_config = find_eval_config_name(eval_config_list, dataset_name)
            wrst_config = eval_config_list[-1]

            best_scores.append(best_config[1])
            same_scores.append(same_config[1])
            simi_scores.append(simi_config[1])
            wrst_scores.append(wrst_config[1])

            print('---------------')
            print('Dataset: ' + dataset_name)
            print('Similar: ' + similar_dataset)
            print('batch size: ' + str(test_batch_size))
            print('best score: ' + str(best_config[1]))
            print('same score: ' + str(same_config[1]))
            print('simi score: ' + str(simi_config[1]))
            print('wrst score: ' + str(wrst_config[1]))

        plot_dataset_scores(dataset_name = dataset_name,
                            test_batch_sizes = test_batch_sizes,
                            kakaobrain_scores = kakaobrain_scores,
                            best_scores = best_scores,
                            same_scores = same_scores,
                            simi_scores = simi_scores,
                            wrst_scores = wrst_scores)


if __name__ == "__main__":
    # Run bohb paralell
    if False:
        if len(sys.argv) > 1:
            for arg in sys.argv[1:]:
                print(arg)
            cfg = get_configuration()
            res = runBohbParallel(cfg, sys.argv[1])
        else:
        # Run bohb sequentiel
            for test_batch_size in [4,8,16,32,64,128,256,512,1024]:
                cfg = get_configuration(str(test_batch_size), test_batch_size)
                res = runBohbSerial(cfg)

    elif True:
        res = evaluate_dl_on_datasets()

    elif False:
        cfg = get_configuration()
        cfg["use_model"] = 'cluster'
        os.makedirs(cfg['bohb_log_dir'], exist_ok=True)
        dataset_list = load_datasets_processed(cfg, cfg["train_datasets"])
        session = tf.Session()
        budget = 0
        for b in range(1646 - 10, 1646 + 10):
            cfg['cluster_test_batch_size'] =4
            cfg['cluster_samples'] = b
            print(b)
            res = execute_run_cluster((0,0,0), cfg, budget, dataset_list, session)
            print(res)
    elif False:
        best_configs = [{'nn_dropout': 0.3451839700839631, 'nn_lr': 0.00021643294430478865, 'nn_neurons': 512, 'nn_test_batch_size': 4, 'nn_train_batch_size': 2}]
        cfg = get_configuration()
        cfg["use_model"] = 'dl2'
        cfg['best_acc'] = 0
        for conf in best_configs:
            for k, v in conf.items():
                cfg[k] = v
            for b in np.arange(6000, 8000, 100):  # [5000, 10000, 15000, 20000]:
                cfg['budget'] = int(b)
                #cfg['gamma'] = 0.5
                cfg['epochs'] = 8
                print(conf, b)
                res = continuous_training2(cfg)
    elif False::
        pass
        #cfg = get_configuration()
        # res = verify_data(cfg)
        # cfg = get_configuration()
        # convert datasets to resnet features
        #generate_samples_meta(cfg)
        # convert datasets to resnet features
        #generate_samples_resnet(cfg)



