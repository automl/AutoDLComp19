import os
import sys

sys.path.append(os.path.join(os.getcwd(), 'AutoDL_ingestion_program'))
sys.path.append(os.path.join(os.getcwd(), 'AutoDL_scoring_program'))

import pickle
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
    #cs.add_hyperparameter(CSH.CategoricalHyperparameter(name='test_batch_size', choices = [128,256,512]))
    cs.add_hyperparameter(CSH.CategoricalHyperparameter(name='train_batch_size', choices = [32,64,128]))
    cs.add_hyperparameter(CSH.CategoricalHyperparameter(name='train_batches', choices=[1,2,4]))
    cs.add_hyperparameter(CSH.CategoricalHyperparameter(name='num_hidden_layers', choices=[0,1,2]))
    cs.add_hyperparameter(CSH.CategoricalHyperparameter(name='freeze_last_resnet_layer', choices=[True, False]))
    cs.add_hyperparameter(CSH.UniformFloatHyperparameter(name='lr', lower=1e-4, upper=1e-2, log=True))
    # cs.add_hyperparameter(CSH.CategoricalHyperparameter(name='use_softmax', choices=[True, False]))
    # cs.add_hyperparameter(CSH.CategoricalHyperparameter(name='shuffle_samples', choices=[True, False]))
    # cs.add_hyperparameter(CSH.UniformFloatHyperparameter(name='dropout', lower=1e-2, upper=1e0, log=True))

    # cs.add_hyperparameter(CSH.CategoricalHyperparameter(name='optimizer', choices=['Adam', 'SGD']))
    # cs.add_hyperparameter(CSH.CategoricalHyperparameter(name='model_input_size', choices=[32,64,128]))

    # cs.add_hyperparameter(CSH.CategoricalHyperparameter(name='model_base', choices=['resnet18','resnet50']))
    # cs.add_hyperparameter(CSH.CategoricalHyperparameter(name='model_fc_num', choices=[1,2,3]))
    # cs.add_hyperparameter(CSH.CategoricalHyperparameter(name='model_fc_width', choices=[32,128,512]))
    # cs.add_hyperparameter(CSH.CategoricalHyperparameter(name='model_batch_norm', choices=[True, False]))

    return cs


def get_configuration(log_subfolder=None):
    cfg = {}
    cluster_mode = False
    if cluster_mode:
        cfg["code_dir"] = '/home/nierhoff/AutoDLComp19/src/video3/autodl_starting_kit_stable/AutoDL_sample_code_submission'
        cfg["image_dir"] = '/data/aad/image_datasets/challenge'
        cfg["video_dir"] = '/data/aad/video_datasets/challenge'
        cfg["bohb_interface"] = 'eth0'
    else:
        cfg["code_dir"] = '/home/dingsda/autodl/AutoDLComp19/src/video3/autodl_starting_kit_stable'
        cfg["image_dir"] = '/home/dingsda/data/datasets/challenge/image'
        cfg["video_dir"] = '/home/dingsda/data/datasets/challenge/video'
        cfg["bohb_interface"] = 'lo'

    log_folder = "logs_class_normal"
    if log_subfolder == None:
        cfg["bohb_log_dir"] = os.path.join(os.getcwd(), log_folder, str(int(time.time())))
    else:
        cfg["bohb_log_dir"] = os.path.join(os.getcwd(), log_folder, log_subfolder)

    cfg["bohb_min_budget"] = 30
    cfg["bohb_max_budget"] = 1000
    cfg["bohb_iterations"] = 10
    cfg["bohb_workers"] = 10
    cfg["bohb_run_id"] = '123'
    cfg['model_input_size'] = 128
    cfg['optimizer'] = 'Adam'
    cfg['lr'] = 1e-4
    cfg['train_batches'] = 1
    cfg['use_hidden_layer'] = False
    cfg['freeze_last_resnet_layer'] = False
    cfg['train_batch_size'] = 128
    #cfg["test_batch_size"] = 256
    cfg["model_base"] = 'resnet18'
    cfg['optimizer_momentum'] = 0.9
    cfg['optimizer_weight_decay'] = 1e-6
    cfg['optimizer_nesterov'] = True

    datasets = ['binary_alpha_digits', 'caltech101', 'caltech_birds2010', 'caltech_birds2011',
               'cats_vs_dogs', 'cifar10', 'cifar100', 'coil100',
               'colorectal_histology', 'deep_weeds', 'eurosat', 'fashion_mnist',
               'horses_or_humans', 'kmnist', 'mnist', 'oxford_flowers102',
               'oxford_iiit_pet', 'patch_camelyon', 'rock_paper_scissors', 'smallnorb',
               'svhn_cropped', 'tf_flowers', 'uc_merced']
               #'Chucky', 'Decal', 'Hammer', 'Hmdb51', 'Katze', 'Kraut', 'Kreatur', 'Pedro']
    # without 'SMv2', 'Ucf101', 'emnist', 'Monkeys', 'Munster',

    #datasets = ['binary_alpha_digits', 'caltech101', 'mnist', 'eurosat']

    train_datasets, test_datasets = split_datasets(datasets, 4)
    cfg["train_datasets"] = train_datasets
    cfg["test_datasets"] = test_datasets

    return cfg


class BohbWrapper(Master):
    def __init__(self, configspace=None,
                 eta=3, min_budget=0.01, max_budget=1,
                 min_points_in_model=None, top_n_percent=15,
                 num_samples=64, random_fraction=1 / 3, bandwidth_factor=3,
                 min_bandwidth=1e-3,
                 **kwargs):

        # TODO: Propper check for ConfigSpace object!
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


def load_datasets(cfg, datasets):
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


def copy_config_to_cfg(cfg, config):
    for key, value in config.items():
        cfg[key] = value
    return cfg


def load_transform(is_training, input_size):
    if is_training:
        return torchvision.transforms.Compose([
            SelectSample(),
            AlignAxes(),
            FormatChannels(channels_des=3),
            ToPilFormat(),
            #SaveImage(save_dir = '.', suffix='_2'),
            torchvision.transforms.RandomResizedCrop(size = input_size, scale=(0.7, 1.0)),
            #torchvision.transforms.Resize(size=(input_size, input_size)),
            #SaveImage(save_dir = '.', suffix='_3'),
            torchvision.transforms.RandomHorizontalFlip(),
            #SaveImage(save_dir = '.', suffix='_4'),
            #torchvision.transforms.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.01),
            #SaveImage(save_dir = '.', suffix='_5'),
            ToTorchFormat()])
            #torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
            #SaveImage(save_dir = '.', suffix='_6')])
    else:
        return torchvision.transforms.Compose([
            SelectSample(),
            AlignAxes(),
            FormatChannels(channels_des=3),
            ToPilFormat(),
            torchvision.transforms.Resize(size=(input_size, input_size)),
            #torchvision.transforms.Resize(int(input_size*1.1)),
            #torchvision.transforms.CenterCrop(input_size),
            ToTorchFormat()])
            #torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


def execute_run(config_id, cfg, config, budget):
    dataset_list = load_datasets(cfg, cfg["train_datasets"])
    cfg = copy_config_to_cfg(cfg, config)
    num_classes = len(dataset_list)

    m = Model(config_id = config_id, num_classes = num_classes, cfg = cfg)

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
                       desired_batches = cfg["train_batches"],
                       iteration = i,
                       save_iteration=1)

        loss_list.append(loss)

    avg_loss = sum(loss_list) / len(loss_list)

    print(' ')
    acc_list = []

    for i in range(num_classes):
        selected_class = i % num_classes
        dataset_test  = dataset_list[selected_class][1]
        dataset_name  = dataset_list[selected_class][2]
        class_index   = dataset_list[selected_class][3]

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


class WrapperModel(torch.nn.Module):
    def __init__(self, config_id, num_classes, cfg):
        super().__init__()

        self.filename = cfg["bohb_log_dir"] + '/' + str(config_id[0]) + '_' + str(config_id[1]) + '_' + str(config_id[2]) + '_model' + '.pt'

        self.model = torchvision.models.resnet18(pretrained=True)
        self.prelu = torch.nn.PReLU()
        #self.dropout = torch.nn.Dropout(p=cfg['dropout'])

        if cfg['num_hidden_layers'] == 0:
            self.fc = torch.nn.Sequential(torch.nn.PReLU(),
                                          torch.nn.Linear(1000 * 2, num_classes))
        elif cfg['num_hidden_layers'] == 1:
            self.fc = torch.nn.Sequential(torch.nn.PReLU(),
                                          torch.nn.Linear(1000 * 2, 300),
                                          torch.nn.PReLU(),
                                          torch.nn.Linear(300, num_classes))
        elif cfg['num_hidden_layers'] == 2:
            self.fc = torch.nn.Sequential(torch.nn.PReLU(),
                                          torch.nn.Linear(1000 * 2, 600),
                                          torch.nn.PReLU(),
                                          torch.nn.Linear(600, 150),
                                          torch.nn.PReLU(),
                                          torch.nn.Linear(150, num_classes))


        if os.path.isfile(self.filename):
            self.model.load_state_dict(torch.load(self.filename))

        # freeze base model
        for name, param in self.model.named_parameters():
            param.requires_grad = False

            if (name == 'fc.weight' or name == 'fc.bias') and cfg['freeze_last_resnet_layer'] == False:
                param.requires_grad = True

    def eval(self):
        self.model.eval()

    def test(self):
        self.model.test()

    def forward(self, x):
        x = self.model(x)
        x = torch.cat((torch.mean(x, dim=0), torch.var(x, dim=0)), 0)
        x = self.fc(x)
        x = x.unsqueeze(0)
        return x

    def save(self, suffix):
        torch.save(self.model.state_dict(), self.filename + suffix)


class WrapperOptimizer(object):
    def __init__(self, config_id, model, cfg):
        super().__init__()

        self.filename = cfg["bohb_log_dir"] + '/' + str(config_id[0]) + '_' + str(config_id[1]) + '_' + str(config_id[2]) + '_optimizer' + '.pt'

        if cfg['optimizer'] == 'SGD':
            self.optimizer =  torch.optim.SGD(model.parameters(),
                                              lr=cfg['lr'],
                                              momentum=cfg['optimizer_momentum'],
                                              weight_decay=cfg['optimizer_weight_decay'],
                                              nesterov=cfg['optimizer_nesterov'])
        elif cfg['optimizer'] == 'Adam':
            self.optimizer =  torch.optim.Adam(model.parameters(),
                                               cfg['lr'])
        else:
            raise ValueError("Unknown optimizer type: " + str(cfg['optimizer']))

        if os.path.isfile(self.filename):
            self.optimizer.load_state_dict(torch.load(self.filename))

    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self):
        self.optimizer.step()

    def save(self, suffix):
        torch.save(self.optimizer.state_dict(), self.filename + suffix)


class Model(object):
    def __init__(self, config_id, num_classes, cfg):
        super().__init__()
        self.time_start = time.time()
        self.cfg = cfg
        self.num_samples_testing = None
        self.train_dataloader_dict = {}
        self.session = tf.Session()
        self.model = WrapperModel(config_id = config_id, num_classes = num_classes, cfg = cfg)
        #self.model.save()
        #self.model = WrapperModel(config_id = config_id, num_classes = num_classes, cfg = cfg)
        self.model.cuda()
        self.criterion = torch.nn.CrossEntropyLoss().cuda()
        self.optimizer = WrapperOptimizer(config_id = config_id, model=self.model, cfg=self.cfg)
        #self.optimizer.save()
        #self.optimizer = WrapperOptimizer(config_id = config_id, model=self.model, cfg=self.cfg)


    def train(self, dataset, dataset_name, class_index, desired_batches, iteration, save_iteration=1):
        print('TRAIN START')

        dataloader = self.load_dataloader(dataset=dataset,
                                          dataset_name=dataset_name,
                                          num_samples=int(1e9),
                                          is_training=True)

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
                labels = self.format_labels(class_index, output).cuda()
                #print(output)
                #print(labels)
                loss = self.criterion(output, labels)
                #print('LOSS: ' + str(loss))
                loss.backward()
                self.optimizer.step()

                loss_list.append(loss.cpu().detach().numpy())

                if train_batches > desired_batches:
                    finish_loop = True
                    break

                #subprocess.run(['nvidia-smi'])
                train_batches += 1

        if iteration % save_iteration == 0:
            if save_iteration == 1:
                suffix = ''
            else:
                suffix = '_' + str(iteration)
            self.model.save(suffix)
            self.optimizer.save(suffix)

        print('TRAIN END')

        return sum(loss_list) / len(loss_list)


    def test(self, dataset, dataset_name, class_index):
        print('TEST START')
        dataset_temp = TFDataset(session=self.session, dataset=dataset, num_samples=10000000)
        info = dataset_temp.scan()
        dataloader = self.load_dataloader(dataset=dataset,
                                          dataset_name=dataset_name,
                                          num_samples=info['num_samples'],
                                          is_training=False)

        torch.set_grad_enabled(False)
        self.model.eval()

        t1 = time.time()

        prediction_list = []

        for data, _ in dataloader:
            prediction_list.append(self.model(data.cuda()).cpu())
        prediction = torch.cat(prediction_list, dim=0)

        acc = self.calc_accuracy(prediction, class_index)
        print("ACCURACY: " + str(dataset_name) + ' ' + str(acc))# + ' ' + str(time.time() - t1))

        print('TEST END')

        return acc


    def format_labels(self, class_index, data):
        return torch.LongTensor([class_index])
        #return torch.LongTensor([class_index]).repeat(len(data))


    def calc_accuracy(self, prediction, class_index):
        max_val, max_idx = torch.max(prediction, 1)
        acc = float(torch.sum(max_idx == int(class_index))) / float(len(prediction))
        # torch.set_printoptions(profile="full")
        # print('------------')
        # print(class_index)
        # #print(prediction)
        # #print(max_idx)
        # print(torch.sum(max_idx == int(class_index)))
        # print(len(prediction))
        # print(acc)
        # print('------------')
        # torch.set_printoptions(profile="default")
        return acc


    def load_dataloader(self, dataset, dataset_name, num_samples, is_training):
        print(dataset_name)

        transform = load_transform(is_training=is_training, input_size=self.cfg["model_input_size"])

        ds = TFDataset(
            session=self.session,
            dataset=dataset,
            num_samples=num_samples,
            transform=transform
        )

        if is_training:
            batch_size = cfg['train_batch_size']
        else:
            batch_size = cfg["train_batch_size"] * 2

        if dataset_name not in self.train_dataloader_dict or is_training is False:
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
            self.train_dataloader_dict[dataset_name] = dl

        else:
            dl = self.train_dataloader_dict[dataset_name]

        return dl



class BOHBWorker(Worker):
    def __init__(self, cfg, *args, **kwargs):
        super(BOHBWorker, self).__init__(*args, **kwargs)
        self.cfg = cfg
        print(cfg)

    def compute(self, config_id, config, budget, working_directory, *args, **kwargs):
        print("START BOHB ITERATION")
        print('CONFIG: ' + str(config))
        print('BUDGET: ' + str(budget))

        info = {}

        score = 0
        avg_loss = 0
        try:
            score, avg_loss = execute_run(config_id = config_id, cfg=cfg, config=config, budget=budget)
        except Exception as e:
            status = str(e)
            print(status)

        #info[cfg["dataset"]] = score
        info['avg_loss'] = avg_loss
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
        time.sleep(5)
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


def run_continuous_training(cfg):
    print(cfg)

    os.makedirs(cfg['bohb_log_dir'], exist_ok=True)

    dataset_list = load_datasets(cfg, cfg["train_datasets"])
    num_classes = len(dataset_list)

    m = Model(config_id=(0,0,0), num_classes=num_classes, cfg=cfg)

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
        cfg = get_configuration(sys.argv[1])
        res = runBohbParallel(cfg, sys.argv[2])
    else:
        cfg = get_configuration()
        res = runBohbSerial(cfg)
    #res = run_continuous_training(cfg)