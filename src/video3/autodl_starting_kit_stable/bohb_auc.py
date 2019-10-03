import random
import traceback
import _pickle as pickle
import os
import time
import sys
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
import hpbandster.core.nameserver as hpns
import hpbandster.core.result as hpres
import hpbandster.visualization as hpvis
import matplotlib.pyplot as plt
from hpbandster.core.worker import Worker
from hpbandster.optimizers import BOHB as BOHB


def get_configspace():
    cs = CS.ConfigurationSpace()
    cs.add_hyperparameter(CSH.CategoricalHyperparameter(name='batch_size_train', choices = [16,32,64,128]))
    cs.add_hyperparameter(CSH.UniformFloatHyperparameter(name='dropout', lower=0.1, upper=0.9, log=False))
    cs.add_hyperparameter(CSH.UniformFloatHyperparameter(name='lr', lower=1e-5, upper=1e-2, log=True))
    cs.add_hyperparameter(CSH.CategoricalHyperparameter(name='optimizer', choices=['Adam', 'SGD']))
    cs.add_hyperparameter(CSH.CategoricalHyperparameter(name='model',
        choices=['squeezenet_64', 'squeezenet_128', 'squeezenet_224',
                 'shufflenet05_64', 'shufflenet05_128', 'shufflenet05_224',
                 'shufflenet10_64', 'shufflenet10_128', 'shufflenet10_224',
                 'shufflenet20_64', 'shufflenet20_128', 'shufflenet20_224',
                 'resnet18_64', 'resnet18_128', 'resnet18_224',
                 'mobilenetv2_64',
                 'efficientnet_b07_64', 'efficientnet_b07_128', 'efficientnet_b07_224',
                 'efficientnet_b05_64', 'efficientnet_b05_128', 'efficientnet_b05_224',
                 'efficientnet_b03_64', 'efficientnet_b03_128', 'efficientnet_b03_224',
                 'efficientnet_b0_64', 'efficientnet_b0_128', 'efficientnet_b0_224',
                 'densenet05_64', 'densenet05_128', 'densenet05_224',
                 'densenet025_64', 'densenet025_128',
                 'densenet_64', 'densenet_128', 'densenet_224']))

    return cs


def get_configuration(dataset):
    cfg = {}
    cfg["code_dir"] = '/home/dingsda/autodl/AutoDLComp19/src/video3/autodl_starting_kit_stable/auc'
    cfg["dataset"] = dataset
    cfg["bohb_min_budget"] = 30
    cfg["bohb_max_budget"] = 300
    cfg["bohb_iterations"] = 20
    cfg["bohb_log_dir"] = "./logs/" + dataset + '_data_' + str(int(time.time()))

    challenge_image_dir = '/home/dingsda/data/datasets/challenge/image/'
    challenge_video_dir = '/home/dingsda/data/datasets/challenge/video/'
    challenge_image_dataset = os.path.join(challenge_image_dir, dataset)
    challenge_video_dataset = os.path.join(challenge_video_dir, dataset)

    if os.path.isdir(challenge_image_dataset):
        cfg['dataset_dir'] = challenge_image_dataset
    elif os.path.isdir(challenge_video_dataset):
        cfg['dataset_dir'] = challenge_video_dataset
    else:
        raise ValueError('unknown dataset type: ' + str(dataset))

    return cfg


def write_config_to_file(cfg):
    path = os.path.join(os.getcwd(), 'bohb_config.txt')
    with open(path, 'wb') as file:
        pickle.dump(cfg, file)  # use `json.loads` to do the reverse


def create_function_call(cfg, budget):
    fc = 'python3 run_local_test.py'
    fc += ' --code_dir=' + cfg["code_dir"]
    fc += ' --dataset_dir=' + cfg["dataset_dir"]
    fc += ' --time_budget=' + str(budget)
    return fc


def read_final_score_from_file():
    path = os.path.join(os.getcwd(), 'AutoDL_scoring_output')
    path = os.path.join(path, 'final_score.txt')

    with open(path, "r") as file:
        score = float(file.read())

    return score


class BOHBWorker(Worker):
    def __init__(self, cfg, *args, **kwargs):
        super(BOHBWorker, self).__init__(*args, **kwargs)
        self.cfg = cfg
        print(cfg)

    def compute(self, config, budget, *args, **kwargs):
        print("START BOHB ITERATION")
        print('CONFIG: ' + str(config))
        print('BUDGET: ' + str(budget))

        info = {}

        score = 0
        try:
            print('BOHB ON DATASET: ' + cfg["dataset"])
            # stored bohb config will be read again in model.py
            write_config_to_file(config)
            # execute main function
            fc = create_function_call(cfg, budget)
            os.system(fc)
            # read final score from score.py
            score = read_final_score_from_file()
        except Exception:
            status = traceback.format_exc()
            print(status)

        info[cfg["dataset"]] = score
        info['config'] = str(config)

        print('FINAL SCORE: ' + str(score))
        print("END BOHB ITERATION")

        return {
            "loss": -score,
            "info": info
        }


def runBOHB(cfg):
    run_id = "0"

    # assign random port in the 30000-40000 range to avoid using a blocked port because of a previous improper bohb shutdown
    port = int(30000 + random.random() * 10000)

    ns = hpns.NameServer(run_id=run_id, host="127.0.0.1", port=port)
    ns.start()

    w = BOHBWorker(cfg=cfg, nameserver="127.0.0.1", run_id=run_id, nameserver_port=port)
    w.run(background=True)

    result_logger = hpres.json_result_logger(
        directory=cfg["bohb_log_dir"], overwrite=True
    )

    bohb = BOHB(
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


if __name__ == "__main__":
    datasets = ['binary_alpha_digits', 'caltech101', 'caltech_birds2010', 'caltech_birds2011',
               'cats_vs_dogs', 'cifar10', 'cifar100', 'coil100', 'colorectal_histology',
               'deep_weeds', 'emnist', 'eurosat', 'fashion_mnist', 'food101',
               'horses_or_humans', 'kmnist', 'mnist', 'omniglot',
               'oxford_flowers102', 'oxford_iiit_pet', 'patch_camelyon', 'rock_paper_scissors',
               'smallnorb', 'stanford_dogs', 'svhn_cropped', 'tf_flowers', 'uc_merced',
               'Chucky', 'Decal', 'Hammer', 'Hmdb51', 'Katze', 'Kraut', 'Kreatur', 'miniciao',
               'Monkeys', 'Munster', 'Pedro', 'SMv2', 'Ucf101']

    if len(sys.argv) == 3:      # parallel processing
        for arg in sys.argv[1:]:
            print(arg)

        id = int(sys.argv[1])
        tot = int(sys.argv[2])
        for i, dataset in enumerate(datasets):
            if (i-id)%tot != 0:
                continue
            cfg = get_configuration(dataset)
            res = runBOHB(cfg)
    else:                       # serial processing
        for dataset in datasets:
            cfg = get_configuration(dataset)
            res = runBOHB(cfg)
