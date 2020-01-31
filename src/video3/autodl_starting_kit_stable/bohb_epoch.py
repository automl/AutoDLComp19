import os
import sys

sys.path.append(os.path.join(os.getcwd(), 'AutoDL_ingestion_program'))
sys.path.append(os.path.join(os.getcwd(), 'AutoDL_scoring_program'))

import random
import traceback
import numpy as np
import time
from sklearn.metrics import auc
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
import hpbandster.core.nameserver as hpns
import hpbandster.core.result as hpres
from hpbandster.core.worker import Worker
from hpbandster.optimizers import BOHB as BOHB
from AutoDL_ingestion_program.dataset import AutoDLDataset
from AutoDL_scoring_program.score import get_solution, accuracy, is_multiclass, autodl_auc
from model_epoch.model import Model
from common.utils import TFDataset
import tensorflow as tf
import logging
import pickle


print(sys.path)
timings = None
with open('timings.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
    timings = pickle.load(f)
LOGGER = logging.getLogger(__name__)

class BadPredictionShapeError(Exception):
    pass

def get_configspace():
    cs = CS.ConfigurationSpace()
    cs.add_hyperparameter(CSH.CategoricalHyperparameter(name='batch_size_train', choices = [16,32,64,128]))
    cs.add_hyperparameter(CSH.UniformFloatHyperparameter(name='dropout', lower=0.01, upper=0.99, log=False))
    cs.add_hyperparameter(CSH.UniformFloatHyperparameter(name='lr', lower=1e-5, upper=0.5, log=True))
    cs.add_hyperparameter(CSH.CategoricalHyperparameter(name='optimizer', choices=['Adam', 'SGD']))
    return cs

def transform_time(t, T, t0=None):
    """Logarithmic time scaling Transform for ALC """
    if t0 is None:
        t0 = T
    return np.log(1 + t / t0) / np.log(1 + T / t0)

def calculate_alc(timestamps, scores, start_time=0, time_budget=7200):
    """Calculate ALC """
    ######################################################
    # Transform X to relative time points
    timestamps = [t for t in timestamps if t <= time_budget + start_time]
    t0 = 60
    transform = lambda t: transform_time(t, time_budget, t0=t0)
    relative_timestamps = [t - start_time for t in timestamps]
    Times = [transform(t) for t in relative_timestamps]
    ######################################################
    Scores = scores.copy()
    # Add origin as the first point of the curve and end as last point
    ######################################################
    Times.insert(0, 0)
    Scores.insert(0, 0)
    Times.append(1)
    Scores.append(Scores[-1])
    ######################################################
    # Compute AUC using step function rule or trapezoidal rule
    alc = auc(Times, Scores)
    return alc


def get_configuration(dataset, model):
    cfg = {}
    cfg["code_dir"] = '/home/nierhoff/AutoDLComp19/src/video3/autodl_starting_kit_stable/AutoDL_sample_code_submission'
    #cfg["code_dir"] = '/home/dingsda/autodl/AutoDLComp19/src/video3/autodl_starting_kit_stable'
    #cfg["code_dir"] = '/home/human/repos/AutoDLComp19/src/video3/autodl_starting_kit_stable/AutoDL_sample_code_submission'
    cfg["dataset"] = dataset
    cfg["model"] = model
    cfg["bohb_min_budget"] = 1
    cfg["bohb_max_budget"] = 9
    cfg["bohb_iterations"] = 10
    cfg["bohb_eta"] = 3
    cfg["bohb_log_dir"] = "./logs_new/" + dataset + '___' + model + '___' + str(int(time.time()))
    cfg["auc_splits"] = 10  # Unused

    challenge_image_dir = '/data/aad/image_datasets/challenge'
    challenge_video_dir = '/data/aad/video_datasets/challenge'
    #challenge_image_dir = '/home/dingsda/data/datasets/challenge/image'
    #challenge_video_dir = '/home/dingsda/data/datasets/challenge/video'
    #challenge_image_dir = '/home/human/repos/AutoDLComp19/datasets/'
    challenge_image_dataset = os.path.join(challenge_image_dir, dataset)
    challenge_video_dataset = os.path.join(challenge_video_dir, dataset)

    if os.path.isdir(challenge_image_dataset):
        cfg['dataset_dir'] = challenge_image_dataset
    elif os.path.isdir(challenge_video_dataset):
        cfg['dataset_dir'] = challenge_video_dataset
    else:
        raise ValueError('unknown dataset type: ' + str(dataset))

    return cfg


def calc_avg(score_list):
    score = sum(score_list) / len(score_list)

    print('--------- score list ---------')
    print(score_list)
    print('--------- score ---------')
    print(score)

    return score


def execute_run(cfg, config, budget):
    dataset_dir = cfg['dataset_dir']
    dataset = cfg['dataset']

    lower_path = os.path.join(dataset_dir, (dataset+'.data').lower())
    capital_path = os.path.join(dataset_dir, dataset+'.data')
    if os.path.exists(lower_path):
        D_train = AutoDLDataset(os.path.join(lower_path, 'train'))
        D_test = AutoDLDataset(os.path.join(lower_path, 'test'))
    else:
        D_train = AutoDLDataset(os.path.join(capital_path, 'train'))
        D_test = AutoDLDataset(os.path.join(capital_path, 'test'))

    ## Get correct prediction shape
    num_examples_test = D_test.get_metadata().size()
    output_dim = D_test.get_metadata().get_output_size()
    correct_prediction_shape = (num_examples_test, output_dim)

    M = Model(D_train.get_metadata(), cfg, config)  # The metadata of D_train and D_test only differ in sample_count
    ds_temp = TFDataset(session=tf.Session(), dataset=D_train.get_dataset())
    info = ds_temp.scan(250)

    t_list = []
    score_list = []
    nb_splits = cfg['auc_splits']
    # Each model, input, bs, dataset has
    # [[sec,batches,test_frames_per_sec],[...]]
    # corresponding to approx [3,...,...][10,][30][100][300]
    model_name = cfg['model']
    # Calculate the dimension used for timing
    if info['avg_shape'][1] < 45 or info['avg_shape'][2] < 45:
        precalc_size = 32
    elif info['avg_shape'][1] < 85 or info['avg_shape'][2] < 85:
        precalc_size = 64
    elif info['avg_shape'][1] < 145 or info['avg_shape'][2] < 145:
        precalc_size = 128
    else:
        precalc_size = 256
    # for budget 1 use up to 30 seconds and for budget 5 use up to 300 sec
    if budget < 2:
        ts = timings[model_name][str(config['batch_size_train'])][precalc_size][:-2]
    else:
        ts = timings[model_name][str(config['batch_size_train'])][precalc_size]
    # Calculate time for full test
    time_for_test = num_examples_test / ts[0][2]
    for t in ts:
        t_cur = M.train(D_train.get_dataset(), desired_batches=int(t[1]))
        t_list.append(t[1] + time_for_test)
        prediction = M.test(D_test.get_dataset())

        if prediction is None:  # Stop train/predict process if Y_pred is None
            LOGGER.info("The method model.test returned `None`. " +
                        "Stop train/predict process.")
        else:  # Check if the prediction has good shape
            prediction_shape = tuple(prediction.shape)
            if prediction_shape != correct_prediction_shape:
                raise BadPredictionShapeError(
                    "Bad prediction shape! Expected {} but got {}." \
                        .format(correct_prediction_shape, prediction_shape)
                )

        solution = get_solution(dataset_dir)
        score = autodl_auc(solution, prediction)
        score_list.append(score)

    # calc alc score
    alc = calculate_alc(t_list, score_list, start_time=0, time_budget=7200)
    return alc


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
            print('BOHB ON DATASET: ' + str(cfg["dataset"]))
            print('BOHB WITH MODEL: ' + str(cfg["model"]))
            score = execute_run(cfg=cfg, config=config, budget=budget)
        except Exception:
            status = traceback.format_exc()
            print(status)

        info[cfg["dataset"]] = score
        info['config'] = str(config)

        print('----------------------------')
        print('FINAL SCORE: ' + str(score))
        print('----------------------------')
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
        eta=cfg["bohb_eta"],
        nameserver="127.0.0.1",
        nameserver_port=port,
        result_logger=result_logger,
    )

    res = bohb.run(n_iterations=cfg["bohb_iterations"])
    bohb.shutdown(shutdown_workers=True)
    ns.shutdown()

    return res


if __name__ == "__main__":
    datasets = ['binary_alpha_digits', 'caltech101', 'caltech_birds2010', 'caltech_birds2011', # 4
               'cats_vs_dogs', 'cifar10', 'cifar100', 'coil100', 'colorectal_histology',       # 5
               'deep_weeds', 'emnist', 'eurosat', 'fashion_mnist', 'food101',                  # 5
               'horses_or_humans', 'kmnist', 'mnist', 'omniglot',                              # 4
               'oxford_flowers102', 'oxford_iiit_pet', 'patch_camelyon', 'rock_paper_scissors',# 4
               'smallnorb', 'stanford_dogs', 'svhn_cropped', 'tf_flowers', 'uc_merced',        # 5
               'Chucky', 'Decal', 'Hammer', 'Hmdb51', 'Katze', 'Kraut', 'Kreatur', 'miniciao', # 8
               'Monkeys', 'Munster', 'Pedro', 'SMv2', 'Ucf101']                                # 5
    models = ['squeezenet_32', 'squeezenet_64', 'squeezenet_128', 'squeezenet_224',
              'shufflenet05_32','shufflenet05_64', 'shufflenet05_128', 'shufflenet05_224',
              'shufflenet10_32', 'shufflenet10_64', 'shufflenet10_128', 'shufflenet10_224',
              'shufflenet20_32', 'shufflenet20_64', 'shufflenet20_128', 'shufflenet20_224',
              'resnet18_32', 'resnet18_64', 'resnet18_128', 'resnet18_224', #  'mobilenetv2_64',
              'efficientnet_b07_32', 'efficientnet_b07_64', 'efficientnet_b07_128', 'efficientnet_b07_224',
              'efficientnet_b05_32', 'efficientnet_b05_64', 'efficientnet_b05_128', 'efficientnet_b05_224',
              'efficientnet_b03_32', 'efficientnet_b03_64', 'efficientnet_b03_128', 'efficientnet_b03_224',
              'efficientnet_b0_32', 'efficientnet_b0_64', 'efficientnet_b0_128', 'efficientnet_b0_224',
              'efficientnet_pytorch_32', 'efficientnet_pytorch_64', 'efficientnet_pytorch_128', 'efficientnet_pytorch_224',
              'densenet05_32', 'densenet05_64', 'densenet05_128', 'densenet05_224',
              'densenet025_32', 'densenet025_64', 'densenet025_128','densenet025_224',
              'densenet_32', 'densenet_64', 'densenet_128', 'densenet_224']

    if len(sys.argv) == 3:      # parallel processing
        for arg in sys.argv[1:]:
            print(arg)

        id = int(sys.argv[1])
        tot = int(sys.argv[2])
        for i, dataset in enumerate(datasets):
            if (i-id)%tot != 0:
                continue
            for model in models:
                cfg = get_configuration(dataset, model)
                res = runBOHB(cfg)
    else:                       # serial processing
        for dataset in datasets:
            for model in models:
                cfg = get_configuration(dataset, model)
                res = runBOHB(cfg)
