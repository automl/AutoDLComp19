import os
import sys

sys.path.append(os.path.join(os.getcwd(), 'AutoDL_ingestion_program'))
sys.path.append(os.path.join(os.getcwd(), 'AutoDL_scoring_program'))

import random
import traceback
import numpy as np
import time
import json
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


BUDGET=5


print(sys.path)
timings = None
with open('timings.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
    timings = pickle.load(f)
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)


class BadPredictionShapeError(Exception):
    pass


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


def get_configuration(dataset, best_dataset):
    cfg = {}
    cfg["code_dir"] = '/home/nierhoff/AutoDLComp19_project/src/video3/autodl_starting_kit_stable/AutoDL_sample_code_submission'
    #cfg["code_dir"] = '/home/dingsda/autodl/AutoDLComp19/src/video3/autodl_starting_kit_stable'
    cfg["dataset"] = dataset
    cfg["nb_rep"] = 3


    # load
    with open('./common/files/best_config_dict.json', 'r') as f:
        bc_dict = json.load(f)

    # dict is stored as a string...
    for k,v in bc_dict.items():
        v[2] = eval(v[2])

    cfg["best_dataset"]     = best_dataset
    cfg["model"]            = bc_dict[best_dataset][0]
    cfg["batch_size_train"] = bc_dict[best_dataset][2]["batch_size_train"]
    cfg["dropout"]          = bc_dict[best_dataset][2]["dropout"]
    cfg["lr"]               = bc_dict[best_dataset][2]["lr"]
    cfg["optimizer"]        = bc_dict[best_dataset][2]["optimizer"]

    challenge_image_dir = '/data/aad/image_datasets/challenge'
    challenge_video_dir = '/data/aad/video_datasets/challenge'
    #challenge_image_dir = '/home/dingsda/data/datasets/challenge/image'
    #challenge_video_dir = '/home/dingsda/data/datasets/challenge/video'
    challenge_image_dataset = os.path.join(challenge_image_dir, dataset)
    challenge_video_dataset = os.path.join(challenge_video_dir, dataset)

    if os.path.isdir(challenge_image_dataset):
        cfg['dataset_dir'] = challenge_image_dataset
    elif os.path.isdir(challenge_video_dataset):
        cfg['dataset_dir'] = challenge_video_dataset
    else:
        raise ValueError('unknown dataset type: ' + str(dataset))

    return cfg


def execute_run(cfg, budget):
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

    M = Model(D_train.get_metadata(), cfg, {})  # The metadata of D_train and D_test only differ in sample_count
    ds_temp = TFDataset(session=tf.Session(), dataset=D_train.get_dataset())
    info = ds_temp.scan(250)

    t_list = []
    score_list = []
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
        ts = timings[model_name][str(cfg['batch_size_train'])][precalc_size][:-2]
    else:
        ts = timings[model_name][str(cfg['batch_size_train'])][precalc_size]
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

def compute(cfg, budget):
    print("START ITERATION")
    print('CONFIG: ' + str(cfg))
    print('BUDGET: ' + str(budget))

    score = 0
    try:
        print('ON DATASET        : ' + str(cfg["dataset"]))
        print('WITH MODEL        : ' + str(cfg["model"]))
        print('WITH BEST DATASET : ' + str(cfg["best_dataset"]))

        score_list = []
        for i in range(cfg["nb_rep"]):
            score_list.append(execute_run(cfg=cfg, budget=budget))

        score_list.sort()
        score = sum(score_list) / len(score_list)
    except Exception:
        status = traceback.format_exc()
        print(status)

    print('----------------------------')
    print('FINAL SCORE: ' + str(score))
    print('----------------------------')
    print("END ITERATION")

    return score


def compute_and_log(dataset, best_datasets):
    result_list = []

    for best_dataset in best_datasets:
        cfg = get_configuration(dataset, best_dataset)
        res = compute(cfg, budget=BUDGET)
        result_list.append([best_dataset, res, cfg])

    json.dump(result_list, open(os.path.join(os.getcwd(), '_'+dataset+'.json'), 'w'))


if __name__ == "__main__":
    datasets = ['Chucky', 'Decal', 'Hammer', 'Katze', 'Kraut', 'Kreatur', 'Munster', 'Pedro']
    best_datasets = ['binary_alpha_digits', 'caltech101', 'caltech_birds2010', 'caltech_birds2011',  # 4
                     'cats_vs_dogs', 'cifar10', 'cifar100', 'coil100', 'colorectal_histology',       # 5
                     'deep_weeds', 'emnist', 'eurosat', 'fashion_mnist',                             # 4
                     'horses_or_humans', 'kmnist', 'mnist',                                          # 3
                     'oxford_flowers102', 'oxford_iiit_pet', 'patch_camelyon', 'rock_paper_scissors',# 4
                     'smallnorb', 'svhn_cropped', 'tf_flowers', 'uc_merced',                         # 4
                     'Chucky', 'Decal', 'Hammer', 'Hmdb51', 'Katze', 'Kraut', 'Kreatur', 'miniciao', # 8
                     'Monkeys', 'Munster', 'Pedro', 'SMv2', 'Ucf101']                                # 5

    if len(sys.argv) == 3:      # parallel processing
        for arg in sys.argv[1:]:
            print(arg)

        id = int(sys.argv[1])
        tot = int(sys.argv[2])
        for i, dataset in enumerate(datasets):
            if (i-id)%tot != 0:
                continue

            compute_and_log(dataset, best_datasets)


    else:                       # serial processing
        for dataset in datasets:
            compute_and_log(dataset, best_datasets)

