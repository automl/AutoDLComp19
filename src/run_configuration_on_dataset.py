import json
import logging
import os
import pickle

import numpy as np
import tensorflow as tf
from sklearn.metrics import auc
from src.available_datasets import available_datasets
from src.competition.ingestion_program.dataset import AutoDLDataset
from src.competition.scoring_program.score import autodl_auc, get_solution
from src.dataset_kakaobrain import TFDataset
from src.model import Model

LOGGER = logging.getLogger(__name__)


class BadPredictionShapeError(Exception):
    pass


# def get_configspace():
#     cs = CS.ConfigurationSpace()
#     cs.add_hyperparameter(CSH.CategoricalHyperparameter(name='batch_size_train', choices = [16,32,64,128]))
#     cs.add_hyperparameter(CSH.UniformFloatHyperparameter(name='dropout', lower=0.01, upper=0.99, log=False))
#     cs.add_hyperparameter(CSH.UniformFloatHyperparameter(name='lr', lower=1e-5, upper=0.5, log=True))
#     cs.add_hyperparameter(CSH.CategoricalHyperparameter(name='optimizer', choices=['Adam', 'SGD']))
#     return cs


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


def run_configuration_on_dataset(config, budget, dataset_raw_dir, model_dir):
    _, dataset_name = os.path.split(dataset_raw_dir)
    D_train = AutoDLDataset(os.path.join(dataset_raw_dir, dataset_name + ".data", "train"))
    D_test = AutoDLDataset(os.path.join(dataset_raw_dir, dataset_name + ".data", "test"))

    ## Get correct prediction shape
    num_examples_test = D_test.get_metadata().size()
    output_dim = D_test.get_metadata().get_output_size()
    correct_prediction_shape = (num_examples_test, output_dim)

    M = Model(
        D_train.get_metadata(), config, model_dir
    )  # The metadata of D_train and D_test only differ in sample_count
    ds_temp = TFDataset(session=tf.Session(), dataset=D_train.get_dataset())
    info = ds_temp.scan(25)

    t_list = []
    score_list = []

    # Each model, input, bs, dataset has
    # [[sec,batches,test_frames_per_sec],[...]]
    # corresponding to approx [3,...,...][10,][30][100][300]
    model_name = config["model_name"]
    # Calculate the dimension used for timing
    if info["avg_shape"][1] < 45 or info["avg_shape"][2] < 45:
        precalc_size = 32
    elif info["avg_shape"][1] < 85 or info["avg_shape"][2] < 85:
        precalc_size = 64
    elif info["avg_shape"][1] < 145 or info["avg_shape"][2] < 145:
        precalc_size = 128
    else:
        precalc_size = 256
    # for budget 1 use up to 30 seconds and for budget 5 use up to 300 sec
    with open("src/configs/timings.pkl", "rb") as f:  # Python 3: open(..., 'rb')
        timings = pickle.load(f)
    if budget < 2:
        ts = timings[model_name][str(config["batch_size_train"])][precalc_size][:-2]
    else:
        ts = timings[model_name][str(config["batch_size_train"])][precalc_size]
    # Calculate time for full test
    time_for_test = num_examples_test / ts[0][2]
    for t in ts:
        t_cur = M.train(D_train.get_dataset(), desired_batches=int(t[1]))
        t_list.append(t[1] + time_for_test)
        prediction = M.test(D_test.get_dataset())

        if prediction is None:  # Stop train/predict process if Y_pred is None
            LOGGER.info("The method model.test returned `None`. " + "Stop train/predict process.")
        else:  # Check if the prediction has good shape
            prediction_shape = tuple(prediction.shape)
            if prediction_shape != correct_prediction_shape:
                raise BadPredictionShapeError(
                    "Bad prediction shape! Expected {} but got {}.".format(
                        correct_prediction_shape, prediction_shape
                    )
                )

        solution = get_solution(dataset_raw_dir)
        score = autodl_auc(solution, prediction)
        score_list.append(score)

    # calc alc score
    alc = calculate_alc(t_list, score_list, start_time=0, time_budget=7200)
    return alc


def load_best_results(best_result_path):
    return json.load(open(best_result_path, "r"))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", choices=available_datasets, default="mnist")
    parser.add_argument("--datasets_dir", default="/data/aad/image_datasets/challenge")
    parser.add_argument("--models_dir", default="/home/ferreira/autodl_data/models_thomas")
    parser.add_argument("--incumbents_file", default="/home/ferreira/autodl_data/incumbent.json")
    parser.add_argument("--budget", choices=[1, 5], default=1)

    args = parser.parse_args()

    best_results = load_best_results(args.incumbents_file)
    config = eval(best_results[args.dataset][2])
    config["model_name"] = best_results[args.dataset][0]
    print(config)

    dataset_raw_dir = args.datasets_dir + "/" + args.dataset
    run_configuration_on_dataset(
        config=config, budget=5, dataset_raw_dir=dataset_raw_dir, model_dir=args.models_dir
    )
