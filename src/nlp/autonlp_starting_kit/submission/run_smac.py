import os
import logging
import argparse
import numpy as np

from smac.configspace import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter, \
    UniformFloatHyperparameter, UniformIntegerHyperparameter
from ConfigSpace.conditions import EqualsCondition

from smac.scenario.scenario import Scenario
from smac.facade.smac_hpo_facade import SMAC4HPO

import sys
sys.path.append('../')
sys.path.append('../../')

from model import Model
from scoring import autodl_auc
# from autonlp_starting_kit.AutoDL_scoring_program.libscores import auc_metric
from autonlp_starting_kit.AutoDL_ingestion_program.ingestion import Timer, TimeoutException
from autonlp_starting_kit.AutoDL_ingestion_program.dataset import AutoNLPDataset


def load_dataset(num = 1):
  sub_path = 'O{}/O{}.data'.format(num, num)
  dataset = AutoNLPDataset(dataset_dir = os.path.join(data_dir, sub_path))
  dataset.read_dataset()
  dataset.test_label = load_test_dataset(num)
  return dataset


def load_test_dataset(num = 1):
  dataset_path = os.path.join(data_dir, "O{}/O{}.solution".format(num, num))
  return np.loadtxt(dataset_path)


def run_autonlp_model(config, **kwargs):
    timeout = kwargs['cutoff'] if 'cutoff' in kwargs else args.cutoff

    timer = Timer()
    timer.set(timeout)

    # ASSUME: train and test dataset defined in the script
    test_x, test_y = test_dataset

    M = Model(metadata, config)
    try:
        with timer.time_limit('training'):
            # Training loop
            while True:  # num_epochs = inf (till timeout)
                M.train(train_dataset)
    except TimeoutException as e:
        print(e)
    preds = M.test(test_x)
    score = autodl_auc(test_y.astype(int), preds.astype(int))
    print("run_autonlp_model score: {}".format(score))
    return score


def run_smac():
    cs = ConfigurationSpace()

    # model hyperparams
    encoder_layers = UniformIntegerHyperparameter("layers", lower=1, upper=6, default_value=2, log=False)
    # TODO : increase batch_size upper when running on cluster
    batch_size = UniformIntegerHyperparameter("batch_size", lower=8, upper=64,
                                              default_value=32, log=False)
    layers = UniformIntegerHyperparameter("classifier_layers", lower=1, upper=3,
                                              default_value=2, log=False)
    classifier_units = UniformIntegerHyperparameter("classifier_units",
                                                    lower=min(metadata['class_num'], 512),
                                                    upper=512, default_value=256, log=False)
    cs.add_hyperparameters([encoder_layers, batch_size, layers, classifier_units])

    #preprocessing hyperparameters
    str_cutoff = UniformIntegerHyperparameter("str_cutoff", lower=50, upper=100,
                                                default_value=75, log=False)
    features = UniformIntegerHyperparameter("features", lower=500, upper=2500,
                                            default_value=2000, log=False)
    cs.add_hyperparameters([str_cutoff, features])

    # training hyperparams
    learning_rate = UniformFloatHyperparameter("learning_rate", lower=0.0001, upper=0.1,
                                               default_value=0.001, log=True)
    optimizer = CategoricalHyperparameter("optimizer", ["adam", "adamw"], default_value="adam")
    weight_decay = UniformFloatHyperparameter("weight_decay", lower=0.00001, upper=0.1,
                                              default_value=0.01, log=True)
    stop_count = UniformIntegerHyperparameter("stop_count", lower=2, upper=5, default_value=5,
                                              log=False)
    cs.add_hyperparameters([learning_rate, optimizer, weight_decay, stop_count])
    optim_cond = EqualsCondition(weight_decay, optimizer, 'adamw')
    cs.add_condition(optim_cond)

    # SMAC scenario oject
    scenario = Scenario({"run_obj": "quality",   # we optimize quality (alternative runtime)
                         "cs": cs,               # configuration space
                         "deterministic": "true",
                         "runcount_limit": args.runcount})
                         # "wallclock_limit": args.wallclock_time})

    # To optimize, we pass the function to the SMAC-object
    smac = SMAC4HPO(scenario=scenario, rng=np.random.RandomState(42),
                    tae_runner=run_autonlp_model)

    try:
        incumbent = smac.optimize()
        print("try")
    except:
        incumbent = smac.solver.incumbent
        print("except")
    print("Inside SMAC, incumbent found: ")
    print(incumbent)
    incumbent_score = run_autonlp_model(incumbent)

    return incumbent, incumbent_score


logger = logging.getLogger("AutoNLP-SMAC")
logging.basicConfig(level=logging.DEBUG)
logging.info("Reading arguments")

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-d", "--dataset", dest="dataset_id", type=int, default=1, choices=[1, 2, 3, 4, 5],
                    help='Which dataset to evaluate on from {1, 2, 3, 4, 5}')
parser.add_argument("-p", "--path", dest="data_dir", type=str, default='offline_data/',
                    help='The path to offline data')
parser.add_argument('-r', "--runcount", dest="runcount", type=int, default=50,
                    help='Total evaluations of the model.')
parser.add_argument('-c', "--cutoff", dest="cutoff", type=float, default=60,
                    help='Maximum time for target function evaluation.')

args, kwargs = parser.parse_known_args()

data_dir = args.data_dir
dataset = load_dataset(args.dataset_id)
# necessary assignments
train_dataset = dataset.get_train()
test_dataset = (dataset.test_dataset, dataset.test_label)
metadata = dataset.metadata_

incumbent, incumbent_score = run_smac()

print("Incumbent configuration: ")
print(incumbent)
print("Incumbent Score: {}".format(incumbent_score))

import json
with open('incumbent_{}_{}.json'.format(args.dataset_id, int(args.wallclock_time)), 'w') as f:
    json.dump(incumbent.get_dictionary(), f)
