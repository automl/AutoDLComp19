import os
import logging
import argparse
import numpy as np

from smac.configspace import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter, \
    UniformFloatHyperparameter, UniformIntegerHyperparameter

from smac.scenario.scenario import Scenario
from smac.facade.smac_hpo_facade import SMAC4HPO

import sys
sys.path.append('../')
sys.path.append('../../')

from model import Model
from autonlp_starting_kit.AutoDL_scoring_program.libscores import auc_metric
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
                preds = M.test(test_x)
                score = auc_metric(test_y.astype(int), preds.astype(int))
            pass
    except TimeoutException as e:
        print(e)
    return score


def run_smac():
    cs = ConfigurationSpace()

    # classifier hyperparams
    # TODO : increase batch_size upper when running on cluster
    batch_size = UniformIntegerHyperparameter("batch_size", lower=8, upper=64,
                                              default_value=32, log=False)
    layers = UniformIntegerHyperparameter("classifier_layers", lower=1, upper=3,
                                              default_value=2, log=False)
    classifier_units = UniformIntegerHyperparameter("classifier_units",
                                                    lower=min(metadata['class_num'], 512),
                                                    upper=512, default_value=256, log=False)
    cs.add_hyperparameters([batch_size, layers, classifier_units])

    # training hyperparams
    learning_rate = UniformFloatHyperparameter("learning_rate", lower=0.0001, upper=0.1,
                                               default_value=0.001, log=True)
    cs.add_hyperparameter(learning_rate)

    # SMAC scenario oject
    scenario = Scenario({"run_obj": "quality",   # we optimize quality (alternative runtime)
                         "cs": cs,               # configuration space
                         "deterministic": "true",
                         "wallclock_limit": args.wallclock_time
                         })

    # To optimize, we pass the function to the SMAC-object
    smac = SMAC4HPO(scenario=scenario, rng=np.random.RandomState(42),
                    tae_runner=run_autonlp_model)

    try:
        incumbent = smac.optimize()
    except:
        incumbent = smac.solver.incumbent

    incumbent_score = run_autonlp_model(incumbent)

    return incumbent, incumbent_score


logger = logging.getLogger("AutoNLP-SMAC")
logging.basicConfig(level=logging.INFO)
logging.info("Reading arguments")

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-d", "--dataset", dest="dataset_id", type=int, default=1, choices=[1, 2, 3, 4, 5],
                    help='Which dataset to evaluate on from {1, 2, 3, 4, 5}')
parser.add_argument("-p", "--path", dest="data_dir", type=str, default='offline_data/',
                    help='The path to offline data')
parser.add_argument('-w', "--wallclock", dest="wallclock_time", type=float, default=300,
                    help='Total time to run for SMAC in seconds.')
parser.add_argument('-c', "--cutoff", dest="cutoff", type=float, default=60,
                    help='Maximum time for function evaluation.')

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

from ConfigSpace.read_and_write import json
with open('incumbent.json', 'w') as f:
    f.write(json.write(cs))