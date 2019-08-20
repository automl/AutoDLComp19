import os
import logging
import argparse
import numpy as np
import time
import json

from smac.configspace import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter, \
    UniformFloatHyperparameter, UniformIntegerHyperparameter
from ConfigSpace.conditions import EqualsCondition, InCondition

from smac.scenario.scenario import Scenario
from smac.facade.smac_hpo_facade import SMAC4HPO

import sys
sys.path.append('../')
sys.path.append('../../')

from model import Model
from scoring import autodl_auc
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

    M = Model(metadata=metadata, config=config)
    try:
        with timer.time_limit('training'):
            # Training loop
            while True:  # num_epochs = inf (till timeout)
                M.train(train_dataset)
                preds = M.test(test_x)
                if M.done_training:
                    break
    except TimeoutException as e:
        print(e)

    score = autodl_auc(test_y.astype(int), preds.astype(int))
    print("run_autonlp_model score: {}".format(score))
    return -1*score  # since smac minimizes


def run_smac():

    # get output dir from arguments
    time_id = time.strftime('%Y%m%d-%H%M%S')
    out_dir = 'smac_output_{}d_{}c_{}'.format(args.dataset_id, args.cutoff, time_id)

    cs = ConfigurationSpace()

    # model hyperparams
    transformer = CategoricalHyperparameter("transformer", ['bert', 'xlnet'], default_value="xlnet")
    encoder_layers = UniformIntegerHyperparameter("layers", lower=1, upper=5, default_value=2, log=False)
    finetune_wait = UniformIntegerHyperparameter("finetune_wait", lower=0, upper=5, default_value=2, log=False)
    layers = UniformIntegerHyperparameter("classifier_layers", lower=1, upper=5, default_value=2, log=False)
    classifier_units = UniformIntegerHyperparameter("classifier_units", lower=32, upper=512,
                                                    default_value=256, log=False)
    cs.add_hyperparameters([transformer, encoder_layers, finetune_wait, layers, classifier_units])

    # training hyperparams
    optimizer = CategoricalHyperparameter("optimizer", ["radam", "adabound", "adamw"], default_value="adabound")
    learning_rate = UniformFloatHyperparameter("learning_rate", lower=0.0001, upper=0.1,
                                               default_value=0.001, log=True)
    weight_decay = UniformFloatHyperparameter("weight_decay", lower=0.00001, upper=0.1,
                                              default_value=0.001, log=True)
    batch_size = UniformIntegerHyperparameter("batch_size", lower=8, upper=128,
                                              default_value=64, log=False)
    stop_count = UniformIntegerHyperparameter("stop_count", lower=1, upper=25, default_value=10,
                                              log=False)
    cs.add_hyperparameters([learning_rate, optimizer, weight_decay, batch_size, stop_count])
    optim_cond = InCondition(weight_decay, optimizer, ['adamw', 'adabound'])
    cs.add_condition(optim_cond)

    # preprocessing hyperparameters
    str_cutoff = UniformIntegerHyperparameter("str_cutoff", lower=50, upper=100,
                                              default_value=75, log=False)
    cs.add_hyperparameters([str_cutoff])

    # Augmentation parameters
    augment = CategoricalHyperparameter("augmentation", [True, False], default_value=False)
    augment_th = UniformFloatHyperparameter("aug_threshold", lower=0.1, upper=0.5, log=False)
    cs.add_hyperparameters([augment, augment_th])

    # SMAC scenario oject
    scenario = Scenario({"run_obj": "quality",   # we optimize quality (alternative runtime)
                         "cs": cs,               # configuration space
                         "deterministic": "true",
                         "runcount_limit": args.runcount,
                         "wallclock_limit": args.wallclock,
                         "abort_on_first_run_crash": False,
                         "output_dir": out_dir})

    # To optimize, we pass the function to the SMAC-object
    smac = SMAC4HPO(scenario=scenario, rng=np.random.RandomState(42),
                    tae_runner=run_autonlp_model)

    print('SMAC optimization begins !')
    print('---'*40)
    #try:
    incumbent = smac.optimize()
    #    print("try")
    #except:
    #    incumbent = smac.solver.incumbent
    #    print("except")
    print("Inside SMAC, incumbent found: ")
    print(incumbent)
    incumbent = cs.get_default_configuration()
    incumbent_score = run_autonlp_model(incumbent)

    return incumbent, incumbent_score


logger = logging.getLogger("AutoNLP-SMAC")
logging.basicConfig(level=10)
logging.info("Reading arguments")

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-d", "--dataset", dest="dataset_id", type=int, default=1, choices=[1, 2, 3, 4, 5],
                    help='Which dataset to evaluate on from {1, 2, 3, 4, 5}')
parser.add_argument("-p", "--path", dest="data_dir", type=str, default='offline_data/',
                    help='The path to offline data')
parser.add_argument('-r', "--runcount", dest="runcount", type=int, default=50,
                    help='Total evaluations of the model.')
parser.add_argument('-w', "--wallclock", dest="wallclock", type=float, default=300,
                    help='Total wallclock time to run SMAC in seconds.')
parser.add_argument('-c', "--cutoff", dest="cutoff", type=float, default=60,
                    help='Maximum time for target function evaluation.')

args, kwargs = parser.parse_known_args()

print(args)

data_dir = args.data_dir
dataset = load_dataset(args.dataset_id)
# necessary assignments
train_dataset = dataset.get_train()
test_dataset = (dataset.test_dataset, dataset.test_label)
metadata = dataset.metadata_

print('Dataset loaded...')

incumbent, incumbent_score = run_smac()

print("Incumbent configuration: ")
print(incumbent)
print("Incumbent Score: {}".format(-1 * incumbent_score))
print("=" * 40)

with open('incumbent_{}_{}.json'.format(args.dataset_id, int(args.cutoff)), 'w') as f:
    json.dump(incumbent.get_dictionary(), f)
