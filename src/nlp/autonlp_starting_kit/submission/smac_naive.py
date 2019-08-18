import os
import logging
import argparse
import numpy as np
import time
import json

from smac.configspace import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter, \
    UniformFloatHyperparameter, UniformIntegerHyperparameter
from ConfigSpace.conditions import EqualsCondition, AndConjunction

from smac.scenario.scenario import Scenario
from smac.facade.smac_hpo_facade import SMAC4HPO

import sys
sys.path.append('../')
sys.path.append('../../')

from model import Model
from scoring import autodl_auc
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

    # ASSUME: train and test dataset defined in the script
    test_x, test_y = test_dataset

    M = Model(metadata=metadata, config=config)
    # train & evaluate model
    M.train(train_dataset)
    preds = M.test(test_x)

    score = autodl_auc(test_y.astype(int), preds.astype(int))
    print("run_autonlp_model score: {}".format(score))
    return -1*score  # since smac minimizes


def run_smac():

    # get output dir from arguments
    time_id = time.strftime('%Y%m%d-%H%M%S')
    out_dir = 'smac_output_{}d_{}c_{}'.format(args.dataset_id, args.cutoff, time_id)

    cs = ConfigurationSpace()

    # preprocessing hyperparameters
    str_cutoff = UniformIntegerHyperparameter("str_cutoff", 50, 100, default_value=75, log=False)
    cs.add_hyperparameters([str_cutoff])

    # naive classifier parameters
    classifier = CategoricalHyperparameter("classifier", ["lr", "ada", "svc"], default_value="ada")
    features = UniformIntegerHyperparameter("features", 500, 3000, default_value=2000, log=False)
    cs.add_hyperparameters([classifier, features])

    # add LR parameters
    lr_multi = CategoricalHyperparameter('lr_multi', ['ovr', 'multinomial'], default_value='multinomial')
    lr_opt = CategoricalHyperparameter('lr_opt', ['lbfgs', 'saga', 'newton-cg'], default_value = 'lbfgs')
    cs.add_hyperparameters([lr_multi, lr_opt])
    lr_multi_cond = EqualsCondition(lr_multi, classifier, 'lr')
    lr_opt_cond = EqualsCondition(lr_opt, classifier, 'lr')
    cs.add_conditions([lr_multi_cond, lr_opt_cond])

    # add AdaBoost parameters
    ada_estimators = UniformIntegerHyperparameter('ada_estimators', 10, 100, default_value=50, log=False)
    ada_rate = UniformFloatHyperparameter('ada_rate', 0.01, 1, default_value=1, log=False)
    cs.add_hyperparameters([ada_estimators, ada_rate])
    ada_estim_cond = EqualsCondition(ada_estimators, classifier, 'ada')
    ada_rate_cond = EqualsCondition(ada_rate, classifier, 'ada')
    cs.add_conditions([ada_estim_cond, ada_rate_cond])

    # add SVC parameters
    svc_kernel = CategoricalHyperparameter('svc_kernel', ['rbf', 'poly'], default_value='rbf')
    cs.add_hyperparameters([svc_kernel])
    svc_kernel_cond = EqualsCondition(svc_kernel, classifier, 'svc')
    cs.add_conditions([svc_kernel_cond])

    # SMAC scenario oject
    scenario = Scenario({"run_obj": "quality",   # we optimize quality (alternative runtime)
                         "cs": cs,               # configuration space
                         "deterministic": "true",
                         "runcount_limit": args.runcount,
                         "wallclock_limit": args.wallclock,
                         # "abort_on_first_run_crash": False,
                         "cutoff": args.cutoff,
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
parser.add_argument('-w', "--wallclock", dest="wallclock", type=float, default=3000,
                    help='Total wallclock time to run SMAC in seconds.')
parser.add_argument('-c', "--cutoff", dest="cutoff", type=float, default=10,
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
