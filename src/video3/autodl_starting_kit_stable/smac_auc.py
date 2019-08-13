import argparse
import logging
import os
import pickle
import re
import sys

import numpy as np
from ConfigSpace.conditions import InCondition
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter, UniformFloatHyperparameter, UniformIntegerHyperparameter
)
# Import ConfigSpace and different types of parameters
from smac.configspace import ConfigurationSpace
from smac.facade.smac_facade import SMAC
from smac.scenario.scenario import Scenario
# Import SMAC-utilities
from smac.tae.execute_func import ExecuteTAFuncDict

parser = argparse.ArgumentParser('smac-auc')
parser.add_argument('--dataset', type=str, default='cifar10')
parser.add_argument(
    '--dataset_path',
    type=str,
    default='/data/aad/image_datasets/tf_records/autodl_comp_format/'
)
args = parser.parse_args()

PATH = '.'


def build_configspace():
    cs = ConfigurationSpace()
    lr = UniformFloatHyperparameter("lr", 0.001, 0.1, default_value=0.025)
    optimizer = CategoricalHyperparameter(
        "optimizer", ['SGD', 'Adam'], default_value='SGD'
    )
    cs.add_hyperparameters([lr, optimizer])
    return cs


def get_arguments():
    cfg = {}
    cfg["code_dir"] = os.path.join(PATH, 'AutoDL_sample_code_submission')
    cfg["dataset_dir"] = os.path.join(args.dataset_path, args.dataset)
    return cfg


def create_function_call(cfg, budget=60):
    fc = 'python3 run_local_test.py'
    fc += ' --code_dir=' + cfg["code_dir"]
    fc += ' --dataset_dir=' + cfg["dataset_dir"]
    fc += ' --time_budget=' + str(budget)
    return fc


def read_final_score_from_file(log_path='AutoDL_scoring_output'):
    path = os.path.join(PATH, log_path)
    path = os.path.join(path, 'scores.txt')

    with open(path, "r") as file:
        score = [x for x in file.readlines()][0]
        score = float(re.findall(r"[-+]?\d*\.\d+|\d+", score)[0])

    return score


def write_config_to_file(cfg):
    paths = get_arguments()
    path = paths["code_dir"]
    path = os.path.join(path, 'smac_config.txt')
    with open(path, 'wb') as file:
        pickle.dump(cfg, file)


def network_from_cfg(cfg):
    """
    Parameters:
    -----------
    cfg: Configuration (ConfigSpace.ConfigurationSpace.Configuration)
        Configuration containing the parameters.
        Configurations are indexable!

    Returns:
    --------
    AUC of the network on the loaded data-set.
    """
    write_config_to_file(cfg)
    fc = create_function_call(get_arguments())
    os.system(fc)

    auc = read_final_score_from_file()
    return 1 - auc  # Minimize!


logging.basicConfig(level=logging.INFO)  # logging.DEBUG for debug output

cs = build_configspace()

# Scenario object
scenario = Scenario(
    {
        "run_obj": "quality",  # we optimize quality (alternatively runtime)
        "runcount-limit": 200,  # maximum function evaluations
        "cs": cs,  # configuration space
        "deterministic": "true"
    }
)

# Example call of the function
# It returns: Status, Cost, Runtime, Additional Infos
def_value = network_from_cfg(cs.get_default_configuration())
print("Default Value: %.2f" % (def_value))

# Optimize, using a SMAC-object
print("Optimizing! Depending on your machine, this might take a few minutes.")
smac = SMAC(scenario=scenario, rng=np.random.RandomState(42), tae_runner=network_from_cfg)

incumbent = smac.optimize()

inc_value = network_from_cfg(incumbent)

print("Optimized Value: %.2f" % (inc_value))

# We can also validate our results (though this makes a lot more sense with instances)
smac.validate(
    config_mode='inc',  # We can choose which configurations to evaluate
    #instance_mode='train+test',  # Defines what instances to validate
    repetitions=100,  # Ignored, unless you set "deterministic" to "false" in line 95
    n_jobs=1
)  # How many cores to use in parallel for optimization
