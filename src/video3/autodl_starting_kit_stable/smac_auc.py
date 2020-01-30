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

from smac.facade.smac_ac_facade import SMAC4AC
from smac.configspace import ConfigurationSpace
#from smac.facade.smac_facade import SMAC
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
parser.add_argument('--budget', type=float, default=60.0)
parser.add_argument('--task_id', type=int, default=0)
args = parser.parse_args()

PATH = '.'
log_dir = os.path.join(
    PATH, "scoring_output/{}_{}_{}".format(args.dataset, args.budget, args.task_id)
)
if not os.path.exists(log_dir):
    os.makedirs(log_dir, exist_ok=True)


def build_configspace():
    cs = ConfigurationSpace()
    #timeslots = CategoricalHyperparameter("timeslots", ["1", "2", "3"])
    #train_duration = UniformIntegerHyperparameter(name='train_duration', lower=10,
    #                                                  upper=60, log=False)

    #dataset_size =
    #cond = cs.EqualsCondition(b, a, 1)

    lr = UniformFloatHyperparameter("lr", 0.001, 0.1, default_value=0.025)
    optimizer = CategoricalHyperparameter(
        "optimizer", ['SGD', 'Adam'], default_value='SGD'
    )
    cs.add_hyperparameters([lr, optimizer])
    return cs


def get_arguments():
    arguments = {}
    arguments["code_dir"] = os.path.join(PATH, 'AutoDL_sample_code_submission')
    arguments["dataset_dir"] = os.path.join(args.dataset_path, args.dataset)
    arguments["dataset_name"] = args.dataset
    arguments["task_id"] = args.task_id
    return arguments


def create_function_call(arguments, budget=60.0):
    fc = 'python run_local_test.py'
    fc += ' --code_dir=' + arguments["code_dir"]
    fc += ' --dataset_dir=' + arguments["dataset_dir"]
    fc += ' --time_budget=' + str(budget)
    fc += ' --task_id=' + str(args.task_id)
    return fc


def read_final_score_from_file():
    path = os.path.join(log_dir, 'scores.txt')
    with open(path, "r") as file:
        score = [x for x in file.readlines()][0]
        score = float(re.findall(r"[-+]?\d*\.\d+|\d+", score)[0])
    return score


def write_config_to_file(cfg):
    path = os.path.join(log_dir, "smac_config.txt")
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
    fc = create_function_call(get_arguments(), args.budget)
    os.system(fc)
    auc = read_final_score_from_file()
    return 1 - auc  # Minimize!


logging.basicConfig(level=logging.INFO)  # logging.DEBUG for debug output

cs = build_configspace()

# Scenario object
scenario = Scenario(
    {
        "run_obj":
            "quality",  # we optimize quality (alternatively runtime)
        "runcount-limit":
            200,  # maximum function evaluations
        "cs":
            cs,  # configuration space
        "deterministic":
            "false",
        "output_dir":
            "smac_outputs/{}_{}_{}".format(args.dataset, args.budget, args.task_id)
    }
)

# Example call of the function
# It returns: Status, Cost, Runtime, Additional Infos
def_value = network_from_cfg(cs.get_default_configuration())
print("Default Value: %.2f" % (def_value))

# Optimize, using a SMAC-object
print("Optimizing! Depending on your machine, this might take a few minutes.")
smac = SMAC4AC(scenario=scenario, rng=np.random.RandomState(42), tae_runner=network_from_cfg)

incumbent = smac.optimize()

inc_value = network_from_cfg(incumbent)

print("Optimized Value: %.2f" % (inc_value))

# We can also validate our results (though this makes a lot more sense with instances)
smac.validate(
    config_mode='inc',  # We can choose which configurations to evaluate
    # instance_mode='train+test',  # Defines what instances to validate
    repetitions=100,  # Ignored, unless you set "deterministic" to "false" in line 95
    n_jobs=1
)  # How many cores to use in parallel for optimization
