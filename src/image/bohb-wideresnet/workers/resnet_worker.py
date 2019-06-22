import pickle

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
import os
from hpbandster.core.worker import Worker

from workers.lib.wrn.train import run_training


# from copy import copy, deepcopy


class WideResnetWorker(Worker):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute(self, config, budget, config_id, working_directory):
        dest_dir = os.path.join(working_directory, "_".join(map(str, config_id)))
        os.makedirs(dest_dir)
        return run_training(epochs=budget, learning_rate=config['learning_rate'], alpha=config['alpha'],
                            weight_decay=config['weight_decay'], save_dir=dest_dir, run=config_id)

    @staticmethod
    def get_config_space():
        config_space = CS.ConfigurationSpace()

        # learning rate
        config_space.add_hyperparameter(
            CSH.UniformFloatHyperparameter('learning_rate', lower=1e-2, upper=1, log=True))

        # weight decay
        config_space.add_hyperparameter(
            CSH.UniformFloatHyperparameter('weight_decay', lower=1e-5, upper=1e-3, log=True))

        # mixup interpolation factor
        config_space.add_hyperparameter(
            CSH.UniformFloatHyperparameter('alpha', lower=0, upper=1))

        return config_space
