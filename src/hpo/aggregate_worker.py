import logging
import time

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
from hpbandster.core.worker import Worker


class AggregateWorker(Worker):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def compute(self, config_id, config, budget, working_directory, *args, **kwargs):
        time.sleep(5)  # Remove this
        return ({
            'loss': 1,  # remember: HpBandSter always minimizes!
            'info': {}
        })

    @staticmethod
    def get_configspace():
        cs = CS.ConfigurationSpace()
        lr = CSH.UniformFloatHyperparameter(
            'lr', lower=1e-6, upper=1e-1, default_value='1e-2', log=True
        )
        cs.add_hyperparameters([lr])
        return cs


if __name__ == "__main__":
    worker = AggregateWorker(run_id='0')
    cs = worker.get_configspace()

    config = cs.sample_configuration().get_dictionary()
    print(config)
    res = worker.compute(config=config, budget=1, working_directory='.')
    print(res)
