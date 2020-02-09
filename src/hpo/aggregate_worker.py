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

        # fmt: off
        # Dataset sizes
        cv_valid_ratio = CSH.UniformFloatHyperparameter("cv_valid_ratio", lower=0.05, upper=0.2)
        max_valid_count = CSH.UniformIntegerHyperparameter(
            "max_valid_count", lower=128, upper=512, log=True
        )
        max_size = CSH.UniformIntegerHyperparameter("log2_max_size", lower=5, upper=7)
        max_times = CSH.UniformIntegerHyperparameter("max_times", lower=4, upper=10)
        train_info_sample = CSH.UniformIntegerHyperparameter(
            "train_info_sample", lower=128, upper=512, log=True
        )
        enough_count_video = CSH.UniformIntegerHyperparameter(
            "enough_count_video", lower=100, upper=10000, log=True
        )
        enough_count_image = CSH.UniformIntegerHyperparameter(
            "enough_count_image", lower=1000, upper=100000, log=True
        )

        # Report intervalls
        steps_per_epoch = CSH.UniformIntegerHyperparameter(
            "steps_per_epoch", lower=5, upper=250, log=True
        )
        early_epoch = CSH.UniformIntegerHyperparameter("early_epoch", lower=1, upper=3)
        skip_valid_score_threshold = CSH.UniformFloatHyperparameter(
            "skip_valid_score_threshold", lower=0.7, upper=0.95
        )
        test_after_at_least_seconds = CSH.UniformIntegerHyperparameter(
            "test_after_at_least_seconds", lower=1, upper=3
        )
        test_after_at_least_seconds_max = CSH.UniformIntegerHyperparameter(
            "test_after_at_least_seconds_max", lower=60, upper=120
        )
        test_after_at_least_seconds_step = CSH.UniformIntegerHyperparameter(
            "test_after_at_least_seconds_step", lower=2, upper=10
        )
        threshold_valid_score_diff = CSH.UniformIntegerHyperparameter(
            "threshold_valid_score_diff", lower=0.0001, upper=0.01, log=True
        )
        max_inner_loop_ratio = CSH.UniformFloatHyperparameter(
            "max_inner_loop_ratio", lower=0.1, upper=0.3
        )

        # Optimization
        batch_size = CSH.UniformIntegerHyperparameter("batch_size", lower=16, upper=64, log=True)
        lr = CSH.UniformFloatHyperparameter('lr', lower=1e-5, upper=1e-1, log=True)
        min_lr = CSH.UniformFloatHyperparameter('min_lr', lower=1e-8, upper=1e-5, log=True)
        # fmt: on

        cs.add_hyperparameters(
            [
                cv_valid_ratio, max_valid_count, max_size, max_times, train_info_sample,
                enough_count_video, enough_count_image, steps_per_epoch, early_epoch,
                skip_valid_score_threshold, test_after_at_least_seconds,
                test_after_at_least_seconds_max, test_after_at_least_seconds_step,
                threshold_valid_score_diff, max_inner_loop_ratio, batch_size, lr, min_lr
            ]
        )
        return cs


if __name__ == "__main__":
    worker = AggregateWorker(run_id='0')
    cs = worker.get_configspace()

    config = cs.sample_configuration().get_dictionary()
    print(config)
    res = worker.compute(config=config, budget=1, working_directory='.')
    print(res)
