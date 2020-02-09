import logging
import time
from copy import deepcopy
from pathlib import Path

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
import yaml
from hpbandster.core.worker import Worker
from src.available_datasets import train_datasets
from src.competition.run_local_test import run_baseline as evaluate_on_dataset


class AggregateWorker(Worker):
    def __init__(self, working_directory, **kwargs):
        super().__init__(**kwargs)

        with Path("src/configs/default.yaml").open() as in_stream:
            self._default_config = yaml.safe_load(in_stream)

        self._datasets_dir = self._default_config["cluster__datasets_dir"]
        self._working_directory = working_directory

    def _construct_model_config(self, config):
        mc = deepcopy(self._default_config)

        # yapf: disable
        mc["autocv"]["dataset"]["cv_valid_ratio"] = config["cv_valid_ratio"]
        mc["autocv"]["dataset"]["max_valid_count"] = config["max_valid_count"]
        mc["autocv"]["dataset"]["log2_max_size"] = 2 ** config["log2_max_size"]
        mc["autocv"]["dataset"]["max_times"] = config["max_times"]
        mc["autocv"]["dataset"]["train_info_sample"] = config["train_info_sample"]
        mc["autocv"]["dataset"]["enough_count"]["image"] = config["enough_count_image"]
        mc["autocv"]["dataset"]["enough_count"]["video"] = config["enough_count_video"]

        mc["autocv"]["dataset"]["steps_per_epoch"] = config["steps_per_epoch"]
        mc["autocv"]["conditions"]["early_epoch"] = config["early_epoch"]
        mc["autocv"]["conditions"]["skip_valid_score_threshold"] = config["skip_valid_score_threshold"]
        mc["autocv"]["conditions"]["test_after_at_least_seconds"] = config["test_after_at_least_seconds"]
        mc["autocv"]["conditions"]["test_after_at_least_seconds_max"] = config["test_after_at_least_seconds_max"]
        mc["autocv"]["conditions"]["test_after_at_least_seconds_step"] = config["test_after_at_least_seconds_step"]
        mc["autocv"]["conditions"]["threshold_valid_score_diff"] = config["threshold_valid_score_diff"]
        mc["autocv"]["conditions"]["max_inner_loop_ratio"] = config["max_inner_loop_ratio"]

        mc["autocv"]["optimizer"]["lr"] = config["lr"]
        mc["autocv"]["optimizer"]["min_lr"] = config["min_lr"]
        mc["autocv"]["dataset"]["batch_size"] = config["batch_size"]
        # yapf: enable

        return mc

    def _run_on_dataset(self, dataset, config_experiment_path, model_config):
        experiment_path = config_experiment_path / dataset
        dataset_path = Path(self._datasets_dir, dataset)

        dataset_score = evaluate_on_dataset(
            dataset_dir=str(dataset_path),
            code_dir="src",
            experiment_dir=str(experiment_path),
            time_budget=60,  # seconds, TODO: this changes behavior
            overwrite=True,
            model_config_name=None,
            model_config=model_config
        )
        return dataset_score

    def compute(self, config_id, config, budget, *args, **kwargs):
        config_id_formated = "_".join(map(str, config_id))
        config_experiment_path = Path(self._working_directory, config_id_formated, str(budget))

        model_config = self._construct_model_config(config)
        scores = [
            self._run_on_dataset(dataset, config_experiment_path, model_config)
            for dataset in train_datasets
        ]
        mean_score = sum(scores) / len(scores)  # TODO: use improvement scores + is mean correct?

        return ({
            'loss': -mean_score,  # remember: HpBandSter always minimizes!
            'info': {}
        })

    @staticmethod
    def get_configspace():
        cs = CS.ConfigurationSpace()

        # yapf: disable
        # Dataset sizes
        cv_valid_ratio = CSH.UniformFloatHyperparameter(
            "cv_valid_ratio", lower=0.05, upper=0.2)
        max_valid_count = CSH.UniformIntegerHyperparameter(
            "max_valid_count", lower=128, upper=512, log=True)
        max_size = CSH.UniformIntegerHyperparameter(
            "log2_max_size", lower=5, upper=7)
        max_times = CSH.UniformIntegerHyperparameter(
            "max_times", lower=4, upper=10)
        train_info_sample = CSH.UniformIntegerHyperparameter(
            "train_info_sample", lower=128, upper=512, log=True)
        enough_count_video = CSH.UniformIntegerHyperparameter(
            "enough_count_video", lower=100, upper=10000, log=True)
        enough_count_image = CSH.UniformIntegerHyperparameter(
            "enough_count_image", lower=1000, upper=100000, log=True)

        # Report intervalls
        steps_per_epoch = CSH.UniformIntegerHyperparameter(
            "steps_per_epoch", lower=5, upper=250, log=True)
        early_epoch = CSH.UniformIntegerHyperparameter(
            "early_epoch", lower=1, upper=3)
        skip_valid_score_threshold = CSH.UniformFloatHyperparameter(
            "skip_valid_score_threshold", lower=0.7, upper=0.95)
        test_after_at_least_seconds = CSH.UniformIntegerHyperparameter(
            "test_after_at_least_seconds", lower=1, upper=3)
        test_after_at_least_seconds_max = CSH.UniformIntegerHyperparameter(
            "test_after_at_least_seconds_max", lower=60, upper=120)
        test_after_at_least_seconds_step = CSH.UniformIntegerHyperparameter(
            "test_after_at_least_seconds_step", lower=2, upper=10)
        threshold_valid_score_diff = CSH.UniformFloatHyperparameter(
            "threshold_valid_score_diff", lower=0.0001, upper=0.01, log=True)
        max_inner_loop_ratio = CSH.UniformFloatHyperparameter(
            "max_inner_loop_ratio", lower=0.1, upper=0.3)

        # Optimization
        batch_size = CSH.UniformIntegerHyperparameter(
            "batch_size", lower=16, upper=64, log=True)
        lr = CSH.UniformFloatHyperparameter(
            'lr', lower=1e-5, upper=1e-1, log=True)
        min_lr = CSH.UniformFloatHyperparameter(
            'min_lr', lower=1e-8, upper=1e-5, log=True)
        # yapf: enable

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
    worker = AggregateWorker(working_directory='experiments/test/test_aggregate_worker', run_id='0')
    cs = worker.get_configspace()

    config = cs.sample_configuration().get_dictionary()
    print(config)
    res = worker.compute(config=config, config_id=(0, 0, 0), budget=1)
    print(res)
