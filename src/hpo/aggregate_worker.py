import random
from pathlib import Path
from time import sleep

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
import numpy as np
import tensorflow as tf
import torch
import yaml
from hpbandster.core.worker import Worker
from src.available_datasets import train_datasets
from src.competition.run_local_test import run_baseline as evaluate_on_dataset
from src.hpo.utils import construct_model_config


def _run_on_dataset(
    dataset, config_experiment_path, model_config, dataset_dir, n_repeat, time_budget,
    time_budget_approx
):
    experiment_path = config_experiment_path
    dataset_path = Path(dataset_dir, dataset)

    repetition_scores = []
    for _ in range(n_repeat):
        score = evaluate_on_dataset(
            dataset_dir=str(dataset_path),
            code_dir="src",
            experiment_dir=str(experiment_path),
            time_budget=time_budget,
            time_budget_approx=time_budget_approx,
            overwrite=True,
            model_config_name=None,
            model_config=model_config
        )
        repetition_scores.append(score)

    repetition_mean = np.mean(repetition_scores)
    return repetition_scores, repetition_mean


def get_configspace():
    cs = CS.ConfigurationSpace()

    # yapf: disable
    # Dataset sizes
    cv_valid_ratio = CSH.UniformFloatHyperparameter("cv_valid_ratio", lower=0.05, upper=0.2)
    max_valid_count = CSH.UniformIntegerHyperparameter("max_valid_count", lower=128, upper=512,
                                                       log=True)
    max_size = CSH.UniformIntegerHyperparameter("log2_max_size", lower=5, upper=7)
    max_times = CSH.UniformIntegerHyperparameter("max_times", lower=4, upper=10)
    train_info_sample = CSH.UniformIntegerHyperparameter("train_info_sample", lower=128, upper=512,
                                                         log=True)
    enough_count_video = CSH.UniformIntegerHyperparameter("enough_count_video", lower=100,
                                                          upper=1000, log=True)
    enough_count_image = CSH.UniformIntegerHyperparameter("enough_count_image", lower=1000,
                                                          upper=5000, log=True)

    # Report intervalls
    steps_per_epoch = CSH.UniformIntegerHyperparameter("steps_per_epoch", lower=5, upper=250,
                                                       log=True)
    early_epoch = CSH.UniformIntegerHyperparameter("early_epoch", lower=1, upper=3)
    skip_valid_score_threshold = CSH.UniformFloatHyperparameter("skip_valid_score_threshold",
                                                                lower=0.7, upper=0.95)
    test_after_at_least_seconds = CSH.UniformIntegerHyperparameter("test_after_at_least_seconds",
                                                                   lower=1, upper=3)
    test_after_at_least_seconds_max = CSH.UniformIntegerHyperparameter(
        "test_after_at_least_seconds_max", lower=60, upper=120)
    test_after_at_least_seconds_step = CSH.UniformIntegerHyperparameter(
        "test_after_at_least_seconds_step", lower=2, upper=10)
    threshold_valid_score_diff = CSH.UniformFloatHyperparameter("threshold_valid_score_diff",
                                                                lower=0.0001, upper=0.01, log=True)
    max_inner_loop_ratio = CSH.UniformFloatHyperparameter("max_inner_loop_ratio", lower=0.1,
                                                          upper=0.3)

    # Optimization
    batch_size = CSH.UniformIntegerHyperparameter("batch_size", lower=16, upper=64, log=True)
    lr = CSH.UniformFloatHyperparameter('lr', lower=1e-5, upper=1e-1, log=True)
    min_lr = CSH.UniformFloatHyperparameter('min_lr', lower=1e-8, upper=1e-5, log=True)
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


class SingleWorker(Worker):
    def __init__(
        self, working_directory, n_repeat, dataset, time_budget, time_budget_approx, **kwargs
    ):
        super().__init__(**kwargs)

        with Path("src/configs/default.yaml").open() as in_stream:
            self._default_config = yaml.safe_load(in_stream)

        self._dataset_dir = self._default_config["cluster_datasets_dir"]
        self._working_directory = working_directory
        self.n_repeat = n_repeat
        self.dataset = dataset
        self.time_budget = time_budget
        self.time_budget_approx = time_budget_approx

    def compute(self, config_id, config, budget, *args, **kwargs):
        config_id_formated = "_".join(map(str, config_id))
        config_experiment_path = Path(self._working_directory, config_id_formated, str(budget))

        model_config = construct_model_config(config, default_config=self._default_config)
        repetition_scores, repetition_mean = _run_on_dataset(
            self.dataset,
            config_experiment_path,
            model_config,
            dataset_dir=self._dataset_dir,
            n_repeat=self.n_repeat,
            time_budget=self.time_budget,
            time_budget_approx=self.time_budget_approx
        )

        info = {
            "{}_repetition_{}_score".format(self.dataset, n): score
            for n, score in enumerate(repetition_scores)
        }

        return ({
            'loss': -repetition_mean,  # remember: HpBandSter always minimizes!
            'info': info
        })


class AggregateWorker(Worker):
    def __init__(self, working_directory, n_repeat, time_budget, time_budget_approx, **kwargs):
        super().__init__(**kwargs)

        with Path("src/configs/default.yaml").open() as in_stream:
            self._default_config = yaml.safe_load(in_stream)

        self._dataset_dir = self._default_config["cluster_datasets_dir"]
        self._working_directory = working_directory
        self.n_repeat = n_repeat
        self.time_budget = time_budget
        self.time_budget_approx = time_budget_approx

    def compute(self, config_id, config, budget, *args, **kwargs):
        config_id_formated = "_".join(map(str, config_id))
        config_experiment_path = Path(self._working_directory, config_id_formated, str(budget))

        model_config = construct_model_config(config, default_config=self._default_config)

        score_results_tuples = []
        for dataset in train_datasets:
            try:
                score = _run_on_dataset(
                    dataset,
                    config_experiment_path / dataset,
                    model_config,
                    dataset_dir=self._dataset_dir,
                    n_repeat=self.n_repeat,
                    time_budget=budget,
                    time_budget_approx=self.time_budget_approx
                )
            except RuntimeError:
                score = 0
            score_results_tuples.append(score)

        # just get the repetition means for optimization
        repetition_scores_mean_per_dataset = np.array(score_results_tuples)[:, 1]
        # TODO: use improvement scores + is mean correct?
        scores_mean_over_datasets = np.mean(repetition_scores_mean_per_dataset)

        # for the report, just use the repetition means
        info = {
            "{}_repetition_{}_score".format(dataset, n): score
            for dataset in train_datasets
            for n, (repetition_scores, repetition_mean) in enumerate(score_results_tuples)
            for score in repetition_scores
        }

        return (
            {
                'loss': -scores_mean_over_datasets,  # remember: HpBandSter always minimizes!
                'info': info
            }
        )


if __name__ == "__main__":
    random.seed(2)
    np.random.seed(2)
    torch.manual_seed(2)
    torch.cuda.manual_seed_all(2)
    tf.set_random_seed(2)

    worker = AggregateWorker(
        working_directory='experiments/test/test_aggregate_worker',
        n_repeat=1,
        run_id='0',
        time_budget=1200,
        time_budget_approx=60
    )

    # worker = SingleWorker(
    #     working_directory='experiments/test/test_aggregate_worker',
    #     n_repeat=1,
    #     run_id='0',
    #     dataset="emnist",
    #     time_budget=1200,
    #     time_budget_approx=60
    # )
    cs = get_configspace()

    config = cs.sample_configuration().get_dictionary()
    print(config)

    res = worker.compute(config=config, config_id=(0, 0, 0), budget=1)
    print(res)
