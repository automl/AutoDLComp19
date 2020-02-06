import argparse
import os
import pickle
import sys
import warnings

import numpy as np
import pandas as pd
import torch

from src.utils import ProcessedDataset

# fmt: off
sys.path.insert(0, os.path.abspath("./AutoFolio"))
from AutoFolio.autofolio.autofolio import AutoFolio  # noqa isort:skip

# fmt: on

warnings.simplefilter(action="ignore", category=FutureWarning)


class AutoFolioPipeline(object):
    """
    A wrapper class for the execution of AutoFolio code.
    """
    def __init__(self, args):
        self.args = args

        self.datasets = [
            "Chucky",
            "Decal",
            "Hammer",
            "Hmdb51",
            "Katze",
            "Kreatur",
            "Munster",
            "Pedro",
            "SMv2",
            "Ucf101",
            "binary_alpha_digits",
            "caltech101",
            "caltech_birds2010",
            "caltech_birds2011",
            "cats_vs_dogs",
            "cifar100",
            "cifar10",
            "coil100",
            "colorectal_histology",
            "deep_weeds",
            "emnist",
            "eurosat",
            "fashion_mnist",
            "horses_or_humans",
            "kmnist",
            "mnist",
            "oxford_flowers102"
        ]
        self.algorithms = ["algo1", "algo2", "algo3"]
        self.n_samples_to_use = self.args.n_samples
        self.n_datasets = len(self.datasets)
        self.n_algorithms = len(self.algorithms)

        self.performance_data_file_path = None
        self.performance_data = None

        self.feature_data_file_path = None
        self.train_feature_data = None
        self.test_feature_data = None
        self.processed_datasets = None

        self.af = None  # AutoFolio instance

    def create_autofolio_function_call(self, arguments, perf_data_path, feature_data_path, budget=60, subprocess_call=True):
        if not subprocess_call:
            function_call = os.path.join(arguments["autofolio_dir"], "scripts", "autofolio")
            function_call += " --performance_csv=" + perf_data_path
            function_call += " --feature_csv=" + feature_data_path
            function_call += " --wallclock_limit=" + str(budget)
            function_call += "--maximize"
            return function_call
        else:
            autofolio_program = os.path.join(arguments["autofolio_dir"], "scripts", "autofolio")
            return [
                autofolio_program,
                "--performance_csv",
                perf_data_path,
                "--feature_csv",
                feature_data_path,
                "wallclock_limit",
                str(budget),
                "--maximize",
            ]

    def load_performance_data(self, path_export_dir):
        if not os.path.isdir(path_export_dir):
            raise NotADirectoryError

        # simulate some data for test
        self.performance_data = np.random.rand(self.n_datasets, self.n_algorithms)
        self.performance_data_file_path = os.path.join(path_export_dir, "performance_data.csv")
        return self.performance_data

    def load_performance_data_and_export_csv(self, path_export_dir):
        self.load_performance_data(path_export_dir=path_export_dir)

        # TODO: perhaps replace pandas df by csv writer for speed improvements
        df = pd.DataFrame(columns=[self.algorithms])
        for i, instance_label in enumerate(self.datasets):
            df.loc[instance_label] = self.performance_data[i]

        df.to_csv(self.performance_data_file_path, float_format="%.5f")
        print("performance data dumped to: {}".format(self.performance_data_file_path))
        return self.performance_data, self.performance_data_file_path

    def load_processed_datasets(self):
        info = {"proc_dataset_dir": self.args.dataset, "datasets": self.datasets}
        self.processed_datasets = ProcessedDataset.load_datasets_processed(info, info["datasets"])
        return self.processed_datasets

    def load_features(self, normalize_features=False):
        self.load_processed_datasets()
        print("getting features ...")
        print("using data: {}".format(self.args.dataset))
        print("using n_samples: {}".format(self.args.n_samples))
        print("{} datasets loaded".format(self.n_datasets))

        print("loading data with one sample per dataset")
        self.train_feature_data = [ds[0].dataset[:self.n_samples_to_use].numpy() for ds in self.processed_datasets]
        self.train_feature_data = np.concatenate(self.train_feature_data, axis=0)

        self.test_feature_data = [ds[1].dataset[:self.n_samples_to_use].numpy() for ds in self.processed_datasets]
        self.test_feature_data = np.concatenate(self.test_feature_data, axis=0)

        # TODO
        if normalize_features:
            raise NotImplementedError

        return self.train_feature_data, self.test_feature_data

    def load_features_and_export_csv(self, path_export_dir, normalize_features=False, use_train_data=True):
        self.load_features(normalize_features=normalize_features)

        if use_train_data:
            features = self.train_feature_data
            csv_file_name = "features_train_data.csv"
        else:
            features = self.test_feature_data
            csv_file_name = "features_test_data.csv"

        feature_dimension =features.shape[1]
        instance_labels = [ds[2] for ds in self.processed_datasets]

        self.feature_data_file_path = os.path.join(path_export_dir, csv_file_name)

        # TODO: perhaps replace pandas df by csv writer
        df = pd.DataFrame(columns=[np.arange(feature_dimension)])
        for i, instance_label in enumerate(instance_labels):
            df.loc[instance_label] = features[i]

        df.to_csv(self.feature_data_file_path, float_format="%.5f")
        print("feature data dumped to: {}".format(self.feature_data_file_path))

    def run_autofolio(self):
        self.af = AutoFolio()
        af_args = {"performance_csv": self.performance_data_file_path, "feature_csv": self.feature_data_file_path,
                   "wallclock_limit": self.args.wallclock_limit, "maximize": self.args.maximize}

        self.af.run_cli(af_args)


if __name__ == "__main__":
    """ Pipeline arguments"""
    parser = argparse.ArgumentParser("AutoFolioPipeline")
    # or '/home/ferreira/autodl_data/processed_datasets/1e4_combined'
    parser.add_argument("--dataset", type=str, default="/home/ferreira/autodl_data/processed_datasets/1e4_meta")
    parser.add_argument("--save_estimator", type=bool, default=True)  # stored under dataset
    parser.add_argument("--n_samples", type=int, default=1)  # per dataset use one meta-feature

    """ AutoFolio arguments """
    parser.add_argument("--wallclock_limit", type=str, default=str(60))  # per dataset use one meta-feature
    parser.add_argument("--maximize", type=bool, default=True)  # per dataset use one meta-feature

    AutoFolioPipelineArgs = parser.parse_args()

    afp = AutoFolioPipeline(args=AutoFolioPipelineArgs)

    performance_feature_dir_path = "./"
    afp.load_performance_data_and_export_csv(path_export_dir=performance_feature_dir_path)
    afp.load_features_and_export_csv(path_export_dir=performance_feature_dir_path, normalize_features=False)

    afp.run_autofolio()

