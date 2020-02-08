from __future__ import absolute_import
import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd
import time

from src.utils import load_datasets_processed
sys.path.insert(0, os.path.abspath("./AutoFolio"))
#from src.AutoFolio.autofolio import AutoFolio
from src.AutoFolio.autofolio.autofolio import AutoFolio

warnings.simplefilter(action="ignore", category=FutureWarning)


class AutoFolioPipeline(object):
    """
    A wrapper class for the execution of AutoFolio code.
    """
    def __init__(self, args):
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

        self.wallclock_limit = args.wallclock_limit
        self.maximize = args.maximize
        self.autofolio_model_path = args.autofolio_model_path
        self.verbose = args.verbose

        self.algorithms = ["algo1", "algo2", "algo3"]
        self.n_samples_to_use = args.n_samples
        self.n_datasets = len(self.datasets)
        self.n_algorithms = len(self.algorithms)

        self.performance_data_file_path = None
        self.performance_data = None

        self.dataset_path = args.dataset_path

        self.feature_data_file_path = None
        self.train_feature_data = None
        self.test_feature_data = None
        self.processed_datasets = None

        self.autofolio = None  # AutoFolio instance

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
        info = {"proc_dataset_dir": self.dataset_path, "datasets": self.datasets}
        self.processed_datasets = load_datasets_processed(info, info["datasets"])
        return self.processed_datasets

    def load_features(self, normalize_features=False):
        self.load_processed_datasets()
        print("getting features ...")
        print("using data: {}".format(self.dataset_path))
        print("using n_samples: {}".format(self.n_samples_to_use))
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

        feature_dimension = features.shape[1]
        instance_labels = [ds[2] for ds in self.processed_datasets]

        self.feature_data_file_path = os.path.join(path_export_dir, csv_file_name)

        # TODO: perhaps replace pandas df by csv writer
        df = pd.DataFrame(columns=[np.arange(feature_dimension)])
        for i, instance_label in enumerate(instance_labels):
            df.loc[instance_label] = features[i]

        df.to_csv(self.feature_data_file_path, float_format="%.5f")
        print("feature data dumped to: {}".format(self.feature_data_file_path))

    def train_and_save_autofolio_model(self):
        self.autofolio = AutoFolio()
        af_args = {"performance_csv": self.performance_data_file_path,
                   "feature_csv": self.feature_data_file_path,
                   "wallclock_limit": self.wallclock_limit,
                   "maximize": self.maximize,
                   "save": self.autofolio_model_path,
                   "verbose": self.verbose,
                   }

        self.autofolio.run_cli(af_args)  # 10 fold cv result

    def load_autofolio_model_and_predict(self):
        if self.autofolio is None:
            self.autofolio = AutoFolio()

        """ pick one random sample from the test data for autofolio prediction """
        rnd = np.random.randint(low=0, high=np.shape(self.test_feature_data)[0]) # 0-> index for the datasets
        feature_vec = self.test_feature_data[rnd]

        """ reformatting ndarray to str since AutoFolio's read_model_and_predict requires
        a string of floats separated by space """
        feature_vec_str = ""
        for feature in feature_vec:
            feature_vec_str += " {}".format(feature)
        feature_vec_str = feature_vec_str[1:]  # remove first unwanted whitespace character

        af_args = {
                    "load":  self.autofolio_model_path,  # path to the AF model file
                    #"feature_csv": self.feature_data_file_path,  # --> does not work here
                    "verbose": self.verbose,
                    "feature_vec": feature_vec_str
                   }

        selected_schedule = self.autofolio.run_cli(af_args)  # yields [(algorithm, budget)]
        return selected_schedule


if __name__ == "__main__":
    """ Pipeline arguments"""
    parser = argparse.ArgumentParser("AutoFolioPipeline")
    # or '/home/ferreira/autodl_data/processed_datasets/1e4_meta'
    parser.add_argument("--dataset_path", type=str, default="/home/ferreira/autodl_data/processed_datasets/1e4_combined")
    parser.add_argument("--save_estimator", type=bool, default=True)  # stored under dataset
    parser.add_argument("--n_samples", type=int, default=1)  # per dataset use one meta-feature

    """ AutoFolio arguments """
    parser.add_argument("--wallclock_limit", type=str, default=str(60))  # per dataset use one meta-feature
    parser.add_argument("--maximize", type=bool, default=True)  # per dataset use one meta-feature
    parser.add_argument("--autofolio_model_path", type=str, default="/home/ferreira/autodl_data/autofolio_models/first_submission/model")
    parser.add_argument("--verbose", type=str, default="INFO")  # ERROR, WARNING etc.

    AutoFolioPipelineArgs = parser.parse_args()
    autofolio_pipeline = AutoFolioPipeline(args=AutoFolioPipelineArgs)

    performance_feature_dir_path = "./"
    autofolio_pipeline.load_performance_data_and_export_csv(path_export_dir=performance_feature_dir_path)
    autofolio_pipeline.load_features_and_export_csv(path_export_dir=performance_feature_dir_path, normalize_features=False)

    #autofolio_pipeline.train_and_save_autofolio_model()
    autofolio_pipeline.load_autofolio_model_and_predict()

