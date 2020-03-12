from __future__ import absolute_import

import argparse
import os
import sys
import warnings

import pandas as pd
from src.AutoFolio_old.autofolio.autofolio import AutoFolio
from src.available_datasets import all_datasets

sys.path.insert(0, os.path.abspath("src/AutoFolio_old"))
warnings.simplefilter(action="ignore", category=FutureWarning)


class AutoFolioPipeline(object):
    """
    A wrapper class for the execution of AutoFolio_old code.
    """

    def __init__(self, args):
        self.datasets = all_datasets
        self.wallclock_limit = args.wallclock_limit
        self.maximize = args.maximize
        self.autofolio_model_path = args.autofolio_model_path
        self.verbose = args.verbose

        self.perf_csv_path = args.perf_csv
        self.feat_csv_path = args.feat_csv

        self.autofolio = None  # AutoFolio_old instance

    def create_autofolio_function_call(
        self, arguments, perf_data_path, feature_data_path, budget=60, subprocess_call=True
    ):
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

    def train_and_save_autofolio_model(self):
        self.autofolio = AutoFolio()
        af_args = {
            "performance_csv": self.perf_csv_path,
            "feature_csv": self.feat_csv_path,
            "wallclock_limit": self.wallclock_limit,
            "maximize": self.maximize,
            "save": self.autofolio_model_path,
            "verbose": self.verbose,
        }

        self.autofolio.run_cli(af_args)  # 10 fold cv result

    def load_autofolio_model_and_predict(self, feature_vec):
        if self.autofolio is None:
            self.autofolio = AutoFolio()

        af_args = {
            "load": self.autofolio_model_path,  # path to the AF model file
            "verbose": self.verbose,
            "feature_vec": feature_vec
        }

        selected_schedule = self.autofolio.run_cli(af_args)  # yields [(algorithm, budget)]
        return selected_schedule


if __name__ == "__main__":
    """ Pipeline arguments"""
    parser = argparse.ArgumentParser("AutoFolioPipeline")

    parser.add_argument(
        "--perf_csv",
        type=str,
        default="experiments/"
        "kakaobrain_optimized_per_dataset_datasets_x_configs_evaluations/"
        "perf_matrix_train.csv"
    )

    parser.add_argument("--feat_csv", type=str, default="src/meta_features/meta_features_train.csv")
    """ AutoFolio_old arguments """
    parser.add_argument("--wallclock_limit", type=str, default=str(1800))
    parser.add_argument("--maximize", type=bool, default=True)
    parser.add_argument("--autofolio_model_path", type=str, default="submission/af_model")
    parser.add_argument("--verbose", type=str, default="INFO")  # ERROR, WARNING etc.
    parser.add_argument("--tune", type=bool, default=True)
    parser.add_argument("--runcount_limit", type=str, default=str(3000000))

    df = pd.read_pickle("src/meta_features/meta_features_valid.pkl")
    feature_vec = df.iloc[0].to_list()
    """ reformatting ndarray to str since AutoFolio_old's read_model_and_predict requires
    a string of floats separated by space """
    feature_vec_str = ""
    for feature in feature_vec:
        feature_vec_str += " {}".format(feature)
    feature_vec_str = feature_vec_str[1:]  # remove first unwanted whitespace character

    args = parser.parse_args()
    autofolio_pipeline = AutoFolioPipeline(args=args)

    autofolio_pipeline.train_and_save_autofolio_model()
    autofolio_pipeline.load_autofolio_model_and_predict(feature_vec=feature_vec_str)
