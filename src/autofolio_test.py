import argparse
import os
import pickle
import sys
import warnings

import numpy as np
import pandas as pd
import torch

from AutoFolio.autofolio.autofolio import AutoFolio


sys.path.insert(0, os.path.abspath("./AutoFolio"))




warnings.simplefilter(action="ignore", category=FutureWarning)


class ProcessedDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, class_index):
        with open(data_path, "rb") as fh:
            self.dataset = torch.tensor(pickle.load(fh)).float()
            self.class_index = torch.tensor(class_index).float()

    def get_dataset(self):
        # for compatibility
        return self

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx], self.class_index


def load_datasets_processed(cfg, datasets, dataset_dir=None):
    """
    load preprocessed datasets from a list, return train/test datasets, dataset index and dataset name
    """
    if dataset_dir is None:
        dataset_dir = cfg["proc_dataset_dir"]
    dataset_list = []
    class_index = 0

    for dataset_name in datasets:
        dataset_train_path = os.path.join(dataset_dir, dataset_name + "_train")
        dataset_test_path = os.path.join(dataset_dir, dataset_name + "_test")

        try:
            dataset_train = ProcessedDataset(dataset_train_path, class_index)
            dataset_test = ProcessedDataset(dataset_test_path, class_index)
        except Exception as e:
            print(e)
            continue

        dataset_list.append((dataset_train, dataset_test, dataset_name, class_index))
        class_index += 1

    return dataset_list


def get_performance_data_and_export_csv(instances_lst, algorithms_lst, path_export_dir):
    if not os.path.isdir(path_export_dir):
        raise NotADirectoryError

    # simulate some data for test
    n_datasets = len(instances_lst)
    n_algorithms = len(algorithms_lst)
    performance_data = np.random.rand(n_datasets, n_algorithms)

    file_path = os.path.join(path_export_dir, "performance_data.csv")

    # TODO: perhaps replace pandas df by csv writer
    df = pd.DataFrame(columns=[algorithms_lst])
    for i, instance_label in enumerate(instances_lst):
        df.loc[instance_label] = performance_data[i]

    df.to_csv(file_path, float_format="%.5f")
    print("performance data dumped to: {}".format(file_path))
    return performance_data, file_path


def get_feature_data_and_export_csv(feature_source, path_export_dir, normalize_features=False):
    n_samples_to_use = feature_source["n_samples_to_use"]
    dataset_list = load_datasets_processed(feature_source, feature_source["datasets"])

    print("using data: {}".format(args.dataset))
    print("using n_samples: {}".format(args.n_samples))
    print("{} datasets loaded".format(len(dataset_list)))

    features = [
        ds[0].dataset[:n_samples_to_use].numpy() for ds in dataset_list
    ]  # 0 -> for now, only use train data for clustering
    features = np.concatenate(features, axis=0)

    if normalize_features:
        raise NotImplementedError  # TODO

    feature_dimension = features.shape[1]
    instance_labels = [ds[2] for ds in dataset_list]

    file_path = os.path.join(path_export_dir, "features_data.csv")

    # TODO: perhaps replace pandas df by csv writer
    df = pd.DataFrame(columns=[np.arange(feature_dimension)])
    for i, instance_label in enumerate(instance_labels):
        df.loc[instance_label] = features[i]

    df.to_csv(file_path, float_format="%.5f")
    print("feature data dumped to: {}".format(file_path))

    return features, file_path


def create_autofolio_function_call(
    arguments, perf_data_path, feature_data_path, budget=60, subprocess_call=True
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


parser = argparse.ArgumentParser("autofolio_test")
parser.add_argument(
    "--dataset", type=str, default="/home/ferreira/autodl_data/processed_datasets/1e4_meta"
)  # or '/home/ferreira/autodl_data/processed_datasets/1e4_combined'
parser.add_argument("--save_estimator", type=bool, default=True)  # stored under dataset
parser.add_argument(
    "--n_samples", type=int, default=1
)  # per dataset use one meta-feature (could also use multiple and compute average)
args = parser.parse_args()

info = {}

meta_features_path = args.dataset
info["proc_dataset_dir"] = meta_features_path
info["n_samples_to_use"] = args.n_samples

info["datasets"] = [
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
    "oxford_flowers102",
]

info["algorithms"] = ["algo1", "algo2", "algo3"]

performance_feature_dir_path = "./"
perf_data, perf_data_path = get_performance_data_and_export_csv(
    instances_lst=info["datasets"],
    algorithms_lst=info["algorithms"],
    path_export_dir=performance_feature_dir_path,
)
feature_data, feature_data_path = get_feature_data_and_export_csv(
    feature_source=info, path_export_dir=performance_feature_dir_path
)

args = {}
args["performance_csv"] = perf_data_path
args["feature_csv"] = feature_data_path
args["wallclock_limit"] = str(60)
args["maximize"] = True
af = AutoFolio()
af.run_cli(args)
