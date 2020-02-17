import yaml
import os
import pandas as pd
import numpy as np
import collections

from src.available_datasets import all_datasets, train_datasets, val_datasets
from pathlib import Path
from src.competition.ingestion_program.dataset import AutoDLDataset
from src.utils import load_datasets_processed

def convert_metadata_to_df(metadata):
    k, v = list(metadata.items())[0]
    columns = sorted(v.keys())
    columns_edited = False

    features_lists = []
    indices = []

    for key, values_dict in sorted(metadata.items()):
        indices.append(key)
        feature_list = [values_dict[k] for k in sorted(values_dict.keys())]

        # below loop flattens feature list since there are tuples in it &
        # it extends columns list accordingly
        for i, element in enumerate(feature_list):
            if type(element) is tuple:
                # convert tuple to single list elements
                slce = slice(i, i+len(element)-1)

                feature_list[slce] = list(element)

                if not columns_edited:
                    columns_that_are_tuples = columns[i]
                    new_columns = [columns_that_are_tuples + "_" + str(i) for i in range(len(element))]
                    columns[slce] = new_columns
                    columns_edited = True

        features_lists.append(feature_list)

    return pd.DataFrame(features_lists, columns=columns, index=indices)


def parse_meta_features(meta_data):
    sequence_size, x, y, num_channels = meta_data.get_tensor_shape()
    num_classes = meta_data.get_output_size()
    return dict(
        num_classes=num_classes,
        sequence_size=sequence_size,
        resolution=(x, y),
        num_channels=num_channels
    )


def load_processed_datasets(dataset_path):
    info = {"proc_dataset_dir": dataset_path, "datasets": all_datasets}
    processed_datasets = load_datasets_processed(info, info["datasets"])
    return processed_datasets


def precompute_nn_meta_features(dataset_path, normalize_features=False):
    processed_datasets = load_processed_datasets(dataset_path=dataset_path)
    print("getting features ...")
    print("using data: {}".format(dataset_path))

    train_feature_data = [ds[0].dataset.numpy()
                          for ds in processed_datasets]
    train_feature_data = np.concatenate(train_feature_data, axis=0)

    test_feature_data = [ds[1].dataset.numpy()
                         for ds in processed_datasets]
    test_feature_data = np.concatenate(test_feature_data, axis=0)

    # TODO
    if normalize_features:
        raise NotImplementedError

    return train_feature_data, test_feature_data, processed_datasets


def export_nn_meta_features(dataset_path, path_export_dir, normalize_features=False, use_train_data=True):
    train_feature_data, test_feature_data, processed_datasets = \
        precompute_nn_meta_features(dataset_path=dataset_path, normalize_features=normalize_features)

    if use_train_data:
        features = train_feature_data
        csv_file_name = "features_train_data.csv"
    else:
        features = test_feature_data
        csv_file_name = "features_test_data.csv"

    feature_dimension = features.shape[1]
    instance_labels = [ds[2] for ds in processed_datasets]

    feature_data_file_path = os.path.join(path_export_dir, csv_file_name)

    df = pd.DataFrame(columns=[np.arange(feature_dimension)])
    for i, instance_label in enumerate(instance_labels):
        df.loc[instance_label] = features[i]

    df.to_csv(feature_data_file_path, float_format="%.5f")
    print("feature data dumped to: {}".format(feature_data_file_path))


def dump_meta_features_df_and_csv(meta_features, output_path, split_df=True):
    df = convert_metadata_to_df(meta_features)
    if split_df:
        df_train = df.loc[df.index.isin(train_datasets)]
        df_valid = df.loc[df.index.isin(val_datasets)]

        df_train.to_csv(output_path.parent / "meta_features_train.csv")
        df_train.to_pickle(output_path.parent / "meta_features_train.pkl")

        df_valid.to_csv(output_path.parent / "meta_features_valid.csv")
        df_valid.to_pickle(output_path.parent / "meta_features_valid.pkl")

    else:
        df.to_csv(output_path.with_suffix(".csv"))
        df.to_pickle(output_path.with_suffix(".pkl"))

    print("meta features data dumped to: {}".format(output_path.parent))


def get_meta_features_from_dataset(dataset_path):
    full_dataset_path = dataset_path / "{}.data/train".format(dataset_path.name)
    if not full_dataset_path.exists():
        full_dataset_path = dataset_path / "{}.data/train".format((dataset_path.name).lower())

    train_dataset = AutoDLDataset(str(full_dataset_path))
    meta_data = train_dataset.get_metadata()
    return parse_meta_features(meta_data)


def precompute_meta_features(dataset_path, output_path, dump_dataframe_csv=True, split_df=True):
    dataset_to_meta_features = {
        dataset.name: get_meta_features_from_dataset(dataset)
        for dataset in dataset_path.iterdir() if dataset.name in all_datasets
    }

    if dump_dataframe_csv:
        dump_meta_features_df_and_csv(
            meta_features=dataset_to_meta_features,
            output_path=output_path,
            split_df=split_df
        )

    with output_path.open("w") as out_stream:
        yaml.dump(dataset_to_meta_features, out_stream)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        "--dataset_path", default="/data/aad/image_datasets/all_symlinks/", type=Path, help=" "
    )
    parser.add_argument(
        "--output_path", default="src/meta_features/meta_features.yaml", type=Path, help=" "
    )

    args = parser.parse_args()

    precompute_meta_features(
         args.dataset_path,
         args.output_path,
         dump_dataframe_csv=True,
         split_df=False
    )
