import yaml
import random
import pandas as pd
import numpy as np

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


def precompute_nn_meta_features(processed_datasets_path,
                                output_path,
                                dataset_path,
                                dump_dataframe_csv=True,
                                split_df=True,
                                n_samples_to_use=100,
                                file_name="meta_features"):

    processed_datasets = load_processed_datasets(dataset_path=str(processed_datasets_path))

    print("getting features ...")
    print("using data: {}".format(processed_datasets_path))

    # we need to get num_channels and num_classes from the original datasets
    dataset_to_meta_features = {dataset.name: get_meta_features_from_dataset(dataset)
                                for dataset in dataset_path.iterdir()
                                if dataset.name in all_datasets}

    # as before, we're for now only using the train ([0]) data
    dataset_to_nn_meta_features = {}
    for dataset in processed_datasets:
        dataset_name = dataset[2]
        data = dataset[0].dataset.numpy()
        sample_indices = np.random.choice(data.shape[0], n_samples_to_use, replace=False)

        # data filtering
        data = data[sample_indices, :]
        data = np.delete(data, [np.arange(4, 17)], axis=1)  # remove elements 4..16 (test data info)
        meta_data = data[0, :5]  # is static over all samples --> take from first samples

        if dataset_name in dataset_to_meta_features:
            num_channels = dataset_to_meta_features[dataset_name]['num_channels']
            num_classes = dataset_to_meta_features[dataset_name]['num_classes']

            meta_data[3] = num_channels
            meta_data[4] = num_classes
        else:
            print("Could not get num_channels and num_classes from dataset due to dataset mismatch "
                  "between datasets here {} and here{}".format(dataset_path, processed_datasets_path))

        resnet_data = data[:, 5:]
        resnet_data = np.concatenate(resnet_data, axis=0)
        data = np.concatenate((meta_data, resnet_data))

        dataset_to_nn_meta_features[dataset_name] = data

    df = pd.DataFrame(columns=[np.arange(data.shape[0])])
    for i, dataset_name in enumerate(sorted(list(dataset_to_nn_meta_features.keys()))):
        df.loc[dataset_name] = dataset_to_nn_meta_features[dataset_name]


    if dump_dataframe_csv:
        dump_meta_features_df_and_csv(
            meta_features=df,
            output_path=output_path,
            split_df=split_df,
            file_name=file_name
        )

    return df


def dump_meta_features_df_and_csv(meta_features,
                                  output_path,
                                  split_df=True,
                                  file_name="meta_features"):

    if not isinstance(meta_features, pd.DataFrame):
        df = convert_metadata_to_df(meta_features)
    else:
        df = meta_features

    if split_df:
        df_train = df.loc[df.index.isin(train_datasets)]
        df_valid = df.loc[df.index.isin(val_datasets)]

        df_train.to_csv(output_path / Path(file_name + "_train.csv"))
        df_train.to_pickle(output_path / Path(file_name + "_train.pkl"))

        df_valid.to_csv(output_path / Path(file_name + "_valid.csv"))
        df_valid.to_pickle(output_path / Path(file_name + "_valid.pkl"))

    else:
        df.to_csv((output_path / file_name).with_suffix(".csv"))
        df.to_pickle((output_path / file_name).with_suffix(".pkl"))

    print("meta features data dumped to: {}".format(output_path))


def get_meta_features_from_dataset(dataset_path):
    full_dataset_path = dataset_path / "{}.data/train".format(dataset_path.name)
    if not full_dataset_path.exists():
        full_dataset_path = dataset_path / "{}.data/train".format((dataset_path.name).lower())

    train_dataset = AutoDLDataset(str(full_dataset_path))
    meta_data = train_dataset.get_metadata()
    return parse_meta_features(meta_data)


def get_nn_meta_features_from_dataset(dataset_path):
    if isinstance(dataset_path, Path):
        dataset_path = str(dataset_path)

    processed_datasets = load_processed_datasets(dataset_path=dataset_path)
    print("getting features ...")
    print("using data: {}".format(dataset_path))

    train_feature_data = [ds[0].dataset.numpy() for ds in processed_datasets]
    train_feature_data = np.concatenate(train_feature_data, axis=0)
    return train_feature_data



def precompute_meta_features(dataset_path,
                             output_path,
                             dump_dataframe_csv=True,
                             split_df=True,
                             file_name="meta_features"):

    dataset_to_meta_features = {
        dataset.name: get_meta_features_from_dataset(dataset)
        for dataset in dataset_path.iterdir() if dataset.name in all_datasets
    }

    if dump_dataframe_csv:
        dump_meta_features_df_and_csv(
            meta_features=dataset_to_meta_features,
            output_path=output_path,
            split_df=split_df,
            file_name=file_name
        )

    output_path_yaml = output_path / file_name
    output_path_yaml.with_suffix(".yaml")
    with output_path_yaml.open("w") as out_stream:
        yaml.dump(dataset_to_meta_features, out_stream)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--seed", type=int, default=2, help="random seed")

    parser.add_argument(
        "--processed_datasets_path",
        default="/data/aad/image_datasets/processed_datasets/1e4_combined",
        type=Path,
        help=" "
    )

    parser.add_argument(
        "--dataset_path",
        default="/data/aad/image_datasets/all_symlinks/",
        type=Path,
        help=" "
    )

    parser.add_argument(
        "--output_path", default="src/meta_features/nn", type=Path, help=" "
    )

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    precompute_nn_meta_features(
            args.processed_datasets_path,
            args.output_path,
            args.dataset_path,
            dump_dataframe_csv=True,
            split_df=True
    )


    """ example (non-nn) meta features """
    # parser.add_argument(
    #     "--output_path", default="src/meta_features/non-nn", type=Path, help=" "
    # )

    # args = parser.parse_args()

    # precompute_meta_features(
    #      args.dataset_path,
    #      args.output_path,
    #      dump_dataframe_csv=True,
    #      split_df=False
    # )
