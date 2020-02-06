import os
import pickle

import numpy as np
import tensorflow as tf
import torch
import torchvision

from src.competition.ingestion_program.dataset import AutoDLDataset
from src.dataset_kakaobrain import TFDataset
from src.utils import *


def generate_meta_output_vector(data, general_vector):
    """
    small helper function
    """

    length = data.shape[-4]
    width = data.shape[-3]
    height = data.shape[-2]
    channels = data.shape[1]

    output_vector = np.array([[length, width, height, channels]])
    output_vector = np.concatenate([output_vector, general_vector], axis=1)

    return output_vector


def generate_samples_meta(dataset_dir_raw_in, dataset_dir_out, des_num_samples):
    """
    extract simple-to-calculate meta features for a given dataset

    :param dataset_dir_raw_in: path to the input AutoDL dataset
    :param dataset_dir_out: path to the output directory where the metadata shall be written
    :param des_num_samples: minimum number of samples to process
    """
    session = tf.Session()
    batch_size = 1  # must not be be increased for datasets with varying image dimensions

    os.makedirs(dataset_dir_out, exist_ok=True)

    # load train and test dataset
    _, dataset_name = os.path.split(dataset_dir_raw_in)
    dataset_train = AutoDLDataset(
        os.path.join(dataset_dir_raw_in, dataset_name + ".data", "train")
    ).get_dataset()
    dataset_test = AutoDLDataset(
        os.path.join(dataset_dir_raw_in, dataset_name + ".data", "test")
    ).get_dataset()
    ds_temp = TFDataset(session=session, dataset=dataset_test, num_samples=int(1e9))
    info = ds_temp.scan()

    ds_train = TFDataset(
        session=session, dataset=dataset_train, num_samples=int(1e9), transform=None
    )
    ds_test = TFDataset(
        session=session, dataset=dataset_test, num_samples=info["num_samples"], transform=None
    )
    dl_train = torch.utils.data.DataLoader(
        ds_train, batch_size=batch_size, shuffle=False, drop_last=False
    )
    dl_test = torch.utils.data.DataLoader(
        ds_test, batch_size=batch_size, shuffle=False, drop_last=False
    )

    # generate part of the meta-feature vector that is identical for all samples
    general_vector = np.array([info["num_samples"]])
    general_vector = np.concatenate([general_vector, info["min_shape"]])
    general_vector = np.concatenate([general_vector, info["max_shape"]])
    general_vector = np.concatenate([general_vector, info["avg_shape"]])
    general_vector = np.concatenate([general_vector, np.array([1 if info["is_multilabel"] else 0])])
    general_vector = np.array([general_vector])

    # process train data
    num_samples = 0
    output_list_train = []
    while num_samples < des_num_samples:
        for data, _ in dl_train:
            output_vector = generate_meta_output_vector(data, general_vector)
            output_list_train.append(output_vector)

            num_samples += batch_size
            if num_samples >= des_num_samples:
                break

    # process test data
    output_list_test = []
    for data, _ in dl_test:
        output_vector = generate_meta_output_vector(data, general_vector)
        output_list_test.append(output_vector)

    output_train = np.concatenate(output_list_train, axis=0)
    output_test = np.concatenate(output_list_test, axis=0)

    # save files
    file_train = os.path.join(dataset_dir_out, dataset_name + "_train")
    file_test = os.path.join(dataset_dir_out, dataset_name + "_test")
    with open(file_train, "wb") as fh_train:
        pickle.dump(output_train, fh_train)
    with open(file_test, "wb") as fh_test:
        pickle.dump(output_test, fh_test)


def generate_samples_resnet(dataset_dir_raw_in, dataset_dir_out, des_num_samples):
    """
    extract meta features from the output of a Resnet18 model

    :param dataset_dir_raw_in: path to the input AutoDL dataset
    :param dataset_dir_out: path to the output directory where the metadata shall be written
    :param des_num_samples: minimum number of samples to process
    """

    model = torchvision.models.resnet18(pretrained=True)
    model.cuda()
    model.fc = Identity()
    session = tf.Session()
    batch_size = 512
    input_size = 128

    os.makedirs(dataset_dir_out, exist_ok=True)

    _, dataset_name = os.path.split(dataset_dir_raw_in)
    dataset_train = AutoDLDataset(
        os.path.join(dataset_dir_raw_in, dataset_name + ".data", "train")
    ).get_dataset()
    dataset_test = AutoDLDataset(
        os.path.join(dataset_dir_raw_in, dataset_name + ".data", "test")
    ).get_dataset()

    transform_train = get_transform(is_training=True, input_size=input_size)
    transform_test = get_transform(is_training=False, input_size=input_size)
    ds_temp = TFDataset(session=session, dataset=dataset_test, num_samples=int(1e9))
    info = ds_temp.scan()

    ds_train = TFDataset(
        session=session, dataset=dataset_train, num_samples=int(1e9), transform=transform_train
    )
    ds_test = TFDataset(
        session=session,
        dataset=dataset_test,
        num_samples=info["num_samples"],
        transform=transform_test,
    )
    dl_train = torch.utils.data.DataLoader(
        ds_train, batch_size=batch_size, shuffle=False, drop_last=False
    )
    dl_test = torch.utils.data.DataLoader(
        ds_test, batch_size=batch_size, shuffle=False, drop_last=False
    )

    torch.set_grad_enabled(False)
    model.eval()

    # process train data
    num_samples = 0
    output_list_train = []
    while num_samples < des_num_samples:
        for data, _ in dl_train:
            output = model(data.cuda())
            output_list_train.append(output.cpu().numpy())

            num_samples += batch_size
            if num_samples > des_num_samples:
                break

    # process test data
    output_list_test = []
    for data, _ in dl_test:
        output = model(data.cuda())
        output_list_test.append(output.cpu().numpy())

    output_train = np.concatenate(output_list_train, axis=0)
    output_test = np.concatenate(output_list_test, axis=0)

    # save files
    file_train = os.path.join(dataset_dir_out, dataset_name + "_train")
    file_test = os.path.join(dataset_dir_out, dataset_name + "_test")
    with open(file_train, "wb") as fh_train:
        pickle.dump(output_train, fh_train)
    with open(file_test, "wb") as fh_test:
        pickle.dump(output_test, fh_test)


def generate_samples_combined(
    dataset_resnet_path_in, dataset_meta_path_in, dataset_combined_dir_out
):
    """
    combine the meta and resnet datasets
    """

    os.makedirs(dataset_combined_dir_out, exist_ok=True)

    for suffix in ["train", "test"]:
        dataset_resnet_file = os.path.normpath(dataset_resnet_path_in + "_" + suffix)
        dataset_meta_file = os.path.normpath(dataset_meta_path_in + "_" + suffix)

        with open(dataset_resnet_file, "rb") as fh:
            dataset_resnet = np.array(pickle.load(fh))
        with open(dataset_meta_file, "rb") as fh:
            dataset_meta = np.array(pickle.load(fh))

        min_len = min(len(dataset_resnet), len(dataset_meta))
        dataset_combined = np.concatenate(
            [dataset_meta[:min_len], dataset_resnet[:min_len]], axis=1
        )
        dataset_combined = np.float32(dataset_combined)

        _, dataset_name_with_suffix = os.path.split(dataset_resnet_file)
        file_combined = os.path.join(dataset_combined_dir_out, dataset_name_with_suffix)
        with open(file_combined, "wb") as fh:
            pickle.dump(dataset_combined, fh)


if __name__ == "__main__":
    # folder containing the raw AutoDL dataset
    dataset_dir_raw_in = "/home/dingsda/data/datasets/challenge/image/mnist"
    # folder where the meta feature dataset shall be written
    dataset_dir_meta_out = "/home/dingsda/data/datasets/demo_meta"
    # folder where the resnet meta feature dataset shall be written
    dataset_dir_resnet_out = "/home/dingsda/data/datasets/demo_resnet"

    generate_samples_meta(dataset_dir_raw_in, dataset_dir_meta_out, 5000)
    generate_samples_resnet(dataset_dir_raw_in, dataset_dir_resnet_out, 5000)

    # create the 530 dimensional dataset by concatenating the 18- and 512-dimensional meta feature datasets
    dataset_resnet_path_in = "/home/dingsda/data/datasets/demo_resnet/mnist"
    dataset_meta_path_in = "/home/dingsda/data/datasets/demo_meta/mnist"
    dataset_combined_dir_out = "/home/dingsda/data/datasets/demo_combined"

    generate_samples_combined(
        dataset_resnet_path_in, dataset_meta_path_in, dataset_combined_dir_out
    )
