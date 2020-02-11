from pathlib import Path

import yaml
from src.competition.ingestion_program.dataset import AutoDLDataset


def parse_meta_features(meta_data):
    sequence_size, x, y, num_channels = meta_data.get_tensor_shape()
    num_classes = meta_data.get_output_size()
    return dict(
        num_classes=num_classes,
        sequence_size=sequence_size,
        resolution=(x, y),
        num_channels=num_channels
    )


def get_meta_features_from_dataset(dataset_path):
    dataset_path = dataset_path / "{}.data/train".format(dataset_path.name)
    train_dataset = AutoDLDataset(str(dataset_path))
    meta_data = train_dataset.get_metadata()
    return parse_meta_features(meta_data)


def precompute_meta_features(dataset_path, output_path):
    dataset_to_meta_features = {
        dataset.name: get_meta_features_from_dataset(dataset)
        for dataset in dataset_path.iterdir()
    }
    with output_path.open("w") as out_stream:
        yaml.dump(dataset_to_meta_features, out_stream)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        "--dataset_path", default="/data/aad/image_datasets/challenge/", type=Path, help=" "
    )
    parser.add_argument(
        "--output_path", default="src/meta_features/meta_features.yaml", type=Path, help=" "
    )

    args = parser.parse_args()

    precompute_meta_features(args.dataset_path, args.output_path)
