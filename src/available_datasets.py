from pathlib import Path

import yaml


def get_available_dataset_names(valid_keys, no_augmented=False):
    """
    Produces two lists of dataset names, one containing training and the other containing
    validation dataset names. Path to the datasets directory must be specified through
    src/configs/default.yaml (cluster_datasets_dir key).
    Parameters:
      valid_keys - A list of strings. Each string indicates which dataset will be placed in the
      returned valid dataset list.
      no_augmented - Flag indicating whether only non-augmented datasets ending with the keyword
      '_original' should be returned.
    """
    # Read the default config
    configs_path = Path("src/configs/")
    default_config_path = configs_path / "default.yaml"

    with default_config_path.open() as in_stream:
        default_config = yaml.safe_load(in_stream)

    cluster_datasets_dir = Path(default_config["cluster_datasets_dir"])
    all_datasets = [path.name for path in cluster_datasets_dir.glob("*") if path.is_dir()]

    if no_augmented:
        all_datasets = [dataset for dataset in all_datasets if dataset.endswith("_original")]

    val_datasets = [dataset for dataset in all_datasets for key in valid_keys if key in dataset]
    train_datasets = [dataset for dataset in all_datasets if dataset not in val_datasets]
    return train_datasets, val_datasets


valid_keys = ["coil100", "kmnist", "vgg-flowers", "oxford_iiit_pet"]
train_datasets, val_datasets = get_available_dataset_names(
    valid_keys=valid_keys, no_augmented=False
)
all_datasets = train_datasets + val_datasets
