from pathlib import Path

import yaml


def get_available_dataset_names(
    valid_keys,
    selected_train_datasets=None,
    no_augmented=False,
    delimiter_starts_with="_",
    no_augmented_str="_original"
):
    """
    Produces two lists of dataset names, one containing training and the other containing
    validation dataset names. Path to the datasets directory must be specified through
    src/configs/default.yaml (cluster_datasets_dir key).
    Parameters:
      valid_keys - A list of strings. Each string indicates which dataset will be placed in the
      returned valid dataset list.
      selected_train_datasets - A list of strings. Each string represents a dataset that is supposed
      to be searched for and if it exists, the respective datasets are returned in the training
      dataset list.
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
        all_datasets = [dataset for dataset in all_datasets if dataset.endswith(no_augmented_str)]

    val_datasets = [dataset for dataset in all_datasets for key in valid_keys if key in dataset]
    train_datasets = [dataset for dataset in all_datasets if dataset not in val_datasets]

    if delimiter_starts_with is not None:
        starts_with_str = "{}" + delimiter_starts_with
    else:
        starts_with_str = "{}"

    if selected_train_datasets:
        train_datasets = [
            dataset for dataset in train_datasets
            for key in selected_train_datasets if dataset.startswith(starts_with_str.format(key))
        ]

    return sorted(set(train_datasets)), sorted(set(val_datasets))


NO_AUGMENTED = True

# train_keys = [
#     "cifar100",  # objects
#     "cifar10",  # objects
#     "mnist",  # handwritten digits, replacement for Munster
#     "colorectal_histology",  # colorectal cancer
#     "caltech_birds2010",  # birds
#     "eurosat",  # satellite images
#     "cars196",  # cars
#     "visual_domain_decathlon_dtd",  # textures, # replacement for Hammer
#     "imagenette",  # subset of imagenet
#     #"imagenet_resized",  # imagenet resized to 32x32, yields problems
#     "caltech101",  # imagenet resized to 32x32
#     "malaria",  # cell images, # replacement for Hammer
#     "svhn_cropped",  # house numbers
#     "uc_merced",  # urban area imagery
#     "visual_domain_decathlon_daimlerpedcls",  # pedestrians
#     "oxford_flowers102",  # flowers
#     "fashion_mnist",  # fashion articles
#     "citrus_leaves",  # citrus fruits and leaves
#     "cycle_gan_summer2winter_yosemite",  # landscape
#     "cycle_gan_facades",  # facades
#     "visual_domain_decathlon_ucf101"  # youtube action
# ]

train_keys = [
    "Chucky",
    "Decal",
    "Hammer",
    "Munster",
    "Pedro",
    "binary_alpha_digits",
    "caltech101",
    "caltech_birds2010",
    "caltech_birds2011",
    "cats_vs_dogs",
    "cifar10",
    "cifar100",
    "coil100",
    "colorectal_histology",
    "deep_weeds",
    "emnist",
    "eurosat",
    "fashion_mnist",
    "horses_or_humans",
    "kmnist",
    "oxford_flowers102",
]

#valid_keys = ["coil100", "kmnist", "vgg-flowers", "oxford_iiit_pet", "cmaterdb_telugu"]
valid_keys = ["cmaterdb_telugu"]
# lists ending without _all contain subset of available datasets (not filtered by keys)
train_datasets, val_datasets = get_available_dataset_names(
    valid_keys=valid_keys,
    selected_train_datasets=train_keys,
    no_augmented=NO_AUGMENTED,
    delimiter_starts_with=None,  # todo: remove this for augmented datasets
    #delimiter_starts_with="_",
    no_augmented_str=""  # todo: remove this for augmented datasets
    #no_augmented_str="_original"
)

# lists ending with _all contain all datasets (with augmented and not filtered by keys), e.g.
# useful for evaluation in src/hpo/performance_matrix_from_evaluation.py
train_datasets_all, val_datasets_all = get_available_dataset_names(
    valid_keys=valid_keys, no_augmented=False
)

all_datasets = train_datasets_all + val_datasets_all

#remove_datasets = ["imagenet_resized"]
#all_datasets = [i for i in all_datasets for j in remove_datasets if j not in i]
all_datasets = train_datasets  # todo: remove
