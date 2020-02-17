datasets_to_exclude = ["Hmdb51", "SMv2", "Ucf101", "emnist"]

train_datasets = [
    "Chucky",
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
    "colorectal_histology",
    "deep_weeds",
    "emnist",
    "eurosat",
    "fashion_mnist",
    "horses_or_humans",
    "mnist",
]

train_datasets = [x for x in train_datasets if x not in datasets_to_exclude]

val_datasets = ["Decal", "coil100", "kmnist", "oxford_flowers102"]

val_datasets = [x for x in val_datasets if x not in datasets_to_exclude]

all_datasets = train_datasets + val_datasets



