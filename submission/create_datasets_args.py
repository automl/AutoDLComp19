from pathlib import Path
from src.available_datasets import all_datasets


def construct_command(dataset):
    return "--experiment_name {}".format(dataset)


def generate_all_commands(commands_file, remove_datasets):
    commands = [
        construct_command(dataset) for dataset in all_datasets if not dataset in remove_datasets
    ]
    commands_file.write_text("\n".join(commands))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--command_file_name", default="dataset_v1.args", help=" ")
    args = parser.parse_args()

    remove_datasets = ["emnist"]

    commands_file = Path("submission") / args.command_file_name
    generate_all_commands(commands_file, remove_datasets)
