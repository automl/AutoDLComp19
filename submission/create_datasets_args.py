from pathlib import Path
from src.available_datasets import all_datasets


def construct_command(dataset, worker):
    if worker:
        return "--experiment_name {} --optimize_generalist --worker".format(dataset)
    else:
        return "--experiment_name {} --optimize_generalist".format(dataset)


def generate_all_commands(commands_file, remove_datasets):
    commands = [
        construct_command(dataset, worker) for dataset in all_datasets for worker in [True, False] if not dataset in remove_datasets
    ]
    commands_file.write_text("\n".join(commands))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--command_file_name", default="dataset_all_opt_v2.args", help=" ")
    args = parser.parse_args()

    remove_datasets = ["emnist"]

    commands_file = Path("submission") / args.command_file_name
    generate_all_commands(commands_file, remove_datasets)
