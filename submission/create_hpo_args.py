from pathlib import Path

from src.available_datasets import train_datasets


def construct_command(dataset, worker, optimize_generalist):
    if worker and optimize_generalist:
        return "--experiment_name {} --optimize_generalist --worker".format(dataset)
    elif not worker and optimize_generalist:
        return "--experiment_name {} --optimize_generalist".format(dataset)
    elif worker and not optimize_generalist:
        return "--experiment_name {} --worker".format(dataset)
    else:
        return "--experiment_name {}".format(dataset)


def generate_all_commands(commands_file, remove_datasets, optimize_generalist):
    commands = [
        construct_command(dataset, worker, optimize_generalist) for dataset in train_datasets
        for worker in [True, False] if not dataset in remove_datasets
    ]
    commands_file.write_text("\n".join(commands))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        "--command_file_name", default="hpo_args_individualists_v3_new_datasets.args", help=" "
    )
    parser.add_argument("--optimize_generalist", default=False, type=bool, help=" ")
    args = parser.parse_args()

    remove_datasets = []

    commands_file = Path("submission") / args.command_file_name
    generate_all_commands(commands_file, remove_datasets, args.optimize_generalist)
