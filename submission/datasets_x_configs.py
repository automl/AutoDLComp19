import itertools as it
from pathlib import Path

import yaml
from src.available_datasets import all_datasets


def construct_command(config, dataset, base_datasets_dir):
    dataset_dir = Path(base_datasets_dir, dataset)
    return "--model_config_name {} --dataset_dir {} --experiment_name {}/{}".format(
        config, dataset_dir, dataset, config.rstrip(".yaml")
    )


configs_path = Path("src/configs")


def generate_all_commands(configs_path, commands_file):
    with Path(configs_path, "default.yaml").open() as in_stream:
        config = yaml.safe_load(in_stream)
    base_datasets_dir = config["cluster_datasets_dir"]

    all_configs = [config_path.name for config_path in configs_path.glob("*")]

    commands = [
        construct_command(config, dataset, base_datasets_dir)
        for config, dataset in it.product(all_configs, all_datasets)
    ]
    commands_file.write_text("\n".join(commands))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--configs_path", default="src/configs", type=Path, help=" ")
    parser.add_argument("--command_file_name", default="dataset_x_configs_v1.args", help=" ")

    args = parser.parse_args()

    commands_file = Path("submission") / args.command_file_name
    generate_all_commands(args.configs_path, commands_file)
