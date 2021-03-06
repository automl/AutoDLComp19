import itertools as it
import os
from pathlib import Path
from random import shuffle

import yaml
from src.available_datasets import all_datasets


def construct_command(config, dataset, base_datasets_dir, repeat, configs_path):
    dataset_dir = Path(base_datasets_dir, dataset)
    return "--model_config_name {} --dataset_dir {} --experiment_name {}/{}_{}".format(
        os.path.join(configs_path.name, config), dataset_dir, dataset,
        config.rsplit(".yaml")[0], repeat
    )


def generate_all_commands(
    configs_path, commands_file, n_repeats, row_wise=True, optimize_access=True
):
    """
    Parameters:
        row_wise: if True, commands are generated s.t. all configs (columns) are evaluated for one
        dataset (rows) first before moving on to the next dataset. If False, all datasets are run for
        one config file before moving on to the next config file.
        optimize_access: shuffle the commands at random for a window of 20 configs x n datasets
        in the table to reduce number of simultaenous dataset accesses. n=20 per default
    """
    with Path(configs_path.parent, "default.yaml").open() as in_stream:
        config = yaml.safe_load(in_stream)
        print("using {} as default config file".format(str(configs_path.parent) + "/default.yaml"))
    base_datasets_dir = config["cluster_datasets_dir"]

    all_configs = [config_path.name for config_path in configs_path.glob("*")]

    if row_wise:

        def _chunks(lst, n):
            """Yield successive n-sized chunks from lst."""
            for i in range(0, len(lst), n):
                yield lst[i:i + n]

        all_datasets_chunks = list(_chunks(all_datasets, 20))

        commands = []
        for dataset_chunk in all_datasets_chunks:
            sub_list_commands = []
            for dataset in dataset_chunk:
                for config in all_configs:
                    sub_list_commands.append((config, dataset))

            if optimize_access:
                shuffle(sub_list_commands)

            for sub_cmd in sub_list_commands:
                for repeat in range(n_repeats):
                    commands.append(
                        construct_command(
                            sub_cmd[0], sub_cmd[1], base_datasets_dir, repeat, configs_path
                        )
                    )

    else:
        commands = [
            construct_command(config, dataset, base_datasets_dir, repeat, configs_path)
            for config, dataset, repeat in it.product(all_configs, all_datasets, range(n_repeats))
        ]
    commands_file.write_text("\n".join(commands))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        "--configs_path",
        default="src/configs",
        type=Path,
        help="Specifies where the incumbent configurations are stored"
    )
    parser.add_argument(
        "--command_file_name",
        default="dataset_x_configs_v4.args",
        help="Specifies the name of the args file to be outputted"
    )
    parser.add_argument(
        "--n_repeats",
        default=3,
        type=int,
        help="Specifies how many times one incumbent configuration should be evaluated per dataset"
    )

    args = parser.parse_args()

    commands_file = Path("submission") / args.command_file_name
    generate_all_commands(args.configs_path, commands_file, args.n_repeats)
