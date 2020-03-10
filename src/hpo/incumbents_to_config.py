from pathlib import Path

import hpbandster.core.result as hpres
import yaml
from src.hpo.utils import construct_model_config


def incumbent_to_config(experiment_path, configs_path, output_dir):
    # Read the incumbent
    result = hpres.logged_results_to_HBS_result(str(experiment_path))
    id2conf = result.get_id2config_mapping()
    inc_id = result.get_incumbent_id()
    incumbent_config = id2conf[inc_id]['config']

    # Read the default config
    default_config_path = configs_path / "default.yaml"
    with default_config_path.open() as in_stream:
        default_config = yaml.safe_load(in_stream)

    # Compute and write incumbent config in the format of default_config
    incumbent_config = construct_model_config(incumbent_config, default_config)

    out_config_path = output_dir / "{}.yaml".format(experiment_path.name)
    with out_config_path.open("w") as out_stream:
        yaml.dump(incumbent_config, out_stream)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--configs_dir",
        default="src/configs/",
        type=Path,
        help="Specifies where the default yaml file is (usually set to src/configs/"
    )
    parser.add_argument(
        "--output_dir",
        default="src/configs/effnet_optimized_per_dataset_new_cs_new_data_03_09",
        type=Path,
        help="Specifies where the incumbent configs should be stored e.g. src/configs/experiment_name"
    )
    parser.add_argument(
        "--experiment_group_dir",
        required=True,
        type=Path,
        help="Specifies the path to the bohb working directory of an experiment"
    )
    args = parser.parse_args()

    for experiment_path in args.experiment_group_dir.iterdir():
        try:
            incumbent_to_config(experiment_path, args.configs_dir, args.output_dir)
        except:
            print(experiment_path.name, " has an issue")
