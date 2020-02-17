import numpy as np
import pandas as pd
from src.available_datasets import train_datasets, val_datasets
from pathlib import Path


def get_scores_dataset_x_configs(dataset_dir):
    paths_list = sorted(dataset_dir.glob("*"))
    all_config_paths = [path for path in paths_list if not path.is_file()]

    n_repeat = len(set([int(str(c.absolute())[-1]) for c in all_config_paths]))

    config_names = []
    [config_names.append(config_path.name[:-2])
     for config_path in all_config_paths
     if config_path.name[:-2] not in config_names
     ]

    # splits all config paths [Chuck_0, Chuck_1, ..., Hammer_0, Hammer_1]
    # into according sublists [[Chuck_0, Chuck_1],..., [Hammer_0, Hammer1]]
    config_sublists = [all_config_paths[x:x + n_repeat] for x in range(0, len(all_config_paths), n_repeat)]

    avg_config_scores = []
    for i, config_path_sublist in enumerate(config_sublists):
        config_scores = []
        for config_path in config_path_sublist:
            score_path = config_path / "score" / "scores.txt"
            try:
                # 1: get score + \nduration, 0: get score only
                score = float(score_path.read_text().split(" ")[1].split("\n")[0])
                config_scores.append(score)
            except:
                print("config {} has an issue".format(config_path.name))

        if not config_scores:
            config_names.pop(i)

        avg_config_scores.append(np.mean(config_scores))

    assert len(avg_config_scores) == len(config_names), \
        "something went wrong, number of configs != scores"
    return avg_config_scores, config_names


def create_df_perf_matrix(experiment_group_dir, split_df=True):
    for i, dataset_dir in enumerate(sorted(experiment_group_dir.iterdir())):
        if dataset_dir.is_dir():  # iterdir also yields files
            avg_config_scores, config_names = get_scores_dataset_x_configs(dataset_dir)

            if i == 0:
                # some datasets have been misnamed, correct here:
                config_names = [config_name.replace("Chuck", "Chucky")
                                for config_name in config_names]
                config_names = [config_name.replace("colorectal_histolog", "colorectal_histology")
                                for config_name in config_names]

                df = pd.DataFrame(columns=config_names, index=config_names)

            df.loc[dataset_dir.name] = avg_config_scores

    if split_df:
        df_train = df.loc[df.index.isin(train_datasets)]
        df_valid = df.loc[df.index.isin(val_datasets)]
        return df, df_train, df_valid
    else:
        return df


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--configs_dir", default="src/configs/", type=Path, help=" ")
    parser.add_argument("--output_dir", default="src/configs/", type=Path, help=" ")
    parser.add_argument("--experiment_group_dir", required=True, type=Path, help=" ")
    args = parser.parse_args()

    df, df_train, df_valid = create_df_perf_matrix(args.experiment_group_dir, split_df=True)

    df_train.to_pickle(path=args.experiment_group_dir / "perf_matrix_train.pkl")
    df_train.to_csv(path_or_buf=args.experiment_group_dir / "perf_matrix_train.csv", float_format="%.5f")

    df_valid.to_pickle(path=args.experiment_group_dir / "perf_matrix_valid.pkl")
    df_valid.to_csv(path_or_buf=args.experiment_group_dir / "perf_matrix_valid.csv", float_format="%.5f")

    df.to_pickle(path=args.experiment_group_dir / "perf_matrix.pkl")
    df.to_csv(path_or_buf=args.experiment_group_dir / "perf_matrix.csv", float_format="%.5f")
