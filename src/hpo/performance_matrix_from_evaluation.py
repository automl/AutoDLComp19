from pathlib import Path

import numpy as np
import pandas as pd
from src.available_datasets import train_datasets, val_datasets


def get_scores_dataset_x_configs(dataset_dir):
    paths_list = sorted(dataset_dir.glob("*"))
    all_config_paths = [path for path in paths_list if not path.is_file()]

    n_repeat = len(set([int(str(c.absolute())[-1]) for c in all_config_paths]))

    config_names = []
    [
        config_names.append(config_path.name.rsplit("_", maxsplit=1)[0])
        for config_path in all_config_paths
        if config_path.name.rsplit("_", maxsplit=1)[0] not in config_names
    ]

    def rreplace(s, old, new, occurrence):
        li = s.rsplit(old, occurrence)
        return new.join(li)

    config_names = [
        rreplace(config_path, "origin", "original", 1)
        if config_path.endswith("origin") else config_path for config_path in config_names
    ]

    # splits all config paths [Chuck_0, Chuck_1, ..., Hammer_0, Hammer_1]
    # into according sublists [[Chuck_0, Chuck_1],..., [Hammer_0, Hammer1]]
    config_sublists = [
        all_config_paths[x:x + n_repeat] for x in range(0, len(all_config_paths), n_repeat)
    ]

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
                print(
                    "following config has an issue: {}".
                    format(config_path.parent.name + "/" + config_path.name)
                )

        if not config_scores:
            avg_config_scores.append(0.0)
        else:
            avg_config_scores.append(np.mean(config_scores))

    assert len(avg_config_scores) == len(config_names), \
        "something went wrong, number of configs != scores"
    return avg_config_scores, config_names


def create_df_perf_matrix(experiment_group_dir, split_df=True, existing_df=None):
    for i, dataset_dir in enumerate(sorted(experiment_group_dir.iterdir())):
        if dataset_dir.is_dir():  # iterdir also yields files
            avg_config_scores, config_names = get_scores_dataset_x_configs(dataset_dir)

            if i == 0:
                # some datasets have been misnamed, correct here:
                #config_names = [
                #    config_name.replace("Chuck", "Chucky") for config_name in config_names
                #]

                #config_names = [
                #    config_name.replace("colorectal_histolog", "colorectal_histology")
                #    for config_name in config_names
                #]

                # remove default from indices (i.e. datasets since there are only configs of it)
                indices = config_names.copy()
                indices.remove("default")
                indices.remove("best_generalist")
                # indices.remove("generalist")

                df = pd.DataFrame(columns=config_names, index=indices)

            df.loc[dataset_dir.name] = avg_config_scores

    if existing_df is not None:
        df = pd.concat([existing_df, df], axis=1)

    if split_df:
        df_train = df.loc[df.index.isin(train_datasets)]
        df_valid = df.loc[df.index.isin(val_datasets)]
        return df, df_train, df_valid
    else:
        return df


def transform_to_long_matrix(df, n_samples):
    """
    transform a dataframe of shape (n_algorithms, n_datasets) to
    shape (n_algorithms * n_samples, n_datasets) by simply copying the rows n_sample times.
    This is required s.t. the perf matrix complies with the shape of the feature matrix.
    """
    new_df = pd.DataFrame(columns=df.columns)
    for index, row in df.iterrows():
        for i in range(n_samples):
            new_index = index + "_{}".format(i)
            new_df.loc[new_index] = row

    train_dataset_names = [d + "_{}".format(i) for d in train_datasets for i in range(n_samples)]
    valid_dataset_names = [d + "_{}".format(i) for d in val_datasets for i in range(n_samples)]
    new_df_train = new_df.loc[new_df.index.isin(train_dataset_names)]
    new_df_valid = new_df.loc[new_df.index.isin(valid_dataset_names)]

    return new_df, new_df_train, new_df_valid


def export_df(
    df,
    experiment_group_dir,
    df_train=None,
    df_valid=None,
    file_name="perf_matrix",
    export_path=None
):
    train_file_name = file_name + "_train.pkl"
    valid_file_name = file_name + "_valid.pkl"
    file_name = file_name + ".pkl"

    if export_path is None:
        export_path = experiment_group_dir / "perf_matrix"

    export_path.mkdir(parents=True, exist_ok=True)

    df.to_pickle(path=export_path / file_name)
    df.to_csv(path_or_buf=(export_path / file_name).with_suffix(".csv"), float_format="%.5f")

    if df_train is not None:
        df_train.to_pickle(path=export_path / train_file_name)
        df_train.to_csv(
            path_or_buf=(export_path / train_file_name).with_suffix(".csv"), float_format="%.5f"
        )

    if df_valid is not None:
        df_valid.to_pickle(path=export_path / valid_file_name)
        df_valid.to_csv(
            path_or_buf=(export_path / valid_file_name).with_suffix(".csv"), float_format="%.5f"
        )


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--experiment_group_dir", required=True, type=Path, help=" ")
    args = parser.parse_args()

    #df_to_merge = pd.read_pickle(
    #    "experiments/kakaobrain_optimized_per_dataset_datasets_x_configs_evaluations/perf_matrix/perf_matrix.pkl"
    #)

    df, df_train, df_valid = create_df_perf_matrix(
        args.experiment_group_dir,
        split_df=True,
        existing_df=None  # df_to_merge
    )
    export_df(
        df=df,
        experiment_group_dir=args.experiment_group_dir,
        #df_train=df_train,
        #df_valid=df_valid,
        file_name="perf_matrix",
        export_path=args.experiment_group_dir.parent /
        (args.experiment_group_dir.name + "_perf_matrix")
    )

    #df, df_train, df_valid = transform_to_long_matrix(df, n_samples=100)
    #export_df(df=df, experiment_group_dir=args.experiment_group_dir, df_train=df_train, df_valid=df_valid, file_name="perf_matrix_samples_along_rows")
