import json
import os

import hpbandster.core.result as hpres


def get_log_subfolders(log_dir):
    log_folder_list = []
    for root, dirs, files in os.walk(log_dir):
        if "___" in root:
            log_folder_list.append(root)
    return log_folder_list


def extract_dataset_and_model(log_folder):
    last_folder = log_folder.split("/")[-1]
    dataset = last_folder.split("___")[0]
    model = last_folder.split("___")[1]
    return dataset, model


def get_all_datasets_and_models(log_dir):
    log_folder_list = get_log_subfolders(log_dir)
    dataset_set = set()
    model_set = set()
    for log_folder in log_folder_list:
        dataset, model = extract_dataset_and_model(log_folder)
        dataset_set.add(dataset)
        model_set.add(model)

    return dataset_set, model_set


def find_complete_datasets(log_dir, exclude_datasets):
    """
    find all datasets
    - for which all BOHB runs have finished (i.e. all models have been evaluated)
    - which are not manually excluded (for example because they are used as test datasets)
    """

    _, all_model_set = get_all_datasets_and_models(log_dir)

    log_folder_list = get_log_subfolders(log_dir)
    dm_dict = {}
    complete_dataset_set = set()
    for log_folder in log_folder_list:
        dataset, model = extract_dataset_and_model(log_folder)
        if dataset in dm_dict:
            dm_dict[dataset].append(model)
        else:
            dm_dict[dataset] = [model]

    for dataset in dm_dict.keys():
        processed_models = set(dm_dict[dataset])
        missing_models = all_model_set.difference(processed_models)
        if not len(missing_models) == 0:
            print(
                'No BOHB run on dataset "'
                + dataset
                + '" with the following models: '
                + str(missing_models)
            )
        elif dataset in exclude_datasets:
            print('Manually excluded dataset "' + dataset + '" from further analysis')
        else:
            complete_dataset_set.add(dataset)

    return complete_dataset_set


def get_max_budget(all_runs):
    max_budget = 0
    for run in all_runs:
        if run.budget > max_budget:
            max_budget = run.budget
    return max_budget


def get_best_run_id(all_runs, max_budget):
    best_loss = 0
    best_id = -1

    for id in range(len(all_runs)):
        loss = all_runs[id].loss
        if loss is None:
            continue

        budget = all_runs[id].budget
        if loss < best_loss and abs(budget - max_budget) < 1e-2:
            best_loss = loss
            best_id = id

    return best_id


def find_best_results(complete_dataset_set):
    """
    find incumbent for every dataset
    """

    log_folder_list = get_log_subfolders(log_dir)
    result_dict = {}

    for log_folder in log_folder_list:
        dataset, model = extract_dataset_and_model(log_folder)

        if dataset not in complete_dataset_set:
            continue

        result = hpres.logged_results_to_HBS_result(log_folder)
        all_runs = result.get_all_runs()
        max_budget = get_max_budget(all_runs)
        best_id = get_best_run_id(all_runs, max_budget)
        best_loss = all_runs[best_id].loss
        best_conf = all_runs[best_id].info["config"]

        result_tuple = (model, -best_loss, best_conf)

        if dataset in result_dict:
            result_dict[dataset].append(result_tuple)
        else:
            result_dict[dataset] = [result_tuple]

    for k, v in result_dict.items():
        v.sort(key=lambda tup: tup[1], reverse=True)
        result_dict[k] = v[0]

    return result_dict


def save_best_results(result_dict, save_path):
    json.dump(result_dict, open(save_path, "w"))


if __name__ == "__main__":
    log_dir = "/home/nierhoff/AutoDLComp19/src/video3/autodl_starting_kit_stable/logs_new"  # directory containing the BOHB results
    best_result_path = "/home/stolld/autodl/incumbent.json"
    exclude_datasets = {
        "Chucky",
        "Decal",
        "Munster",
        "Pedro",
        "Hammer",
        "Katze",
        "Kreatur",
        "Kraut",
    }

    complete_dataset_set = find_complete_datasets(log_dir, exclude_datasets)
    result_dict = find_best_results(complete_dataset_set)
    save_best_results(result_dict, best_result_path)
