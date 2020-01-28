import os
import sys
import json
import matplotlib.pyplot as plt
import hpbandster.core.result as hpres
import hpbandster.visualization as hpvis
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle

sys.path.append(os.path.join(os.getcwd(), 'AutoDL_ingestion_program'))
sys.path.append(os.path.join(os.getcwd(), 'AutoDL_scoring_program'))

#def load_log_names
LOG_FOLDER = '/home/dingsda/logs_new'
SAVE_FOLDER = os.path.join(os.getcwd(), 'common', 'files')
EXCLUDE_DATASETS = {'Chucky', 'Decal', 'Munster', 'Pedro', 'Hammer', 'Katze', 'Kreatur', 'Kraut'}
OPTIM_PORTFOLIO_SIZE = 3
OPTIM_RESTARTS = 3
OPTIM_ITERATIONS = 1000
OPTIM_MODE = 'mixed'

def get_log_folders():
    log_folder_list = []
    for root, dirs, files in os.walk(LOG_FOLDER):
        if '___' in root:
            log_folder_list.append(root)
    return log_folder_list

def extract_dataset_and_model(log_folder):
    last_folder = log_folder.split('/')[-1]
    dataset = last_folder.split('___')[0]
    model = last_folder.split('___')[1]
    return dataset, model

def get_all_datasets_and_models():
    log_folder_list = get_log_folders()
    dataset_set = set()
    model_set = set()
    for log_folder in log_folder_list:
        dataset, model = extract_dataset_and_model(log_folder)
        dataset_set.add(dataset)
        model_set.add(model)

    return dataset_set, model_set

def find_complete_datasets():
    _, all_model_set = get_all_datasets_and_models()

    log_folder_list = get_log_folders()
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
            print('No BOHB run on dataset "' + dataset + '" with the following models: ' + str(missing_models))
        elif dataset in EXCLUDE_DATASETS:
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
        if loss < best_loss and abs(budget-max_budget) < 1e-2:
            best_loss = loss
            best_id = id

    return best_id

def find_best_results(complete_dataset_set):
    log_folder_list = get_log_folders()
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
        best_conf = all_runs[best_id].info['config']

        result_tuple = (model, -best_loss, best_conf)

        if dataset in result_dict:
            result_dict[dataset].append(result_tuple)
        else:
            result_dict[dataset] = [result_tuple]

    return result_dict

def rank_result(result_dict):
    _, all_model_set = get_all_datasets_and_models()

    model_rank_dict = {}
    model_loss_dict = {}
    for model in all_model_set:
        model_rank_dict[model] = []
        model_loss_dict[model] = []

    for k,v in result_dict.items():
        v.sort(key=lambda tup: tup[1], reverse=True)

        for id in range(len(v)):
            model = v[id][0]
            best_acc = v[id][1]
            model_rank_dict[model].append(id+1)
            model_loss_dict[model].append(best_acc)

    return model_rank_dict, model_loss_dict

def plot_rank_result(model_dict, xlabel, ylabel, invert=False):
    name_list = []
    data = []
    median = []

    for k,v in sorted(model_dict.items()):
        name_list.append(k)
        data_new = np.asarray(v)
        data.append(data_new)
        median.append(np.median(data_new))

    median = np.array(median)
    order_list = list(np.argsort(median))

    name_list_sorted = [None] * len(order_list)
    for i in range(len(order_list)):
        name_list_sorted[i] = name_list[order_list[i]]

    data = pd.DataFrame(data)
    data = data.transpose()
    data.columns = name_list

    plt.figure(figsize=(5, 12))
    #sns.boxplot(x=data, y=name_list, order=name_list_sorted)
    ax = sns.boxplot(data=data, order=name_list_sorted, orient='h')
    #sns.swarmplot(data=data, order=name_list_sorted, orient='h', color='k')
    #sns.boxplot(x=data, y=name_list)
    ax.set(xlabel=xlabel, ylabel=ylabel)
    if invert:
        plt.ylim(reversed(plt.ylim()))
    else:
        plt.xlim(reversed(plt.xlim()))
    plt.savefig("rank_results.svg", format="svg")
    plt.show()


def plot_dataset_results(result_dict, xlabel, ylabel):
    name_list = []
    data = []
    median = []

    for k,v in sorted(result_dict.items()):
        name_list.append(k)
        data_new = np.asarray([tup[1] for tup in v])
        data.append(data_new)
        median.append(np.median(data_new))

    median = np.array(median)
    order_list = list(np.argsort(median))

    name_list_sorted = [None] * len(order_list)
    for i in range(len(order_list)):
        name_list_sorted[i] = name_list[order_list[i]]

    data = pd.DataFrame(data)
    data = data.transpose()
    data.columns = name_list

    plt.figure(figsize=(5, 5))
    ax = sns.boxplot(data=data, order=name_list_sorted, orient='h')
    ax.set(xlabel=xlabel, ylabel=ylabel)
    plt.ylim(reversed(plt.ylim()))
    plt.savefig("dataset_results.svg", format="svg")
    plt.show()


def calculate_portfolio_performance(pf, data, optim_mode=None):
    pf_perf = data[pf]
    if pf_perf.shape[0] > 1:
        pf_perf_best = np.max(pf_perf, axis=0)
    else:
        pf_perf_best = pf_perf

    if optim_mode == 'avg':
        return np.mean(pf_perf_best)
    elif optim_mode == 'min':
        return np.min(pf_perf_best)
    elif optim_mode == 'mixed':
        return 0.5*(np.mean(pf_perf_best)+np.min(pf_perf_best))
    else:
        print('unknown optimization option')


def store_model_portfolio(result_dict, best_models):
    json.dump(result_dict, open(os.path.join(SAVE_FOLDER, 'result_dict.json'), 'w'))
    json.dump(best_models, open(os.path.join(SAVE_FOLDER, 'best_models.json'), 'w'))

def optimize_model_portfolio(result_dict):
    print(result_dict)
    _, all_model_set = get_all_datasets_and_models()
    nb_dataset = len(result_dict)
    nb_model = len(all_model_set)

    # initialize array structure
    data = np.empty([nb_model, nb_dataset])
    for i,kv in enumerate(sorted(result_dict.items())):
        v = kv[1]
        v.sort(key=lambda tup: tup[0])
        for j in range(len(v)):
            data[j,i] = v[j][1]

    best_pf = np.random.randint(nb_model, size=OPTIM_PORTFOLIO_SIZE)
    best_perf = calculate_portfolio_performance(best_pf, data, OPTIM_MODE)

    for rs in range(OPTIM_RESTARTS):
        if rs % 100 == 0:
            print(rs)

        # initialize model portfolio
        pf = np.random.randint(nb_model, size=OPTIM_PORTFOLIO_SIZE)
        perf = calculate_portfolio_performance(pf, data, OPTIM_MODE)

        for it in range(OPTIM_ITERATIONS):
            # select random portfolio position to be optimized
            pos = np.random.randint(OPTIM_PORTFOLIO_SIZE)

            # modify portfolio and recalculate performance
            new_pf = np.copy(pf)

            for i in range(data.shape[0]):
                new_pf[pos] = i
                new_perf = calculate_portfolio_performance(new_pf, data, OPTIM_MODE)
                if new_perf > perf:
                    pf = np.copy(new_pf)
                    perf = new_perf

        if perf > best_perf:
            best_pf = np.copy(pf)
            best_perf = perf

    best_pf = np.sort(best_pf)
    print(best_pf)
    print(best_perf)
    print(calculate_portfolio_performance(best_pf, data, 'avg'))
    print(calculate_portfolio_performance(best_pf, data, 'min'))
    best_models = [list(sorted(all_model_set))[i] for i in best_pf]
    print(best_models)

    store_model_portfolio(result_dict, best_models)


def plot_model_portfolio_performance():
    x_data = [1,2,3,4,5,6,7,8,9,10]
    y_data_avg = [0.6288899419576274,
                  0.7324926642859595,
                  0.7504368549613724,
                  0.7595902190598857,
                  0.7632494748379203,
                  0.7650573938313365,
                  0.7662108166810708,
                  0.7670273560173989,
                  0.7677931442940776,
                  0.7683707398174213]
    y_data_min = [0.22896530551963362,
                  0.2534047552367036,
                  0.27800448495806007,
                  0.27800448495806007,
                  0.27800448495806007,
                  0.27800448495806007,
                  0.27800448495806007,
                  0.27800448495806007,
                  0.27800448495806007,
                  0.27800448495806007]

    plt.figure(figsize=(5,3))
    ax = plt.plot(y_data_avg, lw=2, label='avg. acc.')
    ax = plt.plot(y_data_min, lw=2, label='min. acc.')
    plt.xlabel('portfolio size')
    plt.ylabel('accuracy')
    plt.xticks(np.arange(10), x_data)
    plt.legend()
    plt.savefig("portfolio_performance.svg", format="svg")
    plt.show()


if __name__ == "__main__":
    complete_dataset_set = find_complete_datasets()
    result_dict = find_best_results(complete_dataset_set)
    model_rank_dict, model_loss_dict = rank_result(result_dict)
    optimize_model_portfolio(result_dict)
    plot_dataset_results(result_dict, xlabel='accuracy', ylabel='datasets')
    plot_rank_result(model_dict=model_rank_dict, xlabel='relative rank', ylabel='models', invert=False)
    plot_rank_result(model_dict=model_loss_dict, xlabel='accuracy', ylabel='models', invert=True)
    plot_model_portfolio_performance()







































