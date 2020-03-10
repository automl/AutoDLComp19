import os
import random
import colorsys

import numpy as np
import hpbandster.core.result as hpres
import hpbandster.visualization as hpvis

from decimal import Decimal
from ConfigSpace.read_and_write import pcs_new, json

import pickle

import matplotlib.pyplot as plt

def visualizeBOHB(log_dir):
    # load the example run from the log files
    result = hpres.logged_results_to_HBS_result(log_dir)

    # get all executed runs
    all_runs = result.get_all_runs()

    # get the 'dict' that translates config ids to the actual configurations
    id2conf = result.get_id2config_mapping()

    # Here is how you get he incumbent (best configuration)
    inc_id = result.get_incumbent_id()

    # let's grab the run on the highest budget
    inc_runs = result.get_runs_by_id(inc_id)
    inc_run = inc_runs[-1]

    # We have access to all information: the config, the loss observed during
    # optimization, and all the additional information
    inc_valid_score = inc_run.loss
    inc_config = id2conf[inc_id]['config']

    print(inc_config)

    print('Best found configuration:')
    print(inc_config)
    #print('It achieved accuracies of %f (validation) and %f (test).' % (-inc_valid_score, inc_test_score))

    # Let's plot the observed losses grouped by budget,
    hpvis.losses_over_time(all_runs)

    # the number of concurent runs,
    hpvis.concurrent_runs_over_time(all_runs)

    # and the number of finished runs.
    hpvis.finished_runs_over_time(all_runs)

    # This one visualizes the spearman rank correlation coefficients of the losses
    # between different budgets.
    hpvis.correlation_across_budgets(result)

    # For model based optimizers, one might wonder how much the model actually helped.
    # The next plot compares the performance of configs picked by the model vs. random ones
    hpvis.performance_histogram_model_vs_random(all_runs, id2conf)

    plot_accuracy_over_budget(result)

    plot_parallel_scatter(result)

    plt.show()


def plot_accuracy_over_budget(result):
    fig, ax = plt.subplots()

    # plot hyperband plot
    index = None
    color = None

    for key, value1 in result.data.items():
        if key[0] is not index:
            index = key[0]
            color = getBrightRandomColor()

        x = []
        y = []
        for key2, value2 in value1.results.items():
            try:
                x.append(key2)
                y.append(-value2["loss"])
                plt.semilogx(x,y,color=color)
                plt.ylim((0, 1))
            except:
                pass

    ax.set_title('Classification accuracy for different configurations')
    ax.set_xlabel('epochs')
    ax.set_ylabel('classification accuracy')


def plot_parallel_scatter(result):
    fig, ax = plt.subplots(dpi=300)

    ep_m = 1e9
    ep_M = -1e9

    # get all possible keys
    config_params = {}
    for value in result.data.values():
        for config_param, config_param_val in value.config.items():
            for epoch, epoch_result in value.results.items():
                try:
                    epoch_accuracy = -epoch_result["loss"]
                    ep_m = min(ep_m, epoch)
                    ep_M = max(ep_M, epoch)
                    if config_param in config_params.keys():
                        config_params[config_param].append((config_param_val, epoch, epoch_accuracy))
                    else:
                        config_params[config_param] = [(config_param_val, epoch, epoch_accuracy)]
                except:
                    pass

    index = 0
    for config_param, data in (dict(sorted(config_params.items()))).items():
        print(config_param)
        # get all unique possible values for each config parameter
        values = [elem[0] for elem in data]
        values = list(set(values))

        x_dev = 0.3
        r_min = 0.1
        r_max = 6
        alpha = 0.5
        text_x_offset = -0.2
        text_y_offset = 0.02
        size_text = 8

        # check for type (categorical,int,float,log)
        # yeah, I know that this type of loop is stupid and slow, but I had no time making it nicer...
        if type(values[0]) is bool:
            y_dev = x_dev/2
            for i in range(len(values)):
                plt.text(index+text_x_offset, values[i]+text_y_offset, str(values[i]), rotation=90, size=size_text)
                for elem in data:
                    if elem[0] == values[i]:
                        x = index + np.random.uniform(-x_dev,x_dev)
                        y = values[i] + np.random.uniform(-y_dev,y_dev)
                        ep  = elem[1]
                        acc = elem[2]
                        rad = linearInterpolation(np.log(ep), np.log(ep_m), np.log(ep_M), r_min, r_max)
                        plt.scatter(x, y, s=rad**2, c=getColor(acc), alpha=alpha, edgecolors='none')

        elif type(values[0]) is str:
            y_dev = min(1 / len(values) / 2.5, x_dev/2)
            for i in range(len(values)):
                plt.text(index+text_x_offset, i/(max(len(values)-1,1))+text_y_offset, values[i], rotation=90, size=size_text)
                for elem in data:
                    if elem[0] == values[i]:
                        x = index + np.random.uniform(-x_dev,x_dev)
                        y = i/(max(len(values)-1,1)) + np.random.uniform(-y_dev,y_dev)
                        ep  = elem[1]
                        acc = elem[2]
                        rad = linearInterpolation(np.log(ep), np.log(ep_m), np.log(ep_M), r_min, r_max)
                        plt.scatter(x, y, s=rad**2, c=getColor(acc), alpha=alpha, edgecolors='none')

        elif type(values[0]) is int:
            y_dev = min(1 / len(values) / 2.5, x_dev/2)

            plotAllStr = len(values) < 20

            if not plotAllStr:
                min_val = min(values)
                max_val = max(values)
                plt.text(index, 0, str(f"{Decimal(min_val):.2E}"), rotation=90, size=size_text)
                plt.text(index, 1, str(f"{Decimal(max_val):.2E}"), rotation=90, size=size_text)

            for i in range(len(values)):
                if plotAllStr:
                    plt.text(index, i/(max(len(values)-1,1)), str(values[i]), rotation=90, size=size_text)

                for elem in data:
                    if elem[0] == values[i]:
                        x = index + np.random.uniform(-x_dev,x_dev)
                        y = i/(max(len(values)-1,1)) + np.random.uniform(-y_dev,y_dev)
                        ep  = elem[1]
                        acc = elem[2]
                        rad = linearInterpolation(np.log(ep), np.log(ep_m), np.log(ep_M), r_min, r_max)
                        plt.scatter(x, y, s=rad**2, c=getColor(acc), alpha=alpha, edgecolors='none')

        else:
            min_val = min(values)
            max_val = max(values)

            # log scale if min/max value differs to much
            if max_val / min_val > 100:
                val050 = np.exp(linearInterpolation(0.50, 0, 1, np.log(min_val), np.log(max_val)))
                plt.text(index, 0, str(f"{Decimal(min_val):.2E}"), rotation=90, size=size_text)
                plt.text(index, 0.5, str(f"{Decimal(val050):.2E}"), rotation=90, size=size_text)
                plt.text(index, 1, str(f"{Decimal(max_val):.2E}"), rotation=90, size=size_text)
                for i in range(len(values)):
                    for elem in data:
                        if elem[0] == values[i]:
                            x = index + np.random.uniform(-x_dev, x_dev)
                            y = linearInterpolation(np.log(elem[0]), np.log(min_val), np.log(max_val), 0, 1)
                            ep = elem[1]
                            acc = elem[2]
                            rad = linearInterpolation(np.log(ep), np.log(ep_m), np.log(ep_M), r_min, r_max)
                            plt.scatter(x, y, s=rad**2, c=getColor(acc), alpha=alpha, edgecolors='none')

            # linear scale
            else:
                val050 = linearInterpolation(0.50, 0, 1, min_val, max_val)
                plt.text(index, 0, str(f"{Decimal(min_val):.2E}"), rotation=90, size=size_text)
                plt.text(index, 0.5, str(f"{Decimal(val050):.2E}"), rotation=90, size=size_text)
                plt.text(index, 1, str(f"{Decimal(max_val):.2E}"), rotation=90, size=size_text)
                for i in range(len(values)):
                    for elem in data:
                        if elem[0] == values[i]:
                            x = index + np.random.uniform(-x_dev, x_dev)
                            y = linearInterpolation(np.log(elem[0]), np.log(min_val), np.log(max_val), 0, 1)
                            ep = elem[1]
                            acc = elem[2]
                            rad = linearInterpolation(np.log(ep), np.log(ep_m), np.log(ep_M), r_min, r_max)
                            plt.scatter(x, y, s=rad**2, c=getColor(acc), alpha=alpha, edgecolors='none')

        index +=1

    plt.yticks([],[])
    plt.xticks(np.arange(index), (tuple(sorted(config_params.keys()))), rotation = 90)


def linearInterpolation(x, x0, x1, y0, y1):
    return y0 + (y1-y0)*(x-x0)/(x1-x0)


def getColor(acc):
    if acc < 0.5:
        return np.array([[1,0,0]]) + 2*acc*np.array([[0,1,0]])
    else:
        return np.array([[1, 1, 0]]) + 2 * (acc-0.5) * np.array([[-1, 0, 0]])


def getBrightRandomColor():
    h, s, l = random.random(), 1, 0.5
    return colorsys.hls_to_rgb(h, l, s)


if __name__ == '__main__':
    log_dir = '/home/dingsda/logs/NLP'
    visualizeBOHB(log_dir)



