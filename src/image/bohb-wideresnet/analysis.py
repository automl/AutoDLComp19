import os

import click
import hpbandster.core.result as hpres
import hpbandster.visualization as hpvis
import matplotlib.pyplot as plt


@click.command()
@click.option("--run_name", default=1, help="Directory of Hpbandster run.", type=click.STRING)
def analysis(run_name):
    # load the example run from the log files
    result = hpres.logged_results_to_HBS_result(run_name)

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
    inc_loss = inc_run.loss
    inc_config = id2conf[inc_id]['config']
    inc_test_loss = 1 - inc_run.info['test accuracy']

    print('Best found configuration:')
    print(inc_config)
    print('It achieved errors of %f (validation) and %f (test).' % (inc_loss, inc_test_loss))

    # Let's plot the observed losses grouped by budget,
    hpvis.losses_over_time(all_runs)
    plt.savefig(os.path.join(run_name, 'loss_over_time.pdf'))
    plt.close()

    # the number of concurent runs,
    hpvis.concurrent_runs_over_time(all_runs)
    plt.savefig(os.path.join(run_name, 'concurrent_runs_over_time.pdf'))
    plt.close()

    # and the number of finished runs.
    hpvis.finished_runs_over_time(all_runs)
    plt.savefig(os.path.join(run_name, 'finished_runs_over_time.pdf'))
    plt.close()

    # This one visualizes the spearman rank correlation coefficients of the losses
    # between different budgets.
    hpvis.correlation_across_budgets(result)
    plt.savefig(os.path.join(run_name, 'correlation_across_budgets.pdf'))
    plt.close()

    if "random" or "RS" not in run_name:
        # For model based optimizers, one might wonder how much the model actually helped.
        # The next plot compares the performance of configs picked by the model vs. random ones
        hpvis.performance_histogram_model_vs_random(all_runs, id2conf)
        plt.savefig(os.path.join(run_name, 'performance_histogram_model_vs_random.pdf'))
    plt.close()


if __name__ == '__main__':
    analysis()
