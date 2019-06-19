import hpbandster.visualization as hpvis
import hpbandster.core.result as hpres
import argparse
import matplotlib as mpl
mpl.use('Agg')
######################################################################
# Parsing
parser = argparse.ArgumentParser(description="Plotting for BOHB")
parser.add_argument('result_path', type=str, help='Path to results.json')
parser.add_argument('output_path', type=str, default="./",
                    help='Path where the plotted images are saves')

parser_args = parser.parse_args()
result = hpres.logged_results_to_HBS_result(parser_args.result_path)
result_directory = parser_args.output_path
######################################################################
# printing incumbent

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
inc_config = id2conf[inc_id]["config"]
# inc_train_time = inc_run.info["train_time"]

print("Best found configuration:")
print(inc_config)
print("It achieved accuracies of %f (validation)." % (-inc_valid_score))
######################################################################
# Plotting
# Let's plot the observed losses grouped by budget,
hpvis.losses_over_time(all_runs)
mpl.pyplot.savefig(result_directory + 'losses_over_time.png')
# the number of concurent runs,
hpvis.concurrent_runs_over_time(all_runs, show=False)
mpl.pyplot.savefig(result_directory + 'concurrent_runs_over_time.png')
# and the number of finished runs.
hpvis.finished_runs_over_time(all_runs, show=False)
mpl.pyplot.savefig(result_directory + 'finished_runs_over_time.png')
# This one visualizes the spearman rank correlation coefficients of the losses
# between different budgets.
hpvis.correlation_across_budgets(result, show=False)
mpl.pyplot.savefig(result_directory + 'correlation_across_budgets.png')
# For model based optimizers, one might wonder how much the model
# actually helped.
# The next plot compares the performance of configs picked by the
# model vs. random ones
hpvis.performance_histogram_model_vs_random(all_runs, id2conf, show=False)
mpl.pyplot.savefig(
    result_directory + 'performance_histogram_model_vs_random.png')
