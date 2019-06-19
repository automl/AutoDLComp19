import os
import sys
from opts import parser
# BOHB
import hpbandster.core.nameserver as hpns
import hpbandster.core.result as hpres
import hpbandster.visualization as hpvis
from hpbandster.optimizers import BOHB as BOHB
from bohb import ChallengeWorker
import configuration
# Troch
import torch
import torchvision
# GPU/CPU statistics
import GPUtil
import psutil
# Plotting
import matplotlib as mpl

mpl.use('Agg')  # Server plotting to Agg
GPUtil.showUtilization()  # Show GPUs
print(psutil.virtual_memory())


def main():
    ############################################################
    parser_args = parser.parse_args()
    print("------------------------------------")
    print("Environment Versions:")
    print("- Python: {}".format(sys.version))
    print("- PyTorch: {}".format(torch.__version__))
    print("- TorchVison: {}".format(torchvision.__version__))
    args_dict = parser_args.__dict__
    print("------------------------------------")
    print(parser_args.arch + " Configurations:")
    for key in args_dict.keys():
        print("- {}: {}".format(key, args_dict[key]))
    print("------------------------------------")
    ############################################################
    run_id = "bohb_run"
    os.getcwd()
    result_directory = os.path.join(parser_args.working_directory,
                                    'bob_res')
    port = 0
    nic_name = 'lo'
    host = hpns.nic_name_to_host(nic_name)
    ns = hpns.NameServer(run_id=run_id,
                         host=host,
                         port=port,
                         working_directory=parser_args.working_directory)
    ns_host, ns_port = ns.start()
    # time.sleep(2) # Wait for nameserver

    # only works if datasets are small
    workers = []
    for i in range(parser_args.bohb_workers):
        w = ChallengeWorker(
            # Nameserver params
            run_id=run_id, host=host, nameserver=ns_host,
            nameserver_port=ns_port, id=i,
            # timeout=120, sleep_interval = 0.5
        )
        w.run(background=True)
        workers.append(w)

    result_logger = hpres.json_result_logger(
        directory=result_directory, overwrite=True
    )
    cs = configuration.get_configspace(model_name=parser_args.arch)
    bohb = BOHB(
        configspace=cs,
        eta=parser_args.eta,
        run_id=run_id,
        host=host,
        nameserver=ns_host,
        nameserver_port=ns_port,
        result_logger=result_logger,
        min_budget=parser_args.min_budget,
        max_budget=parser_args.max_budget,
    )

    result = bohb.run(n_iterations=parser_args.bohb_iterations)
    bohb.shutdown(shutdown_workers=True)
    ns.shutdown()

    if result is None:
        result = hpres.logged_results_to_HBS_result(result_directory)

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
    inc_config = id2conf[inc_id]["config"]
    # inc_train_time = inc_run.info["train_time"]

    print("Best found configuration:")
    print(inc_config)
    print("It achieved accuracies of %f (validation)." % (-inc_valid_score))

    # Let's plot the observed losses grouped by budget,
    hpvis.losses_over_time(all_runs)
    mpl.pyplot.savefig(result_directory + 'losses_over_time.png')
    # the number of concurent runs,
    hpvis.concurrent_runs_over_time(all_runs, show=False)
    mpl.pyplot.savefig(result_directory + 'concurrent_runs_over_time.png')
    # and the number of finished runs.
    hpvis.finished_runs_over_time(all_runs, show=False)
    mpl.pyplot.savefig(result_directory + 'finished_runs_over_time.png')
    # This one visualizes the spearman rank correlation coefficients
    # of the losses between different budgets.
    hpvis.correlation_across_budgets(result, show=False)
    mpl.pyplot.savefig(result_directory + 'correlation_across_budgets.png')
    # For model based optimizers, one might wonder how much the model
    # actually helped.
    # The next plot compares the performance of configs picked by the model
    # vs. random ones
    hpvis.performance_histogram_model_vs_random(all_runs, id2conf, show=False)
    mpl.pyplot.savefig(
        result_directory + 'performance_histogram_model_vs_random.png')


if __name__ == '__main__':
    main()
