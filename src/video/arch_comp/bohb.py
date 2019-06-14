import random
import traceback

import configuration
import hpbandster.core.nameserver as hpns
import hpbandster.core.result as hpres
import hpbandster.visualization as hpvis
import matplotlib.pyplot as plt
import train_test
from hpbandster.core.worker import Worker
from hpbandster.optimizers import BOHB as BOHB


class NasWorker(Worker):
    def __init__(self, cfg, *args, **kwargs):
        super(NasWorker, self).__init__(*args, **kwargs)
        self.cfg = cfg

    def compute(self, config, budget, *args, **kwargs):
        print("start compute")
        self.cfg["train_epochs"] = budget
        configuration.map_config_space_object_to_configuration(config, cfg)

        try:
            valid_score, train_time, status = train_test.train(self.cfg)
            print(valid_score)
        except Exception:
            status = traceback.format_exc()
            print(status)
            valid_score = 0
            train_time = 0

        print("end compute")
        return {
            "loss": -valid_score,
            "info": {
                "train_time": train_time,
                "status": status
            },
        }


def runBOHB(cfg):
    run_id = "0"

    # assign random port in the 30000-40000 range to avoid using a blocked port because of a previous improper bohb shutdown
    port = int(30000 + random.random() * 10000)

    ns = hpns.NameServer(run_id=run_id, host="127.0.0.1", port=port)
    ns.start()

    w = NasWorker(cfg=cfg, nameserver="127.0.0.1", run_id=run_id, nameserver_port=port)
    w.run(background=True)

    result_logger = hpres.json_result_logger(
        directory=cfg["bohb_log_dir"], overwrite=True
    )

    bohb = BOHB(
        configspace=configuration.get_configspace(),
        run_id=run_id,
        min_budget=cfg["bohb_min_budget"],
        max_budget=cfg["bohb_max_budget"],
        nameserver="127.0.0.1",
        nameserver_port=port,
        result_logger=result_logger,
    )

    res = bohb.run(n_iterations=cfg["bohb_iterations"])
    bohb.shutdown(shutdown_workers=True)
    ns.shutdown()

    return res


def visualizeBOHB(cfg):
    # load the example run from the log files
    result = hpres.logged_results_to_HBS_result(cfg["bohb_log_dir"])

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
    inc_train_time = inc_run.info["train_time"]

    # TODO: run on test dataset
    print("Best found configuration:")
    print(inc_config)
    print("It achieved accuracies of %f (validation)." % (-inc_valid_score))

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

    plt.show()


if __name__ == "__main__":
    cfg = configuration.get_configuration()
    res = runBOHB(cfg)
    visualizeBOHB(cfg)
