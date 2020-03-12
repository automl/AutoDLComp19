import argparse
import logging
import os
import random
import sys
import time
from pathlib import Path

import hpbandster.core.nameserver as hpns
import hpbandster.core.result as hpres
import numpy as np
import tensorflow as tf
import torch
from hpbandster.optimizers import BOHB as BOHB
from src.hpo.aggregate_worker import AggregateWorker, SingleWorker, get_configspace

sys.path.append(os.getcwd())


def run_worker(args):
    time.sleep(5)  # short artificial delay to make sure the nameserver is already running

    if args.optimize_generalist:
        w = AggregateWorker(
            run_id=args.run_id,
            host=args.host,
            working_directory=args.bohb_root_path,
            n_repeat=args.n_repeat,
            has_repeats_as_budget=args.n_repeat is None,
            time_budget=args.time_budget,
            time_budget_approx=args.time_budget_approx,
            performance_matrix=args.performance_matrix
        )
    else:
        w = SingleWorker(
            run_id=args.run_id,
            host=args.host,
            working_directory=args.bohb_root_path,
            n_repeat=args.n_repeat,
            dataset=args.dataset,
            time_budget=args.time_budget,
            time_budget_approx=args.time_budget_approx
        )

    w.load_nameserver_credentials(working_directory=args.bohb_root_path)
    w.run(background=False)


def run_master(args):
    NS = hpns.NameServer(
        run_id=args.run_id, nic_name=args.nic_name, working_directory=args.bohb_root_path
    )
    ns_host, ns_port = NS.start()

    # Start a background worker for the master node
    if args.optimize_generalist:
        w = AggregateWorker(
            run_id=args.run_id,
            host=ns_host,
            nameserver=ns_host,
            nameserver_port=ns_port,
            working_directory=args.bohb_root_path,
            n_repeat=args.n_repeat,
            has_repeats_as_budget=args.n_repeat is None,
            time_budget=args.time_budget,
            time_budget_approx=args.time_budget_approx,
            performance_matrix=args.performance_matrix
        )
    else:
        w = SingleWorker(
            run_id=args.run_id,
            host=ns_host,
            nameserver=ns_host,
            nameserver_port=ns_port,
            working_directory=args.bohb_root_path,
            n_repeat=args.n_repeat,
            dataset=args.dataset,
            time_budget=args.time_budget,
            time_budget_approx=args.time_budget_approx
        )
    w.run(background=True)

    # Create an optimizer
    result_logger = hpres.json_result_logger(directory=args.bohb_root_path, overwrite=False)

    if args.previous_run_dir is not None:
        previous_result = hpres.logged_results_to_HBS_result(args.previous_run_dir)
    else:
        pervious_result = None

    logger = logging.getLogger(__file__)
    logging_level = getattr(logging, args.logger_level)
    logger.setLevel(logging_level)

    optimizer = BOHB(
        configspace=get_configspace(),
        run_id=args.run_id,
        host=ns_host,
        nameserver=ns_host,
        nameserver_port=ns_port,
        min_budget=args.n_repeat_lower_budget,
        max_budget=args.n_repeat_upper_budget,
        result_logger=result_logger,
        logger=logger,
        previous_result=previous_result
    )

    res = optimizer.run(n_iterations=args.n_iterations)

    # Shutdown
    optimizer.shutdown(shutdown_workers=True)
    NS.shutdown()


def main(args):
    args.run_id = args.job_id or args.experiment_name
    args.host = hpns.nic_name_to_host(args.nic_name)

    args.bohb_root_path = str(Path("experiments", args.experiment_group, args.experiment_name))

    args.dataset = args.experiment_name

    # Handle case of budget dictating n_repeat vs. n_repeat directly
    if args.n_repeat_lower_budget is not None and args.n_repeat_upper_budget is not None:
        args.n_repeat = None
    else:
        args.n_repeat_lower_budget = 1
        args.n_repeat_upper_budget = 1

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    tf.set_random_seed(args.seed)

    if args.worker:
        run_worker(args)
    else:
        run_master(args)


if __name__ == '__main__':
    p = argparse.ArgumentParser(
        "",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # fmt: off
    p.add_argument("--experiment_group", default="kakaobrain_optimized_all_datasets")
    p.add_argument("--experiment_name", default="dataset1")

    p.add_argument("--n_repeat_lower_budget", type=int, default=None, help="Overrides n_repeat")
    p.add_argument("--n_repeat_upper_budget", type=int, default=None, help="")
    p.add_argument("--n_repeat", type=int, default=3, help="Number of worker runs per dataset")
    p.add_argument("--job_id", default=None)
    p.add_argument("--seed", type=int, default=2, help="random seed")

    p.add_argument(
        "--n_iterations", type=int, default=100, help="Number of evaluations per BOHB run"
    )

    p.add_argument("--nic_name", default="eth0", help="The network interface to use")
    p.add_argument("--worker", action="store_true", help="Make this execution a worker server")
    p.add_argument("--performance_matrix", default=None, help="Path to the performance_matrix")
    p.add_argument(
        "--previous_run_dir", default=None, help="Path to a previous run to warmstart from"
    )
    p.add_argument(
        "--optimize_generalist",
        action="store_true",
        help="If set, optimize the average score over all datasets. "
        "Otherwise optimize individual configs per dataset"
    )

    p.add_argument(
        "--time_budget_approx",
        type=int,
        default=90,
        help="Specifies <lower_time> to simulate cutting a run with "
        "budget <actual_time> after <lower-time> seconds."
    )
    p.add_argument(
        "--time_budget",
        type=int,
        default=1200,
        help="Specifies <actual_time> (see argument --time_budget_approx"
    )

    p.add_argument(
        "--logger_level",
        type=str,
        default="INFO",
        help=
        "Sets the logger level. Choose from ['INFO', 'DEBUG', 'NOTSET', 'WARNING', 'ERROR', 'CRITICAL']"
    )

    args = p.parse_args()
    main(args)
