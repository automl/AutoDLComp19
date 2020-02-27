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


def run_worker(args, logger):
    time.sleep(5)  # short artificial delay to make sure the nameserver is already running

    if args.optimize_generalist:
        w = AggregateWorker(
            run_id=args.run_id,
            host=args.host,
            working_directory=args.bohb_root_path,
            n_repeat=args.n_repeat,
            time_budget=args.time_budget,
            time_budget_approx=args.time_budget_approx,
            logger=logger
        )
    else:
        w = SingleWorker(
            run_id=args.run_id,
            host=args.host,
            working_directory=args.bohb_root_path,
            n_repeat=args.n_repeat,
            dataset=args.dataset,
            time_budget=args.time_budget,
            time_budget_approx=args.time_budget_approx,
            logger=logger
        )

    w.load_nameserver_credentials(working_directory=args.bohb_root_path)
    w.run(background=False)


def run_master(args, logger):
    NS = hpns.NameServer(
        run_id=args.run_id, host=args.host, port=0, working_directory=args.bohb_root_path
    )
    ns_host, ns_port = NS.start()

    # Start a background worker for the master node
    if args.optimize_generalist:
        w = AggregateWorker(
            run_id=args.run_id,
            host=args.host,
            nameserver=ns_host,
            nameserver_port=ns_port,
            working_directory=args.bohb_root_path,
            n_repeat=args.n_repeat,
            time_budget=args.time_budget,
            time_budget_approx=args.time_budget_approx,
            logger=logger
        )
    else:
        w = SingleWorker(
            run_id=args.run_id,
            host=args.host,
            nameserver=ns_host,
            nameserver_port=ns_port,
            working_directory=args.bohb_root_path,
            n_repeat=args.n_repeat,
            dataset=args.dataset,
            time_budget=args.time_budget,
            time_budget_approx=args.time_budget_approx,
            logger=logger
        )
    w.run(background=True)

    # Create an optimizer
    result_logger = hpres.json_result_logger(directory=args.bohb_root_path, overwrite=False)

    optimizer = BOHB(
        configspace=get_configspace(),
        run_id=args.run_id,
        nameserver=ns_host,
        nameserver_port=ns_port,
        min_budget=1,
        max_budget=1,
        result_logger=result_logger,
        logger=logger
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

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    tf.set_random_seed(args.seed)

    logger = logging.getLogger(__file__)
    logging_level = getattr(logging, args.logger_level)
    logger.setLevel(logging_level)

    if args.worker:
        run_worker(args, logger)
    else:
        run_master(args, logger)


if __name__ == '__main__':
    p = argparse.ArgumentParser(
        "",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # fmt: off
    p.add_argument("--experiment_group", default="kakaobrain_optimized_all_datasets")
    p.add_argument("--experiment_name", default="dataset1")

    p.add_argument("--n_repeat", type=int, default=3, help="Number of worker runs per dataset")
    p.add_argument("--job_id", default=None)
    p.add_argument("--seed", type=int, default=2, help="random seed")

    p.add_argument(
        "--n_iterations", type=int, default=100, help="Number of evaluations per BOHB run"
    )

    p.add_argument("--nic_name", default="lo", help="The network interface to use")
    p.add_argument("--worker", action="store_true", help="Make this execution a worker server")
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
        "Sets the bohb master and worker logger level. Choose from ['INFO', 'DEBUG', 'NOTSET', 'WARNING', 'ERROR', 'CRITICAL']"
    )

    args = p.parse_args()
    main(args)
