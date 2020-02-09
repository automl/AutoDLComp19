import argparse
import logging
import pickle
import random
import time
from pathlib import Path

import hpbandster.core.nameserver as hpns
import hpbandster.core.result as hpres
import numpy as np
import tensorflow as tf
import torch
from hpbandster.optimizers import BOHB as BOHB
from src.hpo.aggregate_worker import AggregateWorker


def run_worker(args):
    time.sleep(5)  # short artificial delay to make sure the nameserver is already running
    w = AggregateWorker(run_id=args.run_id, host=args.host, working_directory=args.log_path)
    w.load_nameserver_credentials(working_directory=args.log_path)
    w.run(background=False)


def run_master(args):
    NS = hpns.NameServer(
        run_id=args.run_id, host=args.host, port=0, working_directory=args.log_path
    )
    ns_host, ns_port = NS.start()

    # Start a background worker for the master node
    w = AggregateWorker(
        run_id=args.run_id,
        host=args.host,
        nameserver=ns_host,
        nameserver_port=ns_port,
        working_directory=args.log_path,
    )
    w.run(background=True)

    # Create an optimizer
    result_logger = hpres.json_result_logger(directory=args.log_path, overwrite=False)
    optimizer = BOHB(
        configspace=AggregateWorker.get_configspace(),
        run_id=args.run_id,
        nameserver=ns_host,
        nameserver_port=ns_port,
        min_budget=1,
        max_budget=1,
        result_logger=result_logger,
    )

    res = optimizer.run(n_iterations=args.n_iterations)

    # Shutdown
    optimizer.shutdown(shutdown_workers=True)
    NS.shutdown()


def main(args):
    args.run_id = args.job_id or args.name
    args.host = hpns.nic_name_to_host(args.nic_name)

    args.log_path = str(Path("experiments", args.experiment_group, args.run_id))

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
    p.add_argument("--experiment_group", default="test")
    p.add_argument(
        "--name",
        default="test_1",
    )
    p.add_argument("--job_id", default=None)
    p.add_argument("--seed", type=int, default=2, help="random seed")
    p.add_argument("--n_iterations", type=int, default=3, help="")
    p.add_argument("--nic_name", default="lo", help="The network interface to use")
    p.add_argument("--worker", action="store_true", help="Make this execution a worker server")
    # fmt: on

    args = p.parse_args()
    main(args)
