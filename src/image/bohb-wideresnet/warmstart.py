"""
Example 8 - Warmstarting for MNIST
==================================

Sometimes it is desired to continue an already finished run because the optimization
requires more function evaluations. In other cases, one might wish to use results
from previous runs to speed up the optimization. This might be useful if initial
runs were done with relatively small budgets, or on only a subset of the data to
get an initial understanding of the problem.

Here we shall see how to use the results from example 5 to initialize BOHB's model.
What changed are
- the number of training points is increased from 8192 to 32768
- the number of validation points is increased from 1024 to 16384
- the mimum budget is now 3 instead of 1 because we have already quite a few runs for a small number of epochs

Note that the loaded runs will show up in the results of the new run. They are all
combined into an iteration with the index -1 and their time stamps are manipulated
such that the last run finishes at time 0 with all other times being negative.
That info can be used to filter those runs when analysing the run.

"""
import os
import pickle
import argparse

import hpbandster.core.nameserver as hpns
import hpbandster.core.result as hpres

from hpbandster.optimizers import BOHB
from workers.darts_worker import DARTSWorker as worker

import logging

logging.basicConfig(level=logging.DEBUG)

parser = argparse.ArgumentParser(description='Example 1 - sequential and local execution.')
parser.add_argument('--min_budget', type=float, help='Minimum number of epochs for training.',
                    default=38)
parser.add_argument('--max_budget', type=float, help='Maximum number of epochs for training.',
                    default=38)
parser.add_argument('--num_iterations', type=int, help='Number of iterations performed by the optimizer',
                    default=120)
parser.add_argument('--n_workers', type=int, default=8)
parser.add_argument('--array_id', type=int, default=1)
parser.add_argument('--run_id', type=int,
                    default=0,
                    help='A unique run id for this optimization run. An easy'
                         'option is to use the job id of the clusters scheduler.')
parser.add_argument('--nic_name', type=str, help='Which network interface to'
                                                 'use for communication.', default='eth1')
parser.add_argument('--working_directory', type=str, help='A directory that is'
                                                          'accessible for all processes, e.g. a NFS share.',
                    default='/home/zelaa/Thesis/bohb-wideresnet/cond/data1')
parser.add_argument('--previous_run_dir', type=str, help='A directory that'
                                                         'contains a config.json and results.json for the same'
                                                         'configuration space.',
                    default='/home/zelaa/Thesis/bohb-wideresnet/cond/data/BOHB/run_3303')
parser.add_argument('--optimizer', type=str, default='BOHB', choices=['BOHB',
                                                                      'H2BO',
                                                                      'HB',
                                                                      'RS'])

args = parser.parse_args()


def get_nic_name_from_system():
    import re
    import subprocess
    process = subprocess.Popen("ip route get 8.8.8.8".split(), stdout=subprocess.PIPE)
    output = process.stdout.read().decode()
    s = re.search(r'dev\s*(\S+)', output)
    return s.group(1)


# Every process has to lookup the hostname
args.nic_name = get_nic_name_from_system()
host = hpns.nic_name_to_host(args.nic_name)

args.working_directory = os.path.join(args.working_directory, args.optimizer,
                                      'run_' + str(args.run_id))
os.makedirs(args.working_directory, exist_ok=True)

if args.array_id != 1:
    import time

    time.sleep(5)  # short artificial delay to make sure the nameserver is already running
    w = worker(run_id=args.run_id, host=host, timeout=120)
    w.load_nameserver_credentials(working_directory=args.working_directory)
    w.run(background=False)
    exit(0)

else:
    # This example shows how to log live results. This is most useful
    # for really long runs, where intermediate results could already be
    # interesting. The core.result submodule contains the functionality to
    # read the two generated files (results.json and configs.json) and
    # create a Result object.
    result_logger = hpres.json_result_logger(directory=args.working_directory,
                                             overwrite=False)

    # Start a nameserver:
    NS = hpns.NameServer(run_id=args.run_id, host=host, port=0,
                         working_directory=args.working_directory)
    ns_host, ns_port = NS.start()

    # Start local worker
    w = worker(run_id=args.run_id, host=host, nameserver=ns_host,
               nameserver_port=ns_port, timeout=120)
    w.run(background=True)

    # Let us load the old run now to use its results to warmstart a new run with slightly
    # different budgets in terms of datapoints and epochs.
    # Note that the search space has to be identical though!
    previous_run = hpres.logged_results_to_HBS_result(args.previous_run_dir)

    # print(args.min_budget, args.max_budget)

    optim = eval(args.optimizer)

    # change this if not warmstarting
    # previous_run = hputil.logged_results_to_HBS_result(args.previous_run_dir)

    if args.optimizer != 'H2BO':
        optimizer = optim(configspace=worker.get_config_space(),
                          working_directory=args.working_directory,
                          run_id=args.run_id,
                          eta=args.eta,
                          min_budget=args.min_budget,
                          max_budget=args.max_budget,
                          host=host,
                          nameserver=ns_host,
                          nameserver_port=ns_port,
                          ping_interval=3600,
                          result_logger=result_logger
                          # previous_result=previous_run, # also this
                          # random_fraction=0 # also this
                          )
    else:
        optimizer = optim(configspace=worker.get_config_space(),
                          working_directory=args.working_directory,
                          run_id=args.run_id,
                          eta=args.eta,
                          min_budget=args.min_budget,
                          max_budget=args.max_budget,
                          fully_dimensional=False,
                          min_points_in_model=10,
                          host=host,
                          nameserver=ns_host,
                          nameserver_port=ns_port,
                          ping_interval=3600,
                          result_logger=result_logger
                          )
    # from IPython import embed
    # embed()
    res = optimizer.run(n_iterations=args.num_iterations,
                        min_n_workers=args.n_workers)

    # store results
    with open(os.path.join(args.working_directory, 'results.pkl'), 'wb') as fh:
        pickle.dump(res, fh)

    # shutdown
    optimizer.shutdown(shutdown_workers=True)
    NS.shutdown()
