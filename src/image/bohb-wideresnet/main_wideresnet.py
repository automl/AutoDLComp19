import os
import re
import time
import pickle
import argparse
import logging
import subprocess

from workers.resnet_worker import WideResnetWorker as worker
from hpbandster.optimizers.bohb import BOHB
from hpbandster.optimizers.h2bo import H2BO
from hpbandster.optimizers.hyperband import HyperBand as HB
from hpbandster.optimizers.randomsearch import RandomSearch as RS
import hpbandster.core.nameserver as hpns
from hpbandster.utils import *
import hpbandster.core.result as hputil

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(message)s',
                    datefmt='%H:%M:%S')

parser = argparse.ArgumentParser(description='Optimize DARTS search space with BOHB')
parser.add_argument('--array_id', type=int, default=1)
parser.add_argument('--nic_name', type=str, help='Which network interface to'
                                                 'use for communication.', default='eth1')
parser.add_argument('--n_workers', type=int, default=20)
parser.add_argument('--num_iterations', type=int, help='number of Hyperband iterations performed.',
                    default=64)
parser.add_argument('--run_id', type=int, default=0)
parser.add_argument('--working_directory', type=str, help='directory where to store the live rundata',
                    default=None)
parser.add_argument('--min_budget', type=int, default=38)
parser.add_argument('--max_budget', type=int, default=600)
parser.add_argument('--eta', type=float, default=2.5)
parser.add_argument('--optimizer', type=str, default='BOHB', choices=['BOHB',
                                                                      'H2BO',
                                                                      'HB',
                                                                      'RS'])
# parser.add_argument('--previous_run_dir', type=str, default='/home/zelaa/Thesis/bohb-wideresnet/cond/data/BOHB/run_1937')
# parser.add_argument('--random_fraction', type=float, default=0)
args = parser.parse_args()

args.working_directory = os.path.join(args.working_directory, args.optimizer,
                                      'run_' + str(args.run_id))


# Every process has to lookup the hostname
# Get the nic_name automatically from the system
def get_nic_name_from_system():
    process = subprocess.Popen("ip route get 8.8.8.8".split(), stdout=subprocess.PIPE)
    output = process.stdout.read().decode()
    s = re.search(r'dev\s*(\S+)', output)
    return s.group(1)


args.nic_name = get_nic_name_from_system()
host = hpns.nic_name_to_host(args.nic_name)

if args.array_id == 1:
    os.makedirs(args.working_directory, exist_ok=True)

    NS = hpns.NameServer(run_id=args.run_id,
                         nic_name=args.nic_name,
                         port=0,
                         working_directory=args.working_directory)
    ns_host, ns_port = NS.start()

    # BOHB is usually so cheap, that we can affort to run a worker on the
    # master node, too.
    worker = worker(  # sleep_interval=0.5,
        host=host,
        nameserver=ns_host,
        nameserver_port=ns_port,
        run_id=args.run_id)

    worker.run(background=True)

    # instantiate BOHB and run it
    result_logger = hputil.json_result_logger(directory=args.working_directory,
                                              overwrite=True)

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

    # BOHB can wait until a minimum number of workers is online before starting
    res = optimizer.run(n_iterations=args.num_iterations,
                        min_n_workers=args.n_workers)

    with open(os.path.join(args.working_directory, 'results.pkl'), 'wb') as fh:
        pickle.dump(res, fh)

    optimizer.shutdown(shutdown_workers=True)
    NS.shutdown()

else:
    time.sleep(10)

    worker = worker(  # sleep_interval=0.5,
        host=host,
        run_id=args.run_id)

    worker.load_nameserver_credentials(working_directory=args.working_directory)
    worker.run(background=False)
    exit(0)
