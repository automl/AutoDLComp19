import os
import sys
import tensorflow as tf
import random
import numpy as np
import time
import yaml
import torch
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
import hpbandster.core.nameserver as hpns
import hpbandster.core.result as hpres
from hpbandster.core.worker import Worker
from hpbandster.core.master import Master
from hpbandster.optimizers.iterations import SuccessiveHalving
from hpbandster.optimizers.config_generators.bohb import BOHB as BOHB
from src.competition.run_local_test import run_baseline
from copy import deepcopy


USE_NLP = True

NLP_DATASETS = ['O1', 'O2', 'O3', 'O4', 'O5']
SPEECH_DATASETS = ['data01', 'data02', 'data03', 'data04', 'data05']

DATASET_DIRS = ['/home/dingsda/data/datasets/AutoDL_public_data']

SEED = 41

BOHB_MIN_BUDGET = 90
BOHB_MAX_BUDGET = 360
BOHB_ETA = 2
BOHB_WORKERS = 10
BOHB_ITERATIONS = 100
BOHB_INTERFACE = 'eth0'
BOHB_RUN_ID = '123'
BOHB_WORKING_DIR = str(os.path.join(os.getcwd(), "experiments", BOHB_RUN_ID))


def get_configspace():
    cs = CS.ConfigurationSpace()

    if USE_NLP:
        # nlp parameters
        cs.add_hyperparameter(CSH.UniformIntegerHyperparameter(name='max_vocab_size', lower=5000, upper=50000, log=True))
        cs.add_hyperparameter(CSH.UniformIntegerHyperparameter(name='max_char_length', lower=30, upper=300, log=True))
        cs.add_hyperparameter(CSH.UniformIntegerHyperparameter(name='max_seq_length', lower=100, upper=900, log=True))

        cs.add_hyperparameter(CSH.UniformIntegerHyperparameter(name='num_epoch', lower=1, upper=3, log=False))
        cs.add_hyperparameter(CSH.UniformIntegerHyperparameter(name='total_num_call', lower=10, upper=40, log=True))
        cs.add_hyperparameter(CSH.UniformFloatHyperparameter(name='valid_ratio', lower=0.05, upper=0.2, log=True))
        cs.add_hyperparameter(CSH.UniformFloatHyperparameter(name='increase_batch_acc', lower=0.4, upper=0.9, log=False))
        cs.add_hyperparameter(CSH.UniformFloatHyperparameter(name='early_stop_auc', lower=0.6, upper=0.95, log=False))
        cs.add_hyperparameter(CSH.CategoricalHyperparameter(name='init_batch_size', choices=[8, 16, 32, 64, 128]))

        cs.add_hyperparameter(CSH.UniformFloatHyperparameter(name='chi_word_length', lower=1, upper=4, log=True))
        cs.add_hyperparameter(CSH.UniformIntegerHyperparameter(name='max_valid_perclass_sample', lower=200, upper=800, log=True))
        cs.add_hyperparameter(CSH.UniformIntegerHyperparameter(name='max_sample_train', lower=9000, upper=36000, log=True))
        cs.add_hyperparameter(CSH.UniformIntegerHyperparameter(name='max_train_perclass_sample', lower=400, upper=1600, log=True))

        cs.add_hyperparameter(CSH.UniformFloatHyperparameter(name='lr', lower=1e-4, upper=1e-2, log=True))
        cs.add_hyperparameter(CSH.UniformFloatHyperparameter(name='rho', lower=0.03, upper=0.3, log=True))
    else:
        # speech parameters (still to do)
        pass

    return cs


def construct_model_config(cso, default_config):
    mc = deepcopy(default_config)

    if USE_NLP:
        mc["autonlp"]["common"]["max_vocab_size"]       = cso["max_vocab_size"]
        mc["autonlp"]["common"]["max_char_length"]      = cso["max_char_length"]
        mc["autonlp"]["common"]["max_seq_length"]       = cso["max_seq_length"]

        mc["autonlp"]["model"]["num_epoch"]             = cso["num_epoch"]
        mc["autonlp"]["model"]["total_num_call"]        = cso["total_num_call"]
        mc["autonlp"]["model"]["valid_ratio"]           = cso["valid_ratio"]
        mc["autonlp"]["model"]["increase_batch_acc"]    = cso["increase_batch_acc"]
        mc["autonlp"]["model"]["early_stop_auc"]        = cso["early_stop_auc"]
        mc["autonlp"]["model"]["init_batch_size"]       = cso["init_batch_size"]

        mc["autonlp"]["data_manager"]["chi_word_length"]            = cso["chi_word_length"]
        mc["autonlp"]["data_manager"]["max_valid_perclass_sample"]  = cso["max_valid_perclass_sample"]
        mc["autonlp"]["data_manager"]["max_sample_train"]           = cso["max_sample_train"]
        mc["autonlp"]["data_manager"]["max_train_perclass_sample"]  = cso["max_train_perclass_sample"]

        mc["autonlp"]["optimizer"]["lr"]  = cso["lr"]
        mc["autonlp"]["optimizer"]["rho"] = cso["rho"]

    else:
        pass

    return mc

class BOHBWorker(Worker):
    def __init__(self, *args, **kwargs):
        super(BOHBWorker, self).__init__(*args, **kwargs)
        self.session = tf.Session()

        with open(os.path.join(os.getcwd(), "src/configs/default.yaml")) as in_stream:
            self.default_config = yaml.safe_load(in_stream)

    def compute(self, config_id, config, budget, *args, **kwargs):
        print('----------------------------')
        print("START BOHB ITERATION")
        print('CONFIG: ' + str(config))
        print('BUDGET: ' + str(budget))
        print('----------------------------')

        model_config = construct_model_config(config, self.default_config)
        print('!!!!!!!!!!!!!')
        print(model_config)
        print('!!!!!!!!!!!!!')

        info = {}

        status = 'ok'
        score_list = []

        if USE_NLP:
            datasets = NLP_DATASETS
        else:
            datasets = SPEECH_DATASETS

        for dataset in datasets:
            dataset_dir = self.get_dataset_dir(dataset)
            try:
                score_ind = run_baseline(
                    dataset_dir=dataset_dir,
                    code_dir="src",
                    experiment_dir=BOHB_WORKING_DIR,
                    time_budget=budget,
                    time_budget_approx=budget,
                    overwrite=True,
                    model_config_name=None,
                    model_config=model_config)
                score_list.append(score_ind)
            except Exception as e:
                score_list.append[0]
                status = str(e)
                print(status)

        score = sum(score_list) / len(score_list)

        info['config'] = str(config)
        info['model_config'] = str(model_config)
        info['status'] = status

        print('----------------------------')
        print('FINAL SCORE: ' + str(score))
        print("END BOHB ITERATION")
        print('----------------------------')


        return {
            "loss": -score,
            "info": info
        }

    def get_dataset_dir(self, dataset):
        for directory in DATASET_DIRS:
            dataset_dir = os.path.join(directory, dataset)
            if os.path.isdir(dataset_dir):
                return dataset_dir

        raise IOError("suitable dataset directory not found")


class BohbWrapper(Master):
    def __init__(self, configspace=None,
                 eta=3, min_budget=0.01, max_budget=1,
                 min_points_in_model=None, top_n_percent=15,
                 num_samples=64, random_fraction=1 / 3, bandwidth_factor=3,
                 min_bandwidth=1e-3,
                 **kwargs):
        # TODO: Proper check for ConfigSpace object!
        if configspace is None:
            raise ValueError("You have to provide a valid CofigSpace object")

        cg = BOHB(configspace=configspace,
                  min_points_in_model=min_points_in_model,
                  top_n_percent=top_n_percent,
                  num_samples=num_samples,
                  random_fraction=random_fraction,
                  bandwidth_factor=bandwidth_factor,
                  min_bandwidth=min_bandwidth
                  )

        super().__init__(config_generator=cg, **kwargs)

        # Hyperband related stuff
        self.eta = eta
        self.min_budget = min_budget
        self.max_budget = max_budget

        # precompute some HB stuff
        self.max_SH_iter = -int(np.log(min_budget / max_budget) / np.log(eta)) + 1
        self.budgets = max_budget * np.power(eta, -np.linspace(self.max_SH_iter - 1, 0,
                                                               self.max_SH_iter))

        self.config.update({
            'eta': eta,
            'min_budget': min_budget,
            'max_budget': max_budget,
            'budgets': self.budgets,
            'max_SH_iter': self.max_SH_iter,
            'min_points_in_model': min_points_in_model,
            'top_n_percent': top_n_percent,
            'num_samples': num_samples,
            'random_fraction': random_fraction,
            'bandwidth_factor': bandwidth_factor,
            'min_bandwidth': min_bandwidth
        })

    def get_next_iteration(self, iteration, iteration_kwargs={}):
        # number of 'SH rungs'
        s = self.max_SH_iter - 1
        # number of configurations in that bracket
        n0 = int(np.floor((self.max_SH_iter) / (s + 1)) * self.eta ** s)
        ns = [max(int(n0 * (self.eta ** (-i))), 1) for i in range(s + 1)]

        return (SuccessiveHalving(HPB_iter=iteration, num_configs=ns,
                                  budgets=self.budgets[(-s - 1):],
                                  config_sampler=self.config_generator.get_config,
                                  **iteration_kwargs))


def runBohbParallel(id):
    # every process has to lookup the hostname
    host = hpns.nic_name_to_host(BOHB_INTERFACE)

    cs = get_configspace()
    os.makedirs(BOHB_WORKING_DIR, exist_ok=True)

    if int(id) > 0:
        time.sleep(15)
        w = BOHBWorker(timeout=10,
                       host=host,
                       run_id=BOHB_RUN_ID)
        w.load_nameserver_credentials(working_directory=BOHB_WORKING_DIR)
        w.run(background=False)
        exit(0)

    ns = hpns.NameServer(run_id=BOHB_RUN_ID,
                         host=host,
                         port=0,
                         working_directory=BOHB_RUN_ID)
    ns_host, ns_port = ns.start()

    result_logger = hpres.json_result_logger(directory=BOHB_WORKING_DIR,
                                             overwrite=True)

    bohb = BohbWrapper(
        configspace=get_configspace(),
        run_id=BOHB_RUN_ID,
        eta=BOHB_ETA,
        host=host,
        nameserver=ns_host,
        nameserver_port=ns_port,
        min_budget=BOHB_MIN_BUDGET,
        max_budget=BOHB_MAX_BUDGET,
        result_logger=result_logger)

    res = bohb.run(n_iterations=BOHB_ITERATIONS,
                   min_n_workers=BOHB_WORKERS)
    bohb.shutdown(shutdown_workers=True)
    ns.shutdown()

    return res


def runBohbSerial():
    # assign random port in the 30000-40000 range to avoid using a blocked port because of a previous improper bohb shutdown
    port = int(30000 + random.random() * 10000)

    ns = hpns.NameServer(run_id=BOHB_RUN_ID, host="127.0.0.1", port=port)
    ns.start()

    w = BOHBWorker(nameserver="127.0.0.1",
                   run_id=BOHB_RUN_ID,
                   nameserver_port=port)
    w.run(background=True)

    result_logger = hpres.json_result_logger(directory=BOHB_WORKING_DIR,
                                             overwrite=True)

    bohb = BohbWrapper(
        configspace=get_configspace(),
        run_id=BOHB_RUN_ID,
        eta=BOHB_ETA,
        min_budget=BOHB_MIN_BUDGET,
        max_budget=BOHB_MIN_BUDGET,
        nameserver="127.0.0.1",
        nameserver_port=port,
        result_logger=result_logger)

    res = bohb.run(n_iterations=BOHB_ITERATIONS)
    bohb.shutdown(shutdown_workers=True)
    ns.shutdown()

    return res


if __name__ == "__main__":
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    tf.set_random_seed(SEED)

    # for arg in sys.argv[1:]:
    #     print(arg)
    # res = runBohbParallel(id=sys.argv[1])
    res = runBohbSerial()


