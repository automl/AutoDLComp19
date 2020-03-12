import os
import sys

script_dir = os.path.dirname(os.path.abspath( __file__ ))
par_dir = os.path.join(script_dir, os.pardir)
root_dir = os.path.join(par_dir, os.pardir)

sys.path.append(root_dir)
os.chdir(root_dir)

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
import psutil
from hpbandster.core.worker import Worker
from hpbandster.core.master import Master
from hpbandster.optimizers.iterations import SuccessiveHalving
from hpbandster.optimizers.config_generators.bohb import BOHB as BOHB
from copy import deepcopy
from src.competition.run_local_test import run_baseline

USE_NLP = False

NLP_DATASETS = ['O1', 'O2', 'O3', 'O4', 'O5']
SPEECH_DATASETS = ['data01', 'data02', 'data03', 'data04', 'data05']

DATASET_DIRS = ['/home/dingsda/data/datasets/AutoDL_public_data',
                '/data/aad/nlp_datasets/challenge',
                '/data/aad/speech_datasets/challenge']

SEED = 41

BOHB_MIN_BUDGET = 80
BOHB_MAX_BUDGET = 640
BOHB_ETA = 2
BOHB_WORKERS = 16
BOHB_ITERATIONS = 100000

def get_configspace(use_nlp):
    cs = CS.ConfigurationSpace()

    if use_nlp:
        # nlp parameters
        cs.add_hyperparameter(CSH.UniformIntegerHyperparameter(name='max_vocab_size', lower=5000, upper=50000, log=True, default_value=20000))
        cs.add_hyperparameter(CSH.UniformIntegerHyperparameter(name='max_char_length', lower=5, upper=300, log=True, default_value=96))
        cs.add_hyperparameter(CSH.UniformIntegerHyperparameter(name='max_seq_length', lower=2, upper=100, log=True, default_value=301))

        cs.add_hyperparameter(CSH.UniformIntegerHyperparameter(name='num_epoch', lower=1, upper=3, log=False, default_value=1))
        cs.add_hyperparameter(CSH.UniformIntegerHyperparameter(name='total_num_call', lower=2, upper=40, log=True, default_value=20))
        cs.add_hyperparameter(CSH.UniformFloatHyperparameter(name='valid_ratio', lower=0.02, upper=0.2, log=True, default_value=0.1))
        cs.add_hyperparameter(CSH.UniformFloatHyperparameter(name='increase_batch_acc', lower=0.4, upper=0.9, log=False, default_value=0.65))
        cs.add_hyperparameter(CSH.UniformFloatHyperparameter(name='early_stop_auc', lower=0.65, upper=0.95, log=False, default_value=0.8))
        cs.add_hyperparameter(CSH.CategoricalHyperparameter(name='init_batch_size', choices=[16, 32, 64, 128, 256, 512], default_value=32))

        cs.add_hyperparameter(CSH.UniformFloatHyperparameter(name='chi_word_length', lower=1, upper=4, log=True, default_value=2))
        cs.add_hyperparameter(CSH.UniformIntegerHyperparameter(name='max_valid_perclass_sample', lower=20, upper=800, log=True, default_value=400))
        cs.add_hyperparameter(CSH.UniformIntegerHyperparameter(name='max_sample_train', lower=5000, upper=36000, log=True, default_value=18000))
        cs.add_hyperparameter(CSH.UniformIntegerHyperparameter(name='max_train_perclass_sample', lower=40, upper=1600, log=True, default_value=800))

        cs.add_hyperparameter(CSH.UniformFloatHyperparameter(name='lr', lower=1e-4, upper=1e-1, log=True, default_value=1e-3))
        cs.add_hyperparameter(CSH.UniformFloatHyperparameter(name='rho', lower=0.03, upper=1, log=True, default_value=0.1))

    else:
        cs.add_hyperparameter(CSH.UniformIntegerHyperparameter(name='model_first_max_run_loop', lower=0, upper=5, log=False, default_value=3))
        cs.add_hyperparameter(CSH.UniformIntegerHyperparameter(name='max_audio_duration', lower=1, upper=20, log=True, default_value=5))
        cs.add_hyperparameter(CSH.UniformIntegerHyperparameter(name='first_round_duration', lower=5, upper=40, log=True, default_value=10))
        cs.add_hyperparameter(CSH.UniformIntegerHyperparameter(name='middle_duration', lower=7, upper=60, log=True, default_value=15))
        cs.add_hyperparameter(CSH.UniformIntegerHyperparameter(name='second_round_duration', lower=15, upper=100, log=True, default_value=30))
        cs.add_hyperparameter(CSH.CategoricalHyperparameter(name='audio_sample_rate', choices=[2000, 4000, 8000, 16000, 32000], default_value=16000))
        cs.add_hyperparameter(CSH.UniformIntegerHyperparameter(name='max_frame_num', lower=350, upper=1400, log=True, default_value=700))
        cs.add_hyperparameter(CSH.CategoricalHyperparameter(name='is_cut_audio', choices=[False, True], default_value=True))
        cs.add_hyperparameter(CSH.UniformIntegerHyperparameter(name='num_mfcc', lower=48, upper=192, log=True, default_value=96))
        cs.add_hyperparameter(CSH.UniformFloatHyperparameter(name='fft_duration', lower=0.05, upper=0.2, log=True, default_value=0.1))
        cs.add_hyperparameter(CSH.UniformFloatHyperparameter(name='hop_duration', lower=0.005, upper=0.08, log=True, default_value=0.04))

        cs.add_hyperparameter(CSH.UniformIntegerHyperparameter(name='max_valid_perclass_sample', lower=100, upper=400, log=True, default_value=200))
        cs.add_hyperparameter(CSH.UniformIntegerHyperparameter(name='min_valid_per_class', lower=1, upper=3, log=False, default_value=1))

        cs.add_hyperparameter(CSH.UniformFloatHyperparameter(name='lr_attention_gru', lower=1e-4, upper=1e-1, log=True, default_value=1e-3))
        cs.add_hyperparameter(CSH.UniformFloatHyperparameter(name='lr_bilstm_attention', lower=1e-4, upper=1e-1, log=True, default_value=1e-3))
        cs.add_hyperparameter(CSH.UniformFloatHyperparameter(name='lr_cnn', lower=1e-4, upper=1e-1, log=True, default_value=1e-3))
        cs.add_hyperparameter(CSH.UniformFloatHyperparameter(name='lr_crnn', lower=1e-4, upper=1e-1, log=True, default_value=1e-3))
        cs.add_hyperparameter(CSH.UniformFloatHyperparameter(name='lr_crnn2d', lower=1e-4, upper=1e-1, log=True, default_value=1e-3))
        cs.add_hyperparameter(CSH.UniformFloatHyperparameter(name='lr_crnn2d_larger', lower=1e-4, upper=1e-1, log=True, default_value=1e-3))
        cs.add_hyperparameter(CSH.UniformFloatHyperparameter(name='lr_crnn2d_vgg', lower=1e-4, upper=1e-1, log=True, default_value=1e-3))
        cs.add_hyperparameter(CSH.UniformFloatHyperparameter(name='lr_lstm_attention', lower=1e-4, upper=1e-1, log=True, default_value=1e-3))

        cs.add_hyperparameter(CSH.UniformFloatHyperparameter(name='beta_1', lower=0.03, upper=0.2, log=True, default_value=0.1))
        cs.add_hyperparameter(CSH.UniformFloatHyperparameter(name='beta_2', lower=1e-4, upper=1e-2, log=True, default_value=1e-3))
        cs.add_hyperparameter(CSH.UniformFloatHyperparameter(name='epsilon', lower=1e-9, upper=1e-7, log=True, default_value=1e-8))
        cs.add_hyperparameter(CSH.UniformFloatHyperparameter(name='decay', lower=1e-5, upper=1e-3, log=True, default_value=1e-4))
        cs.add_hyperparameter(CSH.UniformFloatHyperparameter(name='schedule_decay', lower=1e-5, upper=5e-2, log=True, default_value=4e-3))
        cs.add_hyperparameter(CSH.CategoricalHyperparameter(name='amsgrad', choices=[False, True], default_value=True))

    return cs


def construct_model_config(cso, default_config, use_nlp):
    mc = deepcopy(default_config)

    if use_nlp:
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
        mc["autospeech"]["common"]["model_first_max_run_loop"]  = cso["model_first_max_run_loop"]
        mc["autospeech"]["common"]["max_audio_duration"]        = cso["max_audio_duration"]
        mc["autospeech"]["common"]["first_round_duration"]      = cso["first_round_duration"]
        mc["autospeech"]["common"]["middle_duration"]           = cso["middle_duration"]
        mc["autospeech"]["common"]["second_round_duration"]     = cso["second_round_duration"]
        mc["autospeech"]["common"]["audio_sample_rate"]         = cso["audio_sample_rate"]
        mc["autospeech"]["common"]["max_frame_num"]             = cso["max_frame_num"]
        mc["autospeech"]["common"]["is_cut_audio"]              = cso["is_cut_audio"]
        mc["autospeech"]["common"]["num_mfcc"]                  = cso["num_mfcc"]
        mc["autospeech"]["common"]["fft_duration"]              = cso["fft_duration"]
        mc["autospeech"]["common"]["hop_duration"]              = cso["hop_duration"]
        # seems to be the same parameter
        mc["autospeech"]["common"]["sr"]                        = cso["audio_sample_rate"]

        mc["autospeech"]["common"]["max_valid_perclass_sample"] = cso["max_valid_perclass_sample"]
        mc["autospeech"]["common"]["min_valid_per_class"]       = cso["min_valid_per_class"]

        mc["autospeech"]["common"]["lr_attention_gru"]      = cso["lr_attention_gru"]
        mc["autospeech"]["common"]["lr_bilstm_attention"]   = cso["lr_bilstm_attention"]
        mc["autospeech"]["common"]["lr_cnn"]                = cso["lr_cnn"]
        mc["autospeech"]["common"]["lr_crnn"]               = cso["lr_crnn"]
        mc["autospeech"]["common"]["lr_crnn2d"]             = cso["lr_crnn2d"]
        mc["autospeech"]["common"]["lr_crnn2d_larger"]      = cso["lr_crnn2d_larger"]
        mc["autospeech"]["common"]["lr_crnn2d_vgg"]         = cso["lr_crnn2d_vgg"]
        mc["autospeech"]["common"]["lr_lstm_attention"]     = cso["lr_lstm_attention"]

        mc["autospeech"]["common"]["beta_1"]            = cso["beta_1"]
        mc["autospeech"]["common"]["beta_2"]            = cso["beta_2"]
        mc["autospeech"]["common"]["epsilon"]           = cso["epsilon"]
        mc["autospeech"]["common"]["decay"]             = cso["decay"]
        mc["autospeech"]["common"]["schedule_decay"]    = cso["schedule_decay"]
        mc["autospeech"]["common"]["amsgrad"]           = cso["amsgrad"]

    return mc

class BOHBWorker(Worker):
    def __init__(self, working_dir, use_nlp, *args, **kwargs):
        super(BOHBWorker, self).__init__(*args, **kwargs)
        self.session = tf.Session()
        print(kwargs)
        self.working_dir = working_dir
        self.use_nlp = use_nlp

        with open(os.path.join(os.getcwd(), "src/configs/default_nlp_speech.yaml")) as in_stream:
            self.default_config = yaml.safe_load(in_stream)

    def compute(self, config_id, config, budget, *args, **kwargs):
        model_config = construct_model_config(config, self.default_config, self.use_nlp)
        print('----------------------------')
        print("START BOHB ITERATION")
        print('CONFIG: ' + str(config))
        print('BUDGET: ' + str(budget))
        print('MODEL CONFIG: ' + str(model_config))
        print('----------------------------')

        config_id_formatted = "_".join(map(str, config_id))
        config_experiment_dir = os.path.join(self.working_dir, config_id_formatted, str(budget))

        info = {}

        status = 'ok'
        score_list = []

        if self.use_nlp:
            datasets = NLP_DATASETS
        else:
            datasets = SPEECH_DATASETS

        for dataset in datasets:
            dataset_dir = self.get_dataset_dir(dataset)
            try:
                score_ind = run_baseline(
                    dataset_dir=dataset_dir,
                    code_dir="src",
                    experiment_dir=config_experiment_dir,
                    time_budget=budget,
                    time_budget_approx=budget,
                    overwrite=True,
                    model_config_name=None,
                    model_config=model_config)
                score_list.append(score_ind)
            except Exception as e:
                score_list.append(0.01)
                status = str(e)
                print(status)

        score = sum(np.log(score_list)) / len(score_list)

        info['config'] = str(config)
        info['model_config'] = str(model_config)
        info['score_list'] = str(score_list)
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


def get_bohb_interface():
    addrs = psutil.net_if_addrs()
    if 'eth0' in addrs.keys():
        print('FOUND eth0 INTERFACE')
        return 'eth0'
    else:
        print('FOUND lo INTERFACE')
        return 'lo'


def get_working_dir(run_id):
    return str(os.path.join(os.getcwd(), "experiments", run_id))


def runBohbParallel(id, run_id):
    # get suitable interface (eth0 or lo)
    bohb_interface = get_bohb_interface()

    # get BOHB log directory
    working_dir = get_working_dir(run_id)

    # select whether to process NLP or speech datasets
    use_nlp = 'NLP' in run_id

    # every process has to lookup the hostname
    host = hpns.nic_name_to_host(bohb_interface)

    os.makedirs(working_dir, exist_ok=True)

    if int(id) > 0:
        print('START NEW WORKER')
        time.sleep(10)
        w = BOHBWorker(host=host,
                       run_id=run_id,
                       working_dir=working_dir,
                       use_nlp=use_nlp)
        w.load_nameserver_credentials(working_directory=working_dir)
        w.run(background=False)
        exit(0)

    print('START NEW MASTER')
    ns = hpns.NameServer(run_id=run_id,
                         host=host,
                         port=0,
                         working_directory=working_dir)
    ns_host, ns_port = ns.start()

    w = BOHBWorker(host=host,
                   nameserver=ns_host,
                   nameserver_port=ns_port,
                   run_id=run_id,
                   working_dir=working_dir,
                   use_nlp=use_nlp)
    w.run(background=True)

    result_logger = hpres.json_result_logger(directory=working_dir,
                                             overwrite=True)

    bohb = BohbWrapper(
        configspace=get_configspace(use_nlp),
        run_id=run_id,
        eta=BOHB_ETA,
        host=host,
        nameserver=ns_host,
        nameserver_port=ns_port,
        min_budget=BOHB_MIN_BUDGET,
        max_budget=BOHB_MAX_BUDGET,
        result_logger=result_logger)

    res = bohb.run(n_iterations=BOHB_ITERATIONS,
                   min_n_workers=BOHB_WORKERS)
#    res = bohb.run(n_iterations=BOHB_ITERATIONS)

    bohb.shutdown(shutdown_workers=True)
    ns.shutdown()

    return res


def runBohbSerial(run_id):
    # get BOHB log directory
    working_dir = get_working_dir(run_id)

    # assign random port in the 30000-40000 range to avoid using a blocked port because of a previous improper bohb shutdown
    port = int(30000 + random.random() * 10000)

    # select whether to process NLP or speech datasets
    use_nlp = 'NLP' in run_id

    ns = hpns.NameServer(run_id=run_id, host="127.0.0.1", port=port)
    ns.start()

    w = BOHBWorker(nameserver="127.0.0.1",
                   run_id=run_id,
                   nameserver_port=port,
                   working_dir=working_dir,
                   use_nlp=use_nlp)
    w.run(background=True)

    result_logger = hpres.json_result_logger(directory=working_dir,
                                             overwrite=True)

    bohb = BohbWrapper(
        configspace=get_configspace(use_nlp),
        run_id=run_id,
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

    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            print(arg)
        res = runBohbParallel(id=sys.argv[1], run_id=sys.argv[2])
    else:
        res = runBohbSerial(run_id='NL1P')


