#!/usr/bin/env python
import atexit
import json
import os
import random
import signal
import sys
import time
import traceback
from os.path import abspath, join

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
import hpbandster.core.nameserver as hpns
import hpbandster.core.result as hpres
# import hpbandster.visualization as hpvis
# import matplotlib.pyplot as plt
from hpbandster.core.worker import Worker
from hpbandster.optimizers import BOHB as BOHB
from utils import BASEDIR


def sigHandler(sig_no, sig_frame):
    sys.exit(0)


def shutdown():
    path = join(BASEDIR, 'sideload_config.json')
    os.remove(path)


def get_configspace():
    cs = CS.ConfigurationSpace()
    cs.add_hyperparameter(
        CSH.UniformFloatHyperparameter(
            name='selection.video.optim_args.lr', lower=1e-9, upper=1e-2, log=True
        )
    )
    cs.add_hyperparameter(
        CSH.CategoricalHyperparameter(
            name='selection.video.optimizer', choices=['SGD', 'Adam']
        )
    )
    cs.add_hyperparameter(
        CSH.CategoricalHyperparameter(
            name='selection.video.optim_args.nesterov', choices=[True, False]
        )
    )
    cs.add_hyperparameter(
        CSH.UniformFloatHyperparameter(
            name='selection.video.optim_args.weight_decay',
            lower=1e-2,
            upper=0.99,
            log=False
        )
    )
    return cs


def get_configuration():
    cfg = {}
    cfg["dataset_base_dir"] = abspath(
        join(BASEDIR, os.pardir, 'competition', 'AutoDL_public_data')
    )
    cfg["datasets"] = ['Ucf101', 'Kraut', 'Kreatur']  # , 'Decal', 'Hammer']
    cfg["code_dir"] = BASEDIR
    cfg["score_dir"] = abspath(
        join(BASEDIR, os.pardir, 'competition', 'AutoDL_scoring_output')
    )
    cfg["bohb_min_budget"] = 180
    cfg["bohb_max_budget"] = 600
    cfg["bohb_iterations"] = 100
    cfg["bohb_log_dir"] = abspath(
        join(
            BASEDIR, os.pardir, 'bohb_logs',
            time.strftime("%Y%m%d_%H%M%S", time.localtime())
        )
    )
    return cfg


def write_config_to_file(cfg):
    path = join(BASEDIR, 'sideload_config.json')

    with open(path, 'w') as file:
        json.dump(cfg, file)


def create_function_call(cfg, budget, subdir):
    fc = 'python3 '
    fc += abspath(join(BASEDIR, os.pardir, 'competition', 'run_local_test.py'))
    fc += ' --code_dir=' + cfg["code_dir"]
    fc += ' --dataset_dir=' + cfg["dataset_dir"]
    fc += ' --score_subdir=' + subdir
    fc += ' --time_budget=' + str(budget)
    return fc


def read_final_score_from_file(path, timeout=10):
    path = join(path, 'final_score.txt')
    t_s = time.time()
    score = 0.
    while time.time() - t_s < timeout:
        try:
            with open(path, "r") as file:
                score = float(file.read())
            break
        except FileNotFoundError:
            pass

    return score


def move_config(target_path):
    os.rename(
        join(BASEDIR, 'sideload_config.json'), join(target_path, 'sideload_config.json')
    )


class BOHBWorker(Worker):
    def __init__(self, cfg, *args, **kwargs):
        super(BOHBWorker, self).__init__(*args, **kwargs)
        self.cfg = cfg
        print(cfg)

    def compute(self, config, budget, *args, **kwargs):
        print("START BOHB ITERATION")
        print('CONFIG: ' + str(config))
        print('BUDGET: ' + str(budget))

        score = 0
        info = {}
        # if config['selection.video.optimizer'] == 'Adam':
        #     config['selection.video.optim_args.nesterov'] = False
        #     config['selection.video.optim_args.weight_decay'] = 0.

        for dataset in cfg["datasets"]:
            cfg["dataset_dir"] = join(cfg["dataset_base_dir"], dataset)
            score_subdir = "bohb_" + dataset
            score_path = os.path.join(cfg["score_dir"], score_subdir)
            score_temp = 0
            budget = int(budget)
            try:
                print('BOHB ON DATASET: ' + str(dataset))
                config.update(
                    {
                        'earlystop': budget,
                        'tensorboard_logging': False,
                        'profile_mem': False
                    }
                )
                # stored bohb config will be readagain in model.py
                write_config_to_file(config)
                # execute main function
                fc = create_function_call(cfg, budget, score_subdir)
                os.system(fc)
                # read final score from score.py
                score_temp = read_final_score_from_file(score_path)
            except Exception:
                status = traceback.format_exc()
                print(status)

            move_config(score_path)
            score += score_temp
            info[dataset] = score_temp

        print('FINAL SCORE: ' + str(score))
        print("END BOHB ITERATION")

        return {"loss": -score, "info": info}


def runBOHB(cfg):
    run_id = "0"

    # assign random port in the 30000-40000 range to avoid using a blocked port because of a previous improper bohb shutdown
    port = int(30000 + random.random() * 10000)

    ns = hpns.NameServer(run_id=run_id, host="127.0.0.1", port=port)
    ns.start()

    w = BOHBWorker(cfg=cfg, nameserver="127.0.0.1", run_id=run_id, nameserver_port=port)
    w.run(background=True)

    result_logger = hpres.json_result_logger(
        directory=cfg["bohb_log_dir"], overwrite=True
    )

    bohb = BOHB(
        configspace=get_configspace(),
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


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, sigHandler)
    atexit.register(shutdown)
    cfg = get_configuration()
    res = runBOHB(cfg)
