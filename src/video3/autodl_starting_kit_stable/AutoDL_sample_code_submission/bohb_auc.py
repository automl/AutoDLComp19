import random
import traceback
import _pickle as pickle
import os
import time

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
import hpbandster.core.nameserver as hpns
import hpbandster.core.result as hpres
import hpbandster.visualization as hpvis
import matplotlib.pyplot as plt
from hpbandster.core.worker import Worker
from hpbandster.optimizers import BOHB as BOHB


def get_configspace():
    cs = CS.ConfigurationSpace()
    cs.add_hyperparameter(CSH.CategoricalHyperparameter(name='bn_prod_limit', choices = [256]))     # maximum value of batch_size*num_segments
    cs.add_hyperparameter(CSH.CategoricalHyperparameter(name='batch_size_train', choices = [32,64,128]))
    cs.add_hyperparameter(CSH.CategoricalHyperparameter(name='num_segments_test', choices = [2,4,8,16]))
    cs.add_hyperparameter(CSH.UniformFloatHyperparameter(name='num_segments_step', lower=1e3, upper=1e5, log=True))
    cs.add_hyperparameter(CSH.UniformFloatHyperparameter(name='dropout_diff', lower=1e-5, upper=1e-3, log=True))
    cs.add_hyperparameter(CSH.UniformFloatHyperparameter(name='t_diff', lower=0.01, upper=0.1, log=False))
    cs.add_hyperparameter(CSH.UniformFloatHyperparameter(name='lr', lower=1e-4, upper=1e-1, log=True))
    cs.add_hyperparameter(CSH.UniformFloatHyperparameter(name='lr_gamma', lower=1e-5, upper=1e-2, log=True))
    return cs

def get_configuration():
    cfg = {}
    cfg["code_dir"] = '/home/dingsda/autodl/AutoDLComp19/src/video3/autodl_starting_kit_stable/AutoDL_sample_code_submission'
    cfg["dataset_base_dir"] = '/home/dingsda/autodl/AutoDLComp19/src/video3/autodl_starting_kit_stable/datasets/'
    cfg["datasets"] = ['Katze', 'Kraut', 'Kreatur', 'Decal', 'Hammer']
    cfg["bohb_min_budget"] = 30
    cfg["bohb_max_budget"] = 300
    cfg["bohb_iterations"] = 10
    cfg["bohb_log_dir"] = "./logs/" + str(int(time.time()))
    return cfg


def write_config_to_file(cfg):
    path = os.path.join(os.getcwd(), 'bohb_config.txt')
    with open(path, 'wb') as file:
        pickle.dump(cfg, file)  # use `json.loads` to do the reverse


def create_function_call(cfg, budget):
    fc = 'python3 run_local_test.py'
    fc += ' --code_dir=' + cfg["code_dir"]
    fc += ' --dataset_dir=' + cfg["dataset_dir"]
    fc += ' --time_budget=' + str(budget)
    return fc


def read_final_score_from_file():
    path = os.path.join(os.getcwd(), 'AutoDL_scoring_output')
    path = os.path.join(path, 'final_score.txt')

    with open(path, "r") as file:
        score = float(file.read())

    return score

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

        for dataset in cfg["datasets"]:
            cfg["dataset_dir"] = os.path.join(cfg["dataset_base_dir"], dataset)

            score_temp = 0
            try:
                print('BOHB ON DATASET: ' + str(dataset))
                # stored bohb config will be readagain in model.py
                write_config_to_file(config)
                # execute main function
                fc = create_function_call(cfg, budget)
                os.system(fc)
                # read final score from score.py
                score_temp = read_final_score_from_file()
            except Exception:
                status = traceback.format_exc()
                print(status)

            score += score_temp
            info[dataset] = score_temp

        print('FINAL SCORE: ' + str(score))
        print("END BOHB ITERATION")

        return {
            "loss": -score,
            "info": info
        }


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
    cfg = get_configuration()
    res = runBOHB(cfg)
