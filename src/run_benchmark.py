import traceback
import json
import os
from os.path import join, abspath

from utils import BASEDIR

BENCHTIME = 300


def get_configuration():
    cfg = {}
    cfg["dataset_base_dir"] = abspath(join(BASEDIR, os.pardir, 'competition', 'AutoDL_public_data'))
    cfg["datasets"] = ['Ucf101', 'Hmdb51', 'Kraut', 'Kreatur', 'Pedro', 'Hammer']
    cfg["code_dir"] = BASEDIR
    cfg["score_dir"] = abspath(join(BASEDIR, os.pardir, 'competition', 'AutoDL_scoring_output'))
    cfg["earlystop"] = BENCHTIME
    return cfg


def write_config_to_file(cfg):
    path = join(BASEDIR, 'bohb_config.json')

    with open(path, 'w') as file:
        json.dump(cfg, file)


def create_function_call(cfg, subdir):
    fc = 'python3 {}  --code_dir={} --dataset_dir={} --score_subdir={} --time_budget={}'.format(
        abspath(join(BASEDIR, os.pardir, 'competition', 'run_local_test.py')),
        cfg["code_dir"],
        cfg["dataset_dir"],
        subdir,
        BENCHTIME + 10
    )
    return fc


def read_final_score_from_file(path):
    path = join(path, 'final_score.txt')

    with open(path, "r") as file:
        score = float(file.read())

    return score


def runBENCH(cfg):
        score = 0
        info = {}

        for dataset in cfg["datasets"]:
            cfg["dataset_dir"] = join(cfg["dataset_base_dir"], dataset)
            score_subdir = "bench_" + dataset
            score_path = os.path.join(cfg["score_dir"], score_subdir)
            score_temp = 0
            try:
                print('BOHB ON DATASET: ' + str(dataset))
                # stored bohb config will be readagain in model.py
                write_config_to_file(cfg)
                # execute main function
                fc = create_function_call(cfg, score_subdir)
                os.system(fc)
                # read final score from score.py
                score_temp = read_final_score_from_file(score_path)
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


if __name__ == "__main__":
    cfg = get_configuration()
    res = runBENCH(cfg)