#!/usr/bin/env python
import argparse  # noqa: E402
import json
import os
import shutil
import subprocess
import traceback
from os.path import abspath, join

from utils import BASEDIR

BENCHTIME = 300
BRANCHNAME = subprocess.check_output(
    ["git", "status"]
).strip().decode('utf8').split('\n')[0][10:].replace('/', '_')
COMMITHASH = subprocess.check_output(["git", "log"]).strip().decode('utf8')[8:16]


def get_configuration():
    cfg = {}
    cfg["dataset_base_dir"] = abspath(
        join(BASEDIR, os.pardir, 'competition', 'AutoDL_public_data')
    )
    cfg["code_dir"] = BASEDIR
    cfg["score_dir"] = abspath(
        join(BASEDIR, os.pardir, 'competition', 'AutoDL_scoring_output')
    )
    return cfg


def write_config_to_file():
    sideload_config = {"earlystop": BENCHTIME}
    path = join(BASEDIR, 'sideload_config.json')
    with open(path, 'w') as file:
        json.dump(sideload_config, file)


def create_function_call(cfg, subdir):
    fc = 'python3 {}  --code_dir={} --dataset_dir={} --score_subdir={} --time_budget={}'.format(
        abspath(join(BASEDIR, os.pardir, 'competition', 'run_local_test.py')),
        cfg["code_dir"], cfg["dataset_dir"], subdir, BENCHTIME
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
    write_config_to_file()  # Make sure we have the right earlystop and nothing else

    for dataset in cfg["datasets"]:
        cfg["dataset_dir"] = join(cfg["dataset_base_dir"], dataset)
        score_subdir = "bench_" + dataset
        score_path = os.path.join(cfg["score_dir"], score_subdir)
        score_temp = 0
        try:
            print('BENCH ON DATASET: ' + str(dataset))
            # stored BENCH config will be readagain in model.py
            # execute main function
            fc = create_function_call(cfg, score_subdir)
            os.system(fc)
            # read final score from score.py
            score_temp = read_final_score_from_file(score_path)
            shutil.copy(
                os.path.join(BASEDIR, 'config.hjson'),
                os.path.join(score_path, BRANCHNAME + '-' + COMMITHASH + '.hjson')
            )
        except Exception:
            status = traceback.format_exc()
            print(status)

        score += score_temp
        info[dataset] = score_temp

    print('FINAL SCORE: ' + str(score))
    print("END BENCH ITERATION")


if __name__ == "__main__":
    # Making sure nothing gets overwritten
    if os.path.isfile(os.path.join(BASEDIR, 'sideload_config.json')):
        os.remove(os.path.join(BASEDIR, 'sideload_config.json'))
    cfg = get_configuration()
    default_datasets = ['Hammer', 'Kraut', 'Kreatur', 'Pedro', 'Ucf101']

    parser = argparse.ArgumentParser()
    parser.add_argument("--time_budget", default=BENCHTIME, type=int)
    parser.add_argument("--datasets", default=default_datasets, type=str, nargs='+')
    pargs = parser.parse_args()

    BENCHTIME = pargs.time_budget
    cfg["datasets"] = pargs.datasets

    print('TIME BUDGET PER DATASET IS \033[92m{} s\033[0m'.format(BENCHTIME))
    print('\033[92m{}\033[0m'.format(cfg["datasets"]))
    res = runBENCH(cfg)
