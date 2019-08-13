################################################################################
# Name:         Run Local Test Tool
# Author:       Zhengying Liu
# Created on:   20 Sep 2018
# Update time:  5 May 2019
# Usage: 		    python run_local_test.py -dataset_dir=<dataset_dir> -code_dir=<code_dir>

VERISION = "v20190505"
DESCRIPTION =\
"""This script allows participants to run local test of their method within the
downloaded starting kit folder (and avoid using submission quota on CodaLab). To
do this, run:
```
python run_local_test.py -dataset_dir=./AutoDL_sample_data/miniciao -code_dir=./AutoDL_sample_code_submission/
```
in the starting kit directory. If you want to test the performance of a
different algorithm on a different dataset, please specify them using respective
arguments.

If you want to use default folders (i.e. those in above command line), simply
run
```
python run_local_test.py
```
"""

# ALL INFORMATION, SOFTWARE, DOCUMENTATION, AND DATA ARE PROVIDED "AS-IS".
# ISABELLE GUYON, CHALEARN, AND/OR OTHER ORGANIZERS OR CODE AUTHORS DISCLAIM
# ANY EXPRESSED OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR ANY PARTICULAR PURPOSE, AND THE
# WARRANTY OF NON-INFRINGEMENT OF ANY THIRD PARTY'S INTELLECTUAL PROPERTY RIGHTS.
# IN NO EVENT SHALL ISABELLE GUYON AND/OR OTHER ORGANIZERS BE LIABLE FOR ANY SPECIAL,
# INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER ARISING OUT OF OR IN
# CONNECTION WITH THE USE OR PERFORMANCE OF SOFTWARE, DOCUMENTS, MATERIALS,
# PUBLICATIONS, OR INFORMATION MADE AVAILABLE FOR THE CHALLENGE.
################################################################################

# Verbosity level of logging.
# Can be: NOTSET, DEBUG, INFO, WARNING, ERROR, CRITICAL
verbosity_level = 'INFO'

import logging
import os
import shutil  # for deleting a whole directory
import time
import webbrowser
from multiprocessing import Process

import tensorflow as tf

logging.basicConfig(
    level=getattr(logging, verbosity_level),
    format='%(asctime)s %(levelname)s %(filename)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


def _HERE(*args):
    h = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(h, *args)


def get_path_to_ingestion_program(starting_kit_dir):
    return os.path.join(starting_kit_dir, 'AutoDL_ingestion_program', 'ingestion.py')


def get_path_to_scoring_program(starting_kit_dir):
    return os.path.join(starting_kit_dir, 'AutoDL_scoring_program', 'score.py')


def create_and_get_path_to_scoring_output(
    starting_kit_dir, dataset_name, budget, task_id, overwrite
):
    path = os.path.join(
        starting_kit_dir, "scoring_output",
        dataset_name + "_" + str(budget) + "_" + str(task_id)
    )
    #if os.path.exists(path):
    #    if overwrite:
    #        remove_dir(path)
    #    else:
    #        raise Exception("Output directory already exists!\n")
    #else:
    #    os.makedirs(path, exist_ok=False)
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    return path


def create_and_get_path_to_result_submission(
    starting_kit_dir, dataset_name, budget, task_id, overwrite
):
    path = os.path.join(
        starting_kit_dir, "sample_result_submission",
        dataset_name + "_" + str(budget) + "_" + str(task_id)
    )
    #if os.path.exists(path):
    #    if overwrite:
    #        remove_dir(path)
    #    else:
    #        raise Exception("Output directory already exists!\n")
    #else:
    #    os.makedirs(path, exist_ok=False)
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    return path


def remove_dir(output_dir):
    """Remove the directory `output_dir`.

  This aims to clean existing output of last run of local test.
  """
    if os.path.isdir(output_dir):
        logging.info("Cleaning existing output directory of last run: {}"\
                    .format(output_dir))
        shutil.rmtree(output_dir)


def get_basename(path):
    if len(path) == 0:
        return ""
    if path[-1] == os.sep:
        path = path[:-1]
    return path.split(os.sep)[-1]


def run_baseline(
    dataset_dir,
    code_dir,
    time_budget=7200,
    dataset_name='cifar10',
    task_id=0,
    overwrite=False
):
    # Current directory containing this script
    starting_kit_dir = os.path.dirname(os.path.realpath(__file__))
    path_ingestion = get_path_to_ingestion_program(starting_kit_dir)
    path_scoring = get_path_to_scoring_program(starting_kit_dir)
    score_dir = create_and_get_path_to_scoring_output(
        starting_kit_dir, dataset_name, time_budget, task_id, overwrite
    )
    ingestion_output_dir = create_and_get_path_to_result_submission(
        starting_kit_dir, dataset_name, time_budget, task_id, overwrite
    )

    # Run ingestion and scoring at the same time
    command_ingestion = "python {} --dataset_dir={} --code_dir={} --time_budget={} --output_dir {} --score_dir {}".format(
        path_ingestion, dataset_dir, code_dir, time_budget, ingestion_output_dir,
        score_dir
    )
    command_scoring = "python {} --solution_dir={} --time_budget={} --prediction_dir {} --score_dir {}".format(
        path_scoring, dataset_dir, time_budget, ingestion_output_dir, score_dir
    )

    def run_ingestion():
        os.system(command_ingestion)

    def run_scoring():
        os.system(command_scoring)

    ingestion_process = Process(name="ingestion", target=run_ingestion)
    scoring_process = Process(name="scoring", target=run_scoring)
    #ingestion_output_dir = os.path.join(starting_kit_dir, "sample_result_submission")
    #score_dir = os.path.join(starting_kit_dir, "scoring_output")
    #remove_dir(ingestion_output_dir)
    #remove_dir(score_dir)
    ingestion_process.start()
    scoring_process.start()
    detailed_results_page = os.path.join(
        starting_kit_dir, "scoring_output", "detailed_results.html"
    )
    detailed_results_page = os.path.abspath(detailed_results_page)

    # Open detailed results page in a browser
    time.sleep(2)
    for i in range(30):
        if os.path.isfile(detailed_results_page):
            webbrowser.open("file://" + detailed_results_page, new=2)
            break
            time.sleep(1)


if __name__ == '__main__':
    default_starting_kit_dir = _HERE()
    # The default dataset is 'miniciao' under the folder AutoDL_sample_data/
    default_dataset_dir = os.path.join(
        default_starting_kit_dir, 'AutoDL_public_data', 'Munster'
    )
    default_code_dir = os.path.join(default_starting_kit_dir, 'models')
    default_time_budget = 1200

    tf.flags.DEFINE_string(
        'dataset_dir', default_dataset_dir,
        "Directory containing the content (e.g. adult.data/ + "
        "adult.solution) of an AutoDL dataset. Specify this "
        "argument if you want to test on a different dataset."
    )

    tf.flags.DEFINE_string(
        'code_dir', default_code_dir,
        "Directory containing a `model.py` file. Specify this "
        "argument if you want to test on a different algorithm."
    )

    tf.flags.DEFINE_float(
        'time_budget', default_time_budget,
        "Time budget for running ingestion " + "(training + prediction)."
    )

    tf.flags.DEFINE_string(
        "dataset_name",
        "cifar10",
        "determines the directory where to log scores",
    )

    tf.flags.DEFINE_integer(
        "task_id",
        0,
        "index for different experiments on same dir",
    )

    tf.flags.DEFINE_bool(
        "overwrite",
        False,
        "overwrite logs or not",
    )

    FLAGS = tf.flags.FLAGS
    dataset_dir = FLAGS.dataset_dir
    code_dir = FLAGS.code_dir
    time_budget = FLAGS.time_budget
    dataset_name = FLAGS.dataset_name
    task_id = FLAGS.task_id
    overwrite = FLAGS.overwrite
    logging.info("#" * 50)
    logging.info("Begin running local test using")
    logging.info("code_dir = {}".format(get_basename(code_dir)))
    logging.info("dataset_dir = {}".format(get_basename(dataset_dir)))
    logging.info("dataset_name = {}".format(dataset_name))
    logging.info("task_id = {}".format(task_id))
    logging.info("#" * 50)
    run_baseline(dataset_dir, code_dir, time_budget, dataset_name, task_id, overwrite)
