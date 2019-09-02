################################################################################
# Name:         Run Local Test Tool
# Author:       Zhengying Liu
# Created on:   20 Sep 2018
# Update time:  5 May 2019
# Usage: 		    python run_local_test.py -dataset_dir=<dataset_dir> -code_dir=<code_dir>

VERSION = "v20190505"
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

import argparse
import atexit
import logging
import os
import shutil  # for deleting a whole directory
import signal
import sys
import time
import webbrowser

import psutil
import tensorflow as tf
from AutoDL_ingestion_program.ingestion import main as ingest

logging.basicConfig(
    level=getattr(logging, verbosity_level),
    format='%(asctime)s %(levelname)s %(filename)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


def _HERE(*args):
    h = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(h, *args)


def get_path_to_ingestion_dir(starting_kit_dir):
    return os.path.join(starting_kit_dir, 'AutoDL_ingestion_program')


def get_path_to_scoring_program(starting_kit_dir):
    return os.path.join(starting_kit_dir, 'AutoDL_scoring_program', 'score.py')


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


def kill_if_alive(p, msg):
    logging.info(msg)
    p.terminate()


def run_baseline(dataset_dir, code_dir, score_subdir, time_budget=1200):
    # Current directory containing this script
    starting_kit_dir = os.path.dirname(os.path.realpath(__file__))
    path_ingestion_dir = get_path_to_ingestion_dir(starting_kit_dir)
    path_scoring = get_path_to_scoring_program(starting_kit_dir)
    score_dir = os.path.join(starting_kit_dir, 'AutoDL_scoring_output', score_subdir)
    ingestion_output_dir = os.path.join(
        starting_kit_dir, 'AutoDL_sample_result_submission', score_subdir
    )

    # Run ingestion and scoring at the same time
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, default=dataset_dir)
    parser.add_argument('--output_dir', type=str, default=ingestion_output_dir)
    parser.add_argument('--ingestion_program_dir', type=str, default=path_ingestion_dir)
    parser.add_argument('--code_dir', type=str, default=code_dir)
    parser.add_argument('--score_dir', type=str, default=score_dir)
    parser.add_argument('--time_budget', type=float, default=time_budget)
    ingest_args = parser.parse_known_args()
    ingest_args = ingest_args[0]
    command_scoring =\
      'python {} --solution_dir={} --score_dir={} --prediction_dir={}'\
      .format(path_scoring, dataset_dir, score_dir, ingestion_output_dir)

    remove_dir(ingestion_output_dir)
    remove_dir(score_dir)

    sp = psutil.Popen([*command_scoring.split(' ')], shell=False)
    logging.info('Started score.py with PID \033[92m{}\033[0m'.format(sp.pid))
    atexit.register(
        kill_if_alive, sp, 'Terminating scoring process \033[92m{}\033[0m'.format(sp.pid)
    )
    ingest(ingest_args)
    # detailed_results_page = os.path.join(starting_kit_dir,
    #                                      'AutoDL_scoring_output',
    #                                      'detailed_results.html')
    # detailed_results_page = os.path.abspath(detailed_results_page)
    #
    # Open detailed results page in a browser
    # time.sleep(2)
    # for i in range(30):
    #   if os.path.isfile(detailed_results_page):
    #     webbrowser.open('file://'+detailed_results_page, new=2)
    #     break
    #     time.sleep(1)


def sigHandler(sig_no, sig_frame):
    logging.info('Received Signal {}'.format(sig_no))
    sys.exit(0)


if __name__ == '__main__':
    signal.signal(signal.SIGTERM, sigHandler)
    default_starting_kit_dir = _HERE()
    # The default dataset is 'miniciao' under the folder AutoDL_sample_data/
    default_dataset_dir = os.path.join(
        default_starting_kit_dir, 'AutoDL_sample_data', 'miniciao'
    )
    default_code_dir = os.path.join(
        default_starting_kit_dir, 'AutoDL_sample_code_submission'
    )
    default_score_subdir = None
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

    tf.flags.DEFINE_string(
        'score_subdir', default_score_subdir,
        "Subdirectory which will be created in the default directories"
        "for this run. If it already exits it will be emptied."
    )

    tf.flags.DEFINE_float(
        'time_budget', default_time_budget,
        "Time budget for running ingestion " + "(training + prediction)."
    )

    FLAGS = tf.flags.FLAGS
    dataset_dir = FLAGS.dataset_dir
    code_dir = FLAGS.code_dir
    score_subdir = FLAGS.score_subdir if FLAGS.score_subdir is not None else os.path.basename(
        dataset_dir
    )
    time_budget = FLAGS.time_budget
    logging.info("#" * 50)
    logging.info("Begin running local test using")
    logging.info("code_dir = {}".format(get_basename(code_dir)))
    logging.info("dataset_dir = {}".format(get_basename(dataset_dir)))
    logging.info("score_subdir = {}".format(get_basename(score_subdir)))
    logging.info("#" * 50)
    run_baseline(dataset_dir, code_dir, score_subdir, time_budget)
