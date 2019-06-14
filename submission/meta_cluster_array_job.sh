#!/bin/bash

#SBATCH -p meta_gpu-x
#SBATCH -t 0-01:00
#SBATCH -a 1-4
#SBATCH -w metagpub

#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task 1
#SBATCH --mem 26000

#SBATCH -o experiments/cluster_oe/%A-%a.%x.out
#SBATCH -e experiments/cluster_oe/%A-%a.%x.err
#SBATCH --job-name test_run

source .miniconda/bin/activate autodl

python competition/run_local_test.py --dataset_dir datasets/sample_data/miniciao --code_dir competition/sample_submission --job_id $SLURM_ARRAY_JOB_ID --task_id $SLURM_ARRAY_TASK_ID
