#!/bin/bash

#SBATCH -p ml_gpu-rtx2080
#SBATCH -a 1-16

#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task 3

#SBATCH -o experiments/logs_new_cs_old_data/new_cs_old_data_%A-%a.%x.out
#SBATCH -e experiments/logs_new_cs_old_data/new_cs_old_data_%A-%a.%x.err

#SBATCH --job-name nwcs_odda

source ~/.miniconda/bin/activate autodl

ARGS_FILE=submission/hpo_args_new_cs_old_data.args
TASK_SPECIFIC_ARGS=$(sed "${SLURM_ARRAY_TASK_ID}q;d" $ARGS_FILE)

python src/hpo/optimize.py \
    --nic_name eth0 \
    --job_id $SLURM_ARRAY_JOB_ID \
    --experiment_group new_cs_old_data_bohb \
    --time_budget 1200 \
    --time_budget_approx 120 \
    --n_repeat 3 \
    $TASK_SPECIFIC_ARGS
