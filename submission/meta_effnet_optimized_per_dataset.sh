#!/bin/bash

#SBATCH -p bosch_gpu-rtx2080
#SBATCH -a 1-34

#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task 1

#SBATCH -o experiments/effnet_optimized_per_dataset_test_%A-%a.%x.out
#SBATCH -e experiments/effnet_optimized_per_dataset_test_%A-%a.%x.err

#SBATCH --job-name opt_p_ds_effnet

source activate autodl

ARGS_FILE=submission/hpo_args_individualists_v1.args
TASK_SPECIFIC_ARGS=$(sed "${SLURM_ARRAY_TASK_ID}q;d" $ARGS_FILE)

PYTHONPATH=$PWD python src/hpo/optimize.py \
    --nic_name eth0 \
    --job_id $SLURM_ARRAY_JOB_ID \
    --experiment_group effnet_optimized_per_dataset_test \
    --time_budget 1200 \
    --time_budget_approx 90 \
    --n_repeat 3 \
    $TASK_SPECIFIC_ARGS
