#!/bin/bash

#SBATCH -p bosch_gpu-rtx2080
#SBATCH -t 0-24:00
#SBATCH -a 1-240

#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task 1

#SBATCH -o experiments/kakaobrain_optimized_all_datasets_%A-%a.%x.out
#SBATCH -e experiments/kakaobrain_optimized_all_datasets_%A-%a.%x.err

#SBATCH --job-name kb_all_video

source ~/.miniconda/bin/activate autodl

ARGS_FILE=submission/dataset_v1.args
TASK_SPECIFIC_ARGS=$(sed "${SLURM_ARRAY_TASK_ID}q;d" $ARGS_FILE)

python src/hpo/optimize.py \
    --nic_name eth0 \
    --job_id $SLURM_ARRAY_JOB_ID \
    --experiment_group kakaobrain_optimized_per_dataset_test \
    --time_budget 1200 \
    --time_budget_approx 90 \
    $TASK_SPECIFIC_ARGS
