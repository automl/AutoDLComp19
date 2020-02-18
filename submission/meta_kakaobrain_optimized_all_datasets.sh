#!/bin/bash

#SBATCH -p bosch_gpu-rtx2080
#SBATCH -t 2-00:00
#SBATCH -a 1-2

#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task 1

#SBATCH -o experiments/kakaobrain_optimized_all_datasets_%A-%a.%x.out
#SBATCH -e experiments/kakaobrain_optimized_all_datasets_%A-%a.%x.err

#SBATCH --job-name kb_all_video

source ~/.miniconda/bin/activate autodl

ARGS_FILE=submission/hpo_args_generalist_v1.args
TASK_SPECIFIC_ARGS=$(sed "${SLURM_ARRAY_TASK_ID}q;d" $ARGS_FILE)

python src/hpo/optimize.py \
    --nic_name eth0 \
    --job_id $SLURM_ARRAY_JOB_ID \
    --experiment_group kakaobrain_optimized_all_datasets \
    --time_budget 1200 \
    --time_budget_approx 90 \
    --n_repeat 15 \
    $TASK_SPECIFIC_ARGS
