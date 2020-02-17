#!/bin/bash

#SBATCH -p bosch_gpu-rtx2080
#SBATCH -t 0-01:00
#SBATCH -a 1-115

#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task 1

#SBATCH -o experiments/kakaobrain_datasets_x_configs_%A-%a.%x.out
#SBATCH -e experiments/kakaobrain_datasets_x_configs_%A-%a.%x.err

#SBATCH --job-name def_cfg

source ~/.miniconda/bin/activate autodl

ARGS_FILE=submission/dataset_x_configs_v4.args
TASK_SPECIFIC_ARGS=$(sed "${SLURM_ARRAY_TASK_ID}q;d" $ARGS_FILE)

python -m src.competition.run_local_test \
    --code_dir src \
    --experiment_group kakaobrain_optimized_per_dataset_datasets_x_configs_evaluations \
    --time_budget 1200 \
    --time_budget_approx 300 \
    $TASK_SPECIFIC_ARGS
