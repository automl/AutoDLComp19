#!/bin/bash

#SBATCH -p bosch_gpu-rtx2080
#SBATCH -t 0-24:00
#SBATCH -a 1-20

#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task 1

#SBATCH -o experiments/kakaobrain_datasets_x_configs_%A-%a.%x.out
#SBATCH -e experiments/kakaobrain_datasets_x_configs_%A-%a.%x.err

#SBATCH --job-name dt_x_cf

source ~/.miniconda/bin/activate autodl

ARGS_FILE=submission/dataset_x_configs_v1.args
TASK_SPECIFIC_ARGS=$(sed "${SLURM_ARRAY_TASK_ID}q;d" $ARGS_FILE)

python -m src.competition.run_local_test \
    --code_dir src \
    --experiment_group datasets_x_configs \
    --time_budget 1200 \
    --time_budget_approx 60 \
    $TASK_SPECIFIC_ARGS
