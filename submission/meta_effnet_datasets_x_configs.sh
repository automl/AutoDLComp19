#!/bin/bash

#SBATCH -p bosch_gpu-rtx2080
#SBATCH -t 0-01:00
#SBATCH -a 1-2069

#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task 1

#SBATCH -o experiments/effnet_datasets_x_configs_%A-%a.%x.out
#SBATCH -e experiments/effnet_datasets_x_configs_%A-%a.%x.err

#SBATCH --job-name effnet_def_cfg

source activate autodl

ARGS_FILE=submission/effNet_dataset_x_configs.args
TASK_SPECIFIC_ARGS=$(sed "${SLURM_ARRAY_TASK_ID}q;d" $ARGS_FILE)

python -m src.competition.run_local_test \
    --code_dir src \
    --experiment_group effnet_optimized_per_dataset_datasets_x_configs_evaluations \
    --time_budget 1200 \
    --time_budget_approx 90 \
    $TASK_SPECIFIC_ARGS
