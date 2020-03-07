#!/bin/bash

#SBATCH -p bosch_gpu-rtx2080
#SBATCH -t 0-01:00
#SBATCH -a 1-30

#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task 1

#SBATCH -o experiments/effnet_datasets_x_configs_%A-%a.%x.out
#SBATCH -e experiments/effnet_datasets_x_configs_%A-%a.%x.err

#SBATCH --job-name testing_pipeline

source ~/.miniconda/bin/activate autodl

ARGS_FILE=submission/dataset_x_configs_new_datasets_v1.args
TASK_SPECIFIC_ARGS=$(sed "${SLURM_ARRAY_TASK_ID}q;d" $ARGS_FILE)

python -m src.competition.run_local_test \
    --code_dir src \
    --experiment_group effnet_optimized_per_dataset_datasets_x_configs_evaluations_new_data \
    --time_budget 1200 \
    --time_budget_approx 120 \
    $TASK_SPECIFIC_ARGS
