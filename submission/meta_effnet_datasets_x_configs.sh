#!/bin/bash

#SBATCH -p bosch_gpu-rtx2080
#SBATCH -a 1-25000
#SBATCH -t 0-00:10

#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task 1
#SBATCH -x mlgpu15, mlgpu14

#SBATCH -o experiments/effnet_datasets_x_configs_%A-%a.%x.out
#SBATCH -e experiments/effnet_datasets_x_configs_%A-%a.%x.err

#SBATCH --job-name 20x2450

source ~/.miniconda/bin/activate autodl

ARGS_FILE=submission/hpo_args_individualists_new_cs_new_data_1.args
TASK_SPECIFIC_ARGS=$(sed "${SLURM_ARRAY_TASK_ID}q;d" $ARGS_FILE)

python -m src.competition.run_local_test \
    --code_dir src \
    --experiment_group effnet_optimized_per_dataset_datasets_x_configs_evaluations_new_cs_new_data \
    --time_budget 1200 \
    --time_budget_approx 120 \
    $TASK_SPECIFIC_ARGS
