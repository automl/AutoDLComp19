#!/bin/bash

##SBATCH -p bosch_gpu-rtx2080
#SBATCH -p ml_gpu-rtx2080
#SBATCH -a 1-126

#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task 2
##SBATCH -x mlgpu15

#SBATCH -o experiments/logs/best_current_generalist_approx_300_%A-%a.%x.out
#SBATCH -e experiments/logs/best_current_generalist_approx_300_%A-%a.%x.err

#SBATCH --job-name bst_300

source ~/.miniconda/bin/activate autodl

ARGS_FILE=submission/best_current_generalist.args
TASK_SPECIFIC_ARGS=$(sed "${SLURM_ARRAY_TASK_ID}q;d" $ARGS_FILE)

python -m src.competition.run_local_test \
    --code_dir src \
    --experiment_group best_current_generalist_approx_300 \
    --time_budget 1200 \
    --time_budget_approx 300 \
    $TASK_SPECIFIC_ARGS
