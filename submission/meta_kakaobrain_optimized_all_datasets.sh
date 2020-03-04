#!/bin/bash

#SBATCH -p bosch_gpu-rtx2080

#SBATCH -a 1-100
#SBATCH -x mlgpu05,mlgpu06,mlgpu07,mlgpu13,mlgpu14,mlgpu15

#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task 1

#SBATCH -o experiments/kakaobrain_optimized_all_datasets_new6%A-%a.%x.out
#SBATCH -e experiments/kakaobrain_optimized_all_datasets_new6%A-%a.%x.err

#SBATCH --job-name kb_all_video

source ~/.miniconda/bin/activate autodl

ARGS_FILE=submission/hpo_args_generalist_v1.args
TASK_SPECIFIC_ARGS=$(sed "${SLURM_ARRAY_TASK_ID}q;d" $ARGS_FILE)


if [ $SLURM_ARRAY_TASK_ID -eq 1 ]
   then python src/hpo/optimize.py --job_id $SLURM_ARRAY_JOB_ID --nic_name eth0 --experiment_group kakaobrain_optimized_all_datasets_new6 --time_budget 1200 --time_budget_approx 120 --n_repeat_lower_budget 1 --n_repeat_upper_budget 3 --experiment_name generalist --optimize_generalist --logger_level DEBUG
else
   python src/hpo/optimize.py --job_id $SLURM_ARRAY_JOB_ID --nic_name eth0 --experiment_group kakaobrain_optimized_all_datasets_new6 --time_budget 1200 --time_budget_approx 120 --n_repeat_lower_budget 1 --n_repeat_upper_budget 3 --experiment_name generalist --optimize_generalist --worker --logger_level DEBUG
fi
