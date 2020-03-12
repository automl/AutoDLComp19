#!/bin/bash

#SBATCH -p bosch_gpu-rtx2080

#SBATCH -a 1-55
#SBATCH -x mlgpu15

#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task 3

#SBATCH -o experiments/generalist-%A-%a.%x.out
#SBATCH -e experiments/generalist-%A-%a.%x.err

#SBATCH --job-name adl_generalist

source ~/.conda/bin/activate autocomp

ARGS_FILE=submission/hpo_args_generalist_v1.args
TASK_SPECIFIC_ARGS=$(sed "${SLURM_ARRAY_TASK_ID}q;d" $ARGS_FILE)

if [ $SLURM_ARRAY_TASK_ID -eq 1 ]
   then python -m src.hpo.optimize --job_id $SLURM_ARRAY_JOB_ID --nic_name eth0 --experiment_group generalist --time_budget 1200 --time_budget_approx 120 --n_repeat 1 --experiment_name warmstarted_from_try1 --optimize_generalist --logger_level INFO --performance_matrix ~/AutoDLComp19/experiments/new_config_space_after_March_4/effnet_optimized_per_dataset_datasets_x_configs_evaluations_new_cs_new_data_20x20/perf_matrix/perf_matrix.csv --n_iterations 200 --previous_run_dir experiments/generalist/try1/
else
   python -m src.hpo.optimize --job_id $SLURM_ARRAY_JOB_ID --nic_name eth0 --experiment_group generalist --time_budget 1200 --time_budget_approx 120 --n_repeat 1 --experiment_name warmstarted_from_try1 --optimize_generalist --worker --logger_level INFO --performance_matrix ~/AutoDLComp19/experiments/new_config_space_after_March_4/effnet_optimized_per_dataset_datasets_x_configs_evaluations_new_cs_new_data_20x20/perf_matrix/perf_matrix.csv --n_iterations 200 --previous_run_dir experiments/generalist/try1/
fi
