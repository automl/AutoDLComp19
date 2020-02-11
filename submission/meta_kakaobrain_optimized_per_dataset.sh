#!/bin/bash

#SBATCH -p bosch_gpu-rtx2080
#SBATCH -t 00-24:00
#SBATCH -a 1-6 # array size

#SBATCH --gres=gpu:5
#SBATCH --mem 16000
#SBATCH --cpus-per-task 1

#SBATCH -o experiments/kakaobrain_optimized_per_dataset_%A-%a.%x.out
#SBATCH -e experiments/kakaobrain_optimized_per_dataset_%A-%a.%x.err
#SBATCH --job-name kakaobrain_optimized_per_dataset

source ~/.miniconda/bin/activate autodl

if [ 1 -eq $SLURM_ARRAY_TASK_ID ]; then
  python src/hpo/optimize.py --experiment_name="emnist" --job_id $SLURM_ARRAY_JOB_ID --nic_name eth0
fi

if [ 2 -eq $SLURM_ARRAY_TASK_ID ]; then
  python src/hpo/optimize.py --experiment_name="emnist" --job_id $SLURM_ARRAY_JOB_ID --nic_name eth0 --worker
fi

if [ 3 -eq $SLURM_ARRAY_TASK_ID ]; then
  python src/hpo/optimize.py --experiment_name="cifar10" --job_id $SLURM_ARRAY_JOB_ID --nic_name eth0
fi

if [ 4 -eq $SLURM_ARRAY_TASK_ID ]; then
  python src/hpo/optimize.py --experiment_name="cifar10" --job_id $SLURM_ARRAY_JOB_ID --nic_name eth0 --worker
fi

if [ 5 -eq $SLURM_ARRAY_TASK_ID ]; then
  python src/hpo/optimize.py --experiment_name="caltech101" --job_id $SLURM_ARRAY_JOB_ID --nic_name eth0
fi

if [ 6 -eq $SLURM_ARRAY_TASK_ID ]; then
  python src/hpo/optimize.py --experiment_name="caltech101" --job_id $SLURM_ARRAY_JOB_ID --nic_name eth0 --worker
fi
