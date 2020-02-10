#!/bin/bash

#SBATCH -p ml_gpu-rtx2080-x
#SBATCH -t 02-00:00
#SBATCH -a 1-6

#SBATCH --gres=gpu:10
#SBATCH --cpus-per-task 1
#SBATCH --mem 26000

#SBATCH -o experiments/kakaobrain_optimized_per_dataset/%A-%a.%x.out
#SBATCH -e experiments/kakaobrain_optimized_per_dataset/%A-%a.%x.err
#SBATCH --job-name kakaobrain_optimized_per_dataset

source .miniconda/bin/activate autodl

if [ $SLURM_ARRAY_TASK_ID -eq 1]
  then python src/hpo/optimize.py --experiment_name="emnist" --job_id $SLURM_ARRAY_JOB_ID --task_id $SLURM_ARRAY_TASK_ID --nic_name eth0
fi
if [ $SLURM_ARRAY_TASK_ID -eq 2]
   python src/hpo/optimize.py --experiment_name="emnist" --job_id $SLURM_ARRAY_JOB_ID --task_id $SLURM_ARRAY_TASK_ID --nic_name eth0 --worker
fi

if [ $SLURM_ARRAY_TASK_ID -eq 3]
   then python src/hpo/optimize.py --experiment_name="cifar10" --job_id $SLURM_ARRAY_JOB_ID --task_id $SLURM_ARRAY_TASK_ID --nic_name eth0
fi
if [ $SLURM_ARRAY_TASK_ID -eq 4]
  python src/hpo/optimize.py --experiment_name="cifar10" --job_id $SLURM_ARRAY_JOB_ID --task_id $SLURM_ARRAY_TASK_ID --nic_name eth0 --worker
fi

if [ $SLURM_ARRAY_TASK_ID -eq 3]
   then python src/hpo/optimize.py --experiment_name="caltech101" --job_id $SLURM_ARRAY_JOB_ID --task_id $SLURM_ARRAY_TASK_ID --nic_name eth0
fi
if [ $SLURM_ARRAY_TASK_ID -eq 4]
  python src/hpo/optimize.py --experiment_name="caltech101" --job_id $SLURM_ARRAY_JOB_ID --task_id $SLURM_ARRAY_TASK_ID --nic_name eth0 --worker
fi
