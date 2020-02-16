#!/bin/bash

#SBATCH -p bosch_gpu-rtx2080
#SBATCH -t 00-12:00
#SBATCH -a 1-2 #array size

#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task 1

#SBATCH -o experiments/kakaobrain_optimized_per_dataset_%A-%a.%x.out
#SBATCH -e experiments/kakaobrain_optimized_per_dataset_%A-%a.%x.err
#SBATCH --job-name kakaobrain_optimized_per_dataset_Katze1

source ~/.miniconda/bin/activate autodl

#if [ 1 -eq $SLURM_ARRAY_TASK_ID ]; then
#  python src/hpo/optimize.py --experiment_name="Ucf101" --job_id $SLURM_ARRAY_JOB_ID --nic_name eth0 --experiment_group kakaobrain_optimized_per_datasets
#fi

#if [ 2 -eq $SLURM_ARRAY_TASK_ID ]; then
#  python src/hpo/optimize.py --experiment_name="Ucf101" --job_id $SLURM_ARRAY_JOB_ID --nic_name eth0 --worker --experiment_group kakaobrain_optimized_per_datasets
#fi

#if [ 3 -eq $SLURM_ARRAY_TASK_ID ]; then
#  python src/hpo/optimize.py --experiment_name="SMv2" --job_id $SLURM_ARRAY_JOB_ID --nic_name eth0 --experiment_group kakaobrain_optimized_per_datasets
#fi

#if [ 4 -eq $SLURM_ARRAY_TASK_ID ]; then
#  python src/hpo/optimize.py --experiment_name="SMv2" --job_id $SLURM_ARRAY_JOB_ID --nic_name eth0 --worker --experiment_group kakaobrain_optimized_per_datasets
#fi

#if [ 5 -eq $SLURM_ARRAY_TASK_ID ]; then
#  python src/hpo/optimize.py --experiment_name="Hmdb51" --job_id $SLURM_ARRAY_JOB_ID --nic_name eth0 --experiment_group kakaobrain_optimized_per_datasets
#fi

#if [ 6 -eq $SLURM_ARRAY_TASK_ID ]; then
#  python src/hpo/optimize.py --experiment_name="Hmdb51" --job_id $SLURM_ARRAY_JOB_ID --nic_name eth0 --worker --experiment_group kakaobrain_optimized_per_datasets
#fi

#if [ 7 -eq $SLURM_ARRAY_TASK_ID ]; then
#  python src/hpo/optimize.py --experiment_name="Kreatur" --job_id $SLURM_ARRAY_JOB_ID --nic_name eth0 --experiment_group kakaobrain_optimized_per_datasets
#fi

#if [ 8 -eq $SLURM_ARRAY_TASK_ID ]; then
#  python src/hpo/optimize.py --experiment_name="Kreatur" --job_id $SLURM_ARRAY_JOB_ID --nic_name eth0 --worker --experiment_group kakaobrain_optimized_per_datasets
#fi

if [ 1 -eq $SLURM_ARRAY_TASK_ID ]; then
  python src/hpo/optimize.py --experiment_name="Katze" --job_id $SLURM_ARRAY_JOB_ID --nic_name eth0 --experiment_group kakaobrain_optimized_per_datasets
fi

if [ 2 -eq $SLURM_ARRAY_TASK_ID ]; then
  python src/hpo/optimize.py --experiment_name="Katze" --job_id $SLURM_ARRAY_JOB_ID --nic_name eth0 --worker --experiment_group kakaobrain_optimized_per_datasets
fi


###################################################################################
#if [ 1 -eq $SLURM_ARRAY_TASK_ID ]; then
#  python src/hpo/optimize.py --experiment_name="Chucky" --job_id $SLURM_ARRAY_JOB_ID --nic_name eth0 --experiment_group kakaobrain_optimized_per_dataset
#fi
#
#if [ 2 -eq $SLURM_ARRAY_TASK_ID ]; then
#  python src/hpo/optimize.py --experiment_name="Chucky" --job_id $SLURM_ARRAY_JOB_ID --nic_name eth0 --worker --experiment_group kakaobrain_optimized_per_dataset
#fi
#
#if [ 3 -eq $SLURM_ARRAY_TASK_ID ]; then
#  python src/hpo/optimize.py --experiment_name="Hammer" --job_id $SLURM_ARRAY_JOB_ID --nic_name eth0 --experiment_group kakaobrain_optimized_per_dataset
#fi
#
#if [ 4 -eq $SLURM_ARRAY_TASK_ID ]; then
#  python src/hpo/optimize.py --experiment_name="Hammer" --job_id $SLURM_ARRAY_JOB_ID --nic_name eth0 --worker --experiment_group kakaobrain_optimized_per_dataset
#fi
#
#if [ 5 -eq $SLURM_ARRAY_TASK_ID ]; then
#  python src/hpo/optimize.py --experiment_name="Munster" --job_id $SLURM_ARRAY_JOB_ID --nic_name eth0 --experiment_group kakaobrain_optimized_per_dataset
#fi
#
#if [ 6 -eq $SLURM_ARRAY_TASK_ID ]; then
#  python src/hpo/optimize.py --experiment_name="Munster" --job_id $SLURM_ARRAY_JOB_ID --nic_name eth0 --worker --experiment_group kakaobrain_optimized_per_dataset
#fi
#
#if [ 7 -eq $SLURM_ARRAY_TASK_ID ]; then
#  python src/hpo/optimize.py --experiment_name="Pedro" --job_id $SLURM_ARRAY_JOB_ID --nic_name eth0 --experiment_group kakaobrain_optimized_per_dataset
#fi
#
#if [ 8 -eq $SLURM_ARRAY_TASK_ID ]; then
#  python src/hpo/optimize.py --experiment_name="Pedro" --job_id $SLURM_ARRAY_JOB_ID --nic_name eth0 --worker --experiment_group kakaobrain_optimized_per_dataset
#fi
###################################################################################

#if [ 1 -eq $SLURM_ARRAY_TASK_ID ]; then
#  python src/hpo/optimize.py --experiment_name="emnist" --job_id $SLURM_ARRAY_JOB_ID --nic_name eth0 --experiment_group kakaobrain_optimized_per_dataset
#fi
#
#if [ 2 -eq $SLURM_ARRAY_TASK_ID ]; then
#  python src/hpo/optimize.py --experiment_name="emnist" --job_id $SLURM_ARRAY_JOB_ID --nic_name eth0 --worker --experiment_group kakaobrain_optimized_per_dataset
#fi

#if [ 3 -eq $SLURM_ARRAY_TASK_ID ]; then
#  python src/hpo/optimize.py --experiment_name="cifar10" --job_id $SLURM_ARRAY_JOB_ID --nic_name eth0 --experiment_group kakaobrain_optimized_per_dataset
#fi
#
#if [ 4 -eq $SLURM_ARRAY_TASK_ID ]; then
#  python src/hpo/optimize.py --experiment_name="cifar10" --job_id $SLURM_ARRAY_JOB_ID --nic_name eth0 --worker --experiment_group kakaobrain_optimized_per_dataset
#fi
#
#if [ 5 -eq $SLURM_ARRAY_TASK_ID ]; then
#  python src/hpo/optimize.py --experiment_name="caltech101" --job_id $SLURM_ARRAY_JOB_ID --nic_name eth0 --experiment_group kakaobrain_optimized_per_dataset
#fi
#
#if [ 6 -eq $SLURM_ARRAY_TASK_ID ]; then
#  python src/hpo/optimize.py --experiment_name="caltech101" --job_id $SLURM_ARRAY_JOB_ID --nic_name eth0 --worker --experiment_group kakaobrain_optimized_per_dataset
#fi
#
#if [ 3 -eq $SLURM_ARRAY_TASK_ID ]; then
#  python src/hpo/optimize.py --experiment_name="caltech_birds2010" --job_id $SLURM_ARRAY_JOB_ID --nic_name eth0 --experiment_group kakaobrain_optimized_per_dataset
#fi

#if [ 4 -eq $SLURM_ARRAY_TASK_ID ]; then
#  python src/hpo/optimize.py --experiment_name="caltech_birds2010" --job_id $SLURM_ARRAY_JOB_ID --nic_name eth0 --worker --experiment_group kakaobrain_optimized_per_dataset
#fi

#if [ 9 -eq $SLURM_ARRAY_TASK_ID ]; then
#  python src/hpo/optimize.py --experiment_name="caltech_birds2011" --job_id $SLURM_ARRAY_JOB_ID --nic_name eth0 --experiment_group kakaobrain_optimized_per_dataset
#fi
#
#if [ 10 -eq $SLURM_ARRAY_TASK_ID ]; then
#  python src/hpo/optimize.py --experiment_name="caltech_birds2011" --job_id $SLURM_ARRAY_JOB_ID --nic_name eth0 --worker --experiment_group kakaobrain_optimized_per_dataset
#fi
#
#if [ 11 -eq $SLURM_ARRAY_TASK_ID ]; then
#  python src/hpo/optimize.py --experiment_name="binary_alpha_digits" --job_id $SLURM_ARRAY_JOB_ID --nic_name eth0 --experiment_group kakaobrain_optimized_per_dataset
#fi
#
#if [ 12 -eq $SLURM_ARRAY_TASK_ID ]; then
#  python src/hpo/optimize.py --experiment_name="binary_alpha_digits" --job_id $SLURM_ARRAY_JOB_ID --nic_name eth0 --worker --experiment_group kakaobrain_optimized_per_dataset
#fi
#
#if [ 13 -eq $SLURM_ARRAY_TASK_ID ]; then
#  python src/hpo/optimize.py --experiment_name="cifar100" --job_id $SLURM_ARRAY_JOB_ID --nic_name eth0 --experiment_group kakaobrain_optimized_per_dataset
#fi
#
#if [ 14 -eq $SLURM_ARRAY_TASK_ID ]; then
#  python src/hpo/optimize.py --experiment_name="cifar100" --job_id $SLURM_ARRAY_JOB_ID --nic_name eth0 --worker --experiment_group kakaobrain_optimized_per_dataset
#fi
#
#if [ 15 -eq $SLURM_ARRAY_TASK_ID ]; then
#  python src/hpo/optimize.py --experiment_name="cats_vs_dogs" --job_id $SLURM_ARRAY_JOB_ID --nic_name eth0 --experiment_group kakaobrain_optimized_per_dataset
#fi
#
#if [ 16 -eq $SLURM_ARRAY_TASK_ID ]; then
#  python src/hpo/optimize.py --experiment_name="cats_vs_dogs" --job_id $SLURM_ARRAY_JOB_ID --nic_name eth0 --worker --experiment_group kakaobrain_optimized_per_dataset
#fi
#
#if [ 17 -eq $SLURM_ARRAY_TASK_ID ]; then
#  python src/hpo/optimize.py --experiment_name="colorectal_histology" --job_id $SLURM_ARRAY_JOB_ID --nic_name eth0 --experiment_group kakaobrain_optimized_per_dataset
#fi
#
#if [ 18 -eq $SLURM_ARRAY_TASK_ID ]; then
#  python src/hpo/optimize.py --experiment_name="colorectal_histology" --job_id $SLURM_ARRAY_JOB_ID --nic_name eth0 --worker --experiment_group kakaobrain_optimized_per_dataset
#fi
#
#if [ 19 -eq $SLURM_ARRAY_TASK_ID ]; then
#  python src/hpo/optimize.py --experiment_name="deep_weeds" --job_id $SLURM_ARRAY_JOB_ID --nic_name eth0 --experiment_group kakaobrain_optimized_per_dataset
#fi
#
#if [ 20 -eq $SLURM_ARRAY_TASK_ID ]; then
#  python src/hpo/optimize.py --experiment_name="deep_weeds" --job_id $SLURM_ARRAY_JOB_ID --nic_name eth0 --worker --experiment_group kakaobrain_optimized_per_dataset
#fi
#
#if [ 3 -eq $SLURM_ARRAY_TASK_ID ]; then
#  python src/hpo/optimize.py --experiment_name="eurosat" --job_id $SLURM_ARRAY_JOB_ID --nic_name eth0 --experiment_group kakaobrain_optimized_per_dataset
#fi
#
#if [ 4 -eq $SLURM_ARRAY_TASK_ID ]; then
#  python src/hpo/optimize.py --experiment_name="eurosat" --job_id $SLURM_ARRAY_JOB_ID --nic_name eth0 --worker --experiment_group kakaobrain_optimized_per_dataset
#fi

#if [ 5 -eq $SLURM_ARRAY_TASK_ID ]; then
#  python src/hpo/optimize.py --experiment_name="fashion_mnist" --job_id $SLURM_ARRAY_JOB_ID --nic_name eth0 --experiment_group kakaobrain_optimized_per_dataset
#fi

#if [ 6 -eq $SLURM_ARRAY_TASK_ID ]; then
#  python src/hpo/optimize.py --experiment_name="fashion_mnist" --job_id $SLURM_ARRAY_JOB_ID --nic_name eth0 --worker --experiment_group kakaobrain_optimized_per_dataset
#fi

#if [ 25 -eq $SLURM_ARRAY_TASK_ID ]; then
#  python src/hpo/optimize.py --experiment_name="horses_or_humans" --job_id $SLURM_ARRAY_JOB_ID --nic_name eth0 --experiment_group kakaobrain_optimized_per_dataset
#fi
#
#if [ 26 -eq $SLURM_ARRAY_TASK_ID ]; then
#  python src/hpo/optimize.py --experiment_name="horses_or_humans" --job_id $SLURM_ARRAY_JOB_ID --nic_name eth0 --worker --experiment_group kakaobrain_optimized_per_dataset
#fi
#
#if [ 27 -eq $SLURM_ARRAY_TASK_ID ]; then
#  python src/hpo/optimize.py --experiment_name="mnist" --job_id $SLURM_ARRAY_JOB_ID --nic_name eth0 --experiment_group kakaobrain_optimized_per_dataset
#fi
#
#if [ 28 -eq $SLURM_ARRAY_TASK_ID ]; then
#  python src/hpo/optimize.py --experiment_name="mnist" --job_id $SLURM_ARRAY_JOB_ID --nic_name eth0 --worker --experiment_group kakaobrain_optimized_per_dataset
#fi

