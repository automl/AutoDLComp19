#!/bin/bash

#SBATCH -p meta_gpu-x
#SBATCH -t 0-01:00

#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task 1
#SBATCH --mem 26000

#SBATCH -o experiments/cluster_oe/%j.%x.out
#SBATCH -e experiments/cluster_oe/%j.%x.err
#SBATCH --job-name test_run

source miniconda/bin/activate autocv

python -c "print('Hello World')"
