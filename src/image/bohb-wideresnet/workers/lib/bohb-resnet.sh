#!/bin/bash
#SBATCH -a 1-8 # array size
#SBATCH -c 2 # number of cores
#SBATCH -p gpu_tesla-P100
#SBATCH --gres=gpu:1  # reserves four GPUs
#SBATCH -D /home/siemsj/projects/randomnas/bohb-darts_to_Julien # Change working_dir
# redirect the output/error to some files
#SBATCH -o logs/log.txt # STDOUT  (the folder log has to be created prior to running or this won't work)
#SBATCH -e logs/err.txt # STDERR  (the folder log has to be created prior to running or this won't work)
#SBATCH -J resnet_bohb # sets the job name. If not specified, the file name will be used as job name

# Activate conda environment
source ~/.bashrc
conda activate pytorch1_0_1
# download the dataset before starting BOHB because of error comming from parallelism
# python -c "import torchvision.datasets as dset;dset.CIFAR10(root='workers/lib/data', train=True, download=True);dset.CIFAR10(root='workers/lib/data', train=False, download=True)"

python main_wideresnet.py --array_id $SLURM_ARRAY_TASK_ID --min_budget 200 --max_budget 200 --n_workers 8 --num_iterations 100 --eta=4 --run_id $SLURM_ARRAY_JOB_ID --working_directory $PWD/data_resnet --optimizer "BOHB"