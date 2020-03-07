#!/bin/bash

#SBATCH -p bosch_cpu-cascadelake
#SBATCH -a 1-28

#SBATCH --mem 32000
#SBATCH --cpus-per-task 1

#SBATCH -o experiments/generate_new_datasets_%A-%a.%x.out
#SBATCH -e experiments/generate_new_datasets_%A-%a.%x.err

#SBATCH --job-name generate_new_data

source ~/.miniconda/bin/activate icgen
export PYTHONPATH="/home/ferreira/Projects/AutoDL/autodl-contrib:$PYTHONPATH"
export PYTHONPATH="/home/ferreira/Projects/AutoDL/autodl-contrib/utils:$PYTHONPATH"
export PYTHONPATH="/home/ferreira/Projects/AutoDL/autodl-contrib/utils/image:$PYTHONPATH"
export PYTHONPATH="/home/ferreira/Projects/AutoDL/autodl-contrib/utils/video:$PYTHONPATH"

python ../autodl-contrib/convert_icgen.py \
    --task_id $SLURM_ARRAY_TASK_ID
