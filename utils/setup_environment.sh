#!/bin/bash

if [ ! -d ".miniconda/envs/" ]; then
    # Install conda
    cd .miniconda

    wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O install_miniconda.sh
    bash install_miniconda.sh -b -p . -f  # silent mode + force write in directory that exists
    rm install_miniconda.sh

    cd ..
fi

source .miniconda/bin/activate
conda update -n base -c defaults conda --yes
if [ ! -d ".miniconda/envs/autodl" ]; then
    # Install environment from scratch
    conda env create -f utils/environment.yml
else
    # Install changes according to .yml file
    conda env update -f utils/environment.yml --prune
fi
