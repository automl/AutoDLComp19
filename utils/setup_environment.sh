#!/bin/bash

if [ ! -f ".miniconda/bin/activate" ] && [ -z "$(which conda)" ]; then
    # Ensure conda is installed on the machine
    cd .miniconda

    wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O install_miniconda.sh
    bash install_miniconda.sh -b -p . -f  # silent mode + force write in directory that exists
    rm install_miniconda.sh

    cd ..
fi
if [ -f ".miniconda/bin/activate" ]; then
    # Use local miniconda installation and keep it up to date
    source .miniconda/bin/activate
    conda update -n base -c defaults conda --yes
fi

if [ -z "$(conda env list | grep autodl)" ]; then
    # Install environment from scratch
    conda env create -f .environment.yml
else
    # Install changes according to .yml file
    conda env update -f .environment.yml --prune
fi
conda activate autodl
pre-commit install
