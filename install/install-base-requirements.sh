#!/bin/bash

dev=${1} # Variable argument dev

# Load conda environment
source ~/miniconda3/etc/profile.d/conda.sh

# Create and activate Conda virtual environment
conda create --name flagscale-base python=3.10.12 -y
conda activate flagscale-base

# Navigate to requirements directory and install basic dependencies
pip install -r ../requirements/requirements-common.txt

# Install flagscale-common: TransformerEngine
git clone -b stable https://github.com/NVIDIA/TransformerEngine.git
cd TransformerEngine
git submodule update --init --recursive
pip install pybind11
pip install .
cd ..
rm -r ./TransformerEngine

# If the dev argument is passed, execute the following command
if [ "$dev" == "dev" ]; then
    pip install -r ../requirements/requirements-dev.txt
fi
