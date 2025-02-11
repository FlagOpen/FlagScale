#!/bin/bash

dev=${1} # Variable argument dev

# Load conda environment
source ~/miniconda3/etc/profile.d/conda.sh

# Create and activate Conda virtual environment
conda create --name flagscale-inference --clone flagscale-base -y
conda activate flagscale-inference

# Navigate to requirements directory and install inference dependencies
pip install -r ../vllm/requirements-build.txt
pip install -r ../vllm/requirements-cuda.txt
pip install -r ../vllm/requirements-common.txt

# If the dev argument is passed, execute the following command
if [ "$dev" == "dev" ]; then
    pip install -r ../vllm/requirements-dev.txt
fi

# Navigate to requirements directory and install serving dependencies
pip install -r ../requirements/serving/requirements.txt
