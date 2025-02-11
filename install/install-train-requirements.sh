#!/bin/bash

dev=${1} # Variable argument dev

# Load conda environment
source ~/miniconda3/etc/profile.d/conda.sh

# Create and activate Conda virtual environment
conda create --name flagscale-train --clone flagscale-base -y
conda activate flagscale-train

# Navigate to requirements directory and install train dependencies
pip install -r ../requirements/train/megatron/requirements-cuda.txt

# apex train
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings '--build-option=--cpp_ext' --config-settings '--build-option=--cuda_ext' ./
cd ..
rm -r ./apex

# If the dev argument is passed, execute the following command
if [ "$dev" == "dev" ]; then
    pip install -r ../requirements/train/megatron/requirements-dev.txt
    python -m nltk.downloader -d /root/nltk_data punkt
fi
