#!/bin/bash

# Initialize the variable
env=""

# Parse command-line options
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --env) env="$2"; shift ;;  # Assign the value after '--env'
        *) echo "Error: Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Check if 'env' field is provided and is either 'train' or 'inference'
if [ -z "$env" ]; then
    echo "Error: env field is required. Please specify either 'train' or 'inference'."
    exit 1
fi

# Check the value of env
if [ "$env" != "train" ] && [ "$env" != "inference" ]; then
    echo "Error: env must be 'train' or 'inference'."
    exit 1
fi

# Proceed with setup based on the value of 'env'
echo "Setting up environment for: $env"

# Further logic based on 'train' or 'inference'
if [ "$env" == "train" ]; then
    # Implement the logic for training environment setup
    echo "Installing requirements for training..."
elif [ "$env" == "inference" ]; then
    # Implement the logic for inference environment setup
    echo "Installing requirements for inference..."
fi

# Load conda environment
source ~/miniconda3/etc/profile.d/conda.sh

# Create and activate Conda virtual environment
# The Python version used has been written into the conda config
conda create --name flagscale-${env} -y
conda activate flagscale-${env}

# Navigate to requirements directory and install basic dependencies
pip install -r ../requirements/requirements-common.txt

# Used for automatic fault tolerance
# Set the path to the target Python file
SITE_PACKAGES_DIR=$(python3 -c "import site; print(site.getsitepackages()[0])")
FILE="$SITE_PACKAGES_DIR/torch/distributed/elastic/agent/server/api.py"

# Replace the code in line 894 and its surrounding lines (893 and 895)
if ! sed -i '893,895s/if num_nodes_waiting > 0:/if num_nodes_waiting > 0 and self._remaining_restarts > 0:/' "$FILE"; then
    echo "Error: Replacement failed on line 894."
    exit 1
fi

# Replace the code in line 903 and its surrounding lines (902 and 904)
if ! sed -i '902,904s/^                    self\._restart_workers(self\._worker_group)/                    self._remaining_restarts -= 1\n                    self._restart_workers(self._worker_group)/' "$FILE"; then
    echo "Error: Replacement failed on line 903."
    exit 1
fi

# Install flagscale-common: TransformerEngine
git clone -b stable https://github.com/NVIDIA/TransformerEngine.git
cd TransformerEngine
git submodule update --init --recursive
pip install .
cd ..
rm -r ./TransformerEngine

pip install -r ../requirements/requirements-dev.txt

# If env equals 'train'
if [ "${env}" == "train" ]; then
    # Navigate to requirements directory and install training dependencies
    pip install -r ../requirements/train/megatron/requirements-cuda.txt

    # apex train
    git clone https://github.com/NVIDIA/apex
    cd apex
    pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --global-option="--use-ninja" --config-settings '--build-option=--cpp_ext' --config-settings '--build-option=--cuda_ext' ./
    cd ..
    rm -r ./apex

    pip install -r ../requirements/train/megatron/requirements-lint.txt
    python -m nltk.downloader -d /root/nltk_data punkt
fi

# If env equals 'inference'
if [ "${env}" == "inference" ]; then
    # Navigate to requirements directory and install inference dependencies
    pip install -r ../vllm/requirements-build.txt
    pip install -r ../vllm/requirements-cuda.txt
    pip install -r ../vllm/requirements-common.txt

    # If the dev argument is passed, execute the following command
    if [ "$2" == "dev" ]; then
        pip install -r ../vllm/requirements-dev.txt
    fi

    MAX_JOBS=$(nproc) pip install --no-build-isolation -v ../vllm/.

    # Navigate to requirements directory and install serving dependencies
    pip install -r ../requirements/serving/requirements.txt
fi

# Clean all conda caches
conda clean --all -y