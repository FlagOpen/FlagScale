#!/bin/bash

python -m ensurepip --upgrade
python -m pip install --upgrade setuptools

# Packages that need to be installed outside of the conda environment
pip install -r ../requirements/requirements-base.txt
pip install -r ../requirements/requirements-lint.txt

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

# Load conda environment
source ~/miniconda3/etc/profile.d/conda.sh

# Create and activate Conda virtual environment
# The Python version used has been written into the conda config
if conda env list | grep -q "flagscale-${env}"; then
    # Check if the environment already exists
    echo "Conda environment 'flagscale-${env}' already exists."
else
    echo "Creating conda environment 'flagscale-${env}'..."
    conda create --name "flagscale-${env}" python=$(python --version | awk '{print $2}') -y
fi
conda activate flagscale-${env}

# Blinker 1.4 version is installed using the distutls installation tool, which is an old Python package installation method.
# When installing an updated version with pip, an attempt will be made to uninstall the old version,
# but the uninstallation will fail and manual deletion is required.
rm -r /usr/lib/python3/dist-packages/blinker
rm /usr/lib/python3/dist-packages/blinker-1.4.egg-info

# Exit immediately if any command fails
set -e

# This command updates `setuptools` to the latest version, ensuring compatibility and access to the latest features for Python package management.
pip install --upgrade setuptools

# Navigate to requirements directory and install basic dependencies
pip install -r ../requirements/requirements-common.txt

# TransformerEngine
# Megatron-LM requires TE >= 2.1.0.
git clone --recursive https://github.com/NVIDIA/TransformerEngine.git
cd TransformerEngine
pip install .                   # Build and install
cd ..
rm -r ./TransformerEngine

# Megatron-LM requires flash-attn >= 2.1.1, <= 2.7.3
cu=$(nvcc --version | grep "Cuda compilation tools" | awk '{print $5}' | cut -d '.' -f 1)
torch=$(pip show torch | grep Version | awk '{print $2}' | cut -d '+' -f 1 | cut -d '.' -f 1,2)
cp=$(python3 --version | awk '{print $2}' | awk -F. '{print $1$2}')
cxx=$(g++ --version | grep 'g++' | awk '{print $3}' | cut -d '.' -f 1)
wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3+cu${cu}torch${torch}cxx${cxx}abiFALSE-cp${cp}-cp${cp}-linux_x86_64.whl
pip install flash_attn-2.7.3+cu${cu}torch${torch}cxx${cxx}abiFALSE-cp${cp}-cp${cp}-linux_x86_64.whl
rm flash_attn-2.7.3+cu${cu}torch${torch}cxx${cxx}abiFALSE-cp${cp}-cp${cp}-linux_x86_64.whl

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
