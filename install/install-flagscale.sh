#!/bin/bash

set -e

python -m pip install --upgrade pip

# Packages that need to be installed outside of the conda environment
pip install --no-cache-dir -r ./requirements/requirements-base.txt

# Proceed with setup based on the value of 'env'
echo "Setting up environment for: flagscale"

# Load conda environment
source ~/miniconda3/etc/profile.d/conda.sh

# Create and activate Conda virtual environment
# The Python version used has been written into the conda config
if conda env list | grep -q "flagscale"; then
    # Check if the environment already exists
    echo "Conda environment 'flagscale' already exists."
else
    echo "Creating conda environment 'flagscale'..."
    conda create --name "flagscale" python=$(python --version | awk '{print $2}' | cut -d '.' -f 1,2) -y
fi
conda activate flagscale

# Exit immediately if any command fails
set -e

# This command updates `setuptools` to the latest version, ensuring compatibility and access to the latest features for Python package management.
pip install --upgrade setuptools

# Navigate to requirements directory and install basic dependencies
pip install --no-cache-dir -r ./requirements/requirements-common.txt

# TransformerEngine
# Megatron-LM requires TE >= 2.1.0.
git clone --recursive https://github.com/NVIDIA/TransformerEngine.git
cd TransformerEngine
git checkout 5bee81e
pip install .
cd ..
rm -r ./TransformerEngine

# cudnn frontend
pip install --no-cache-dir nvidia-cudnn-cu12==9.7.1.26
CMAKE_ARGS="-DCMAKE_POLICY_VERSION_MINIMUM=3.5" pip install nvidia-cudnn-frontend
python -c "import torch; print('cuDNN version:', torch.backends.cudnn.version());"
python -c "from transformer_engine.pytorch.utils import get_cudnn_version; get_cudnn_version()"

# Megatron-LM requires flash-attn >= 2.1.1, <= 2.8.0.post2
cu=$(nvcc --version | grep "Cuda compilation tools" | awk '{print $5}' | cut -d '.' -f 1)
torch=$(pip show torch | grep Version | awk '{print $2}' | cut -d '+' -f 1 | cut -d '.' -f 1,2)
cp=$(python3 --version | awk '{print $2}' | awk -F. '{print $1$2}')
cxx=$(g++ --version | grep 'g++' | awk '{print $3}' | cut -d '.' -f 1)
wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.0.post2/flash_attn-2.8.0.post2+cu${cu}torch${torch}cxx${cxx}abiFALSE-cp${cp}-cp${cp}-linux_x86_64.whl
pip install flash_attn-2.8.0.post2+cu${cu}torch${torch}cxx${cxx}abiFALSE-cp${cp}-cp${cp}-linux_x86_64.whl
rm flash_attn-2.8.0.post2+cu${cu}torch${torch}cxx${cxx}abiFALSE-cp${cp}-cp${cp}-linux_x86_64.whl

# From Megatron-LM log
pip install --no-cache-dir "git+https://github.com/Dao-AILab/flash-attention.git@v2.7.2#egg=flashattn-hopper&subdirectory=hopper"
python_path=`python -c "import site; print(site.getsitepackages()[0])"`
mkdir -p $python_path/flashattn_hopper
wget -P $python_path/flashattn_hopper https://raw.githubusercontent.com/Dao-AILab/flash-attention/v2.7.2/hopper/flash_attn_interface.py

conda clean --all -y
