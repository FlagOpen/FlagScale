#!/bin/bash

set -e

print_help() {
    echo "Usage: $0 [--env <train|inference>] [--llama-cpp-backend <cpu|metal|blas|openblas|blis|cuda|gpu|musa|vulkan_mingw64|vulkan_msys2|cann|arm_kleidi|hip|opencl_android|opencl_windows_arm64>]"
    echo "Options:"
    echo "  --env <train|inference|RL>         Specify the environment type (required)"
}

# Initialize the variable
env=""

# Parse command-line options
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --env) env="$2"; shift ;;  # Assign the value after '--env'
        --help|-h) print_help; exit 0 ;;
        *) echo "Error: Unknown parameter passed."; print_help; exit 1 ;;
    esac
    shift
done

# Check if 'env' field is provided and is either 'train' | 'inference'
if [ -z "$env" ]; then
    echo "Error: env field is required. Please specify either 'train' | 'inference'"
    exit 1
fi

# Check the value of env
if [ "$env" != "train" ] && [ "$env" != "inference" ]; then
    echo "Error: env must be 'train' | 'inference'
    exit 1
fi

# Proceed with setup based on the value of 'env'
echo "Setting up environment for: $env"

# Load conda environment
source /etc/profile.d/conda.sh
 
# Create and activate Conda virtual environment
# The Python version used has been written into the conda config
if conda env list | grep -q "flagscale-${env}"; then
    # Check if the environment already exists
    echo "Conda environment 'flagscale-${env}' already exists."
else
    echo "Creating conda environment 'flagscale-${env}'..."
    # Create an flagscale-${env} environment based on the base environment
    conda create --name "flagscale-${env}" --clone base
fi

# Activate the target Conda environment
conda activate flagscale-${env}

# If env equals 'inference'
if [ "${env}" == "inference" ]; then
    pip install hydra-core
    python tools/patch/unpatch.py --backend vllm FlagScale --task inference --device-type Metax_C550
    cd build/Metax_C550/FlagScale/third_party/vllm/
    source /etc/profile.d/conda.sh
    source ./env.sh
    python setup.py bdist_wheel
    pip uninstall vllm
    pip install dist/vllm-0.8.5+maca2.33.0.12torch2.6-cp310-cp310-linux_x86_64.whl
fi

# Clean all conda caches
conda clean --all -y