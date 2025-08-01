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
    echo "Error: env must be 'train' | 'inference'"
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

python -m pip install --upgrade pip

# If the environment is "train" or "inference", install the base dependency package
if [ "$env" == "train" ] || [ "$env" == "inference" ]; then
    pip install -r ./requirements/requirements-base-metax.txt
fi

# Activate the target Conda environment
conda activate flagscale-${env}

# Navigate to requirements directory and install basic dependencies
pip install -r ./requirements/requirements-common-metax.txt

# If env equals 'inference'
if [[ ${env} == "inference" ]]; then
    echo "[INFO] Entering inference mode setup..."
    # Basic dependency installation
    if ! command -v pip &> /dev/null; then
        echo "Error: pip not found. Please install Python package manager first."
        exit 1
    fi

    # Perform unpath operation
    echo "python tools/patch/unpatch.py --backend vllm FlagScale --task inference --device-type Metax_C550 ..."
    python tools/patch/unpatch.py \
        --backend vllm \
        FlagScale \
        --task inference \
        --device-type Metax_C550 || {
        echo "Failed to execute unpatch script"; exit 1
    }

    # Preparation for Building Environment
    build_dir="build/Metax_C550/FlagScale/third_party/vllm"
    if [ ! -d "$build_dir" ]; then
        echo "Build directory not found: $build_dir"; exit 1
    fi

    # Enter the target directory and record the original path
    pushd "$build_dir" > /dev/null

    WHEEL_PATH="dist/vllm-0.8.5+maca2.33.0.12torch2.6-cp310-cp310-linux_x86_64.whl"

    {
        # Load environment variables
        source /etc/profile.d/conda.sh && \
        source ./env.sh || { echo "Failed to source environment files"; exit 1; }

        echo "Setting up Python environment ..."
        python setup.py bdist_wheel || { echo "Failed to build wheel"; exit 1; }

        echo "Installing custom wheel package..."
        pip uninstall -y vllm
        pip install -v "$WHEEL_PATH" || { echo "Failed to install wheel"; exit 1; }
    } || {
        echo "Build process failed in $(pwd)"
        popd > /dev/null
        exit 1
    }
    popd > /dev/null

    echo 'Inference environment setup completed successfully.'
fi

# Clean all conda caches
conda clean --all -y
