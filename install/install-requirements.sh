#!/bin/bash

set -e

print_help() {
    echo "Usage: $0 [--env <train|inference>] [--llama-cpp-backend <cpu|metal|blas|openblas|blis|cuda|gpu|musa|vulkan_mingw64|vulkan_msys2|cann|arm_kleidi|hip|opencl_android|opencl_windows_arm64>]"
    echo "Options:"
    echo "  --env <train|inference|RL>         Specify the environment type (required)"
    echo "  --llama-cpp-backend <backend>      Specify the llama.cpp backend (default: cpu)"
    echo "  --omni_infer <value>               Specify omni inference value (optional)"
}

# Initialize the variable
env=""
llama_cpp_backend="cpu"
omni_infer="0"

# Parse command-line options
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --env) env="$2"; shift ;;  # Assign the value after '--env'
        --llama-cpp-backend) llama_cpp_backend="$2"; shift ;;  # Assign the value after '--llama-cpp-backend'
        --omni_infer) omni_infer="$2"; shift ;;  # Assign the value after '--llama-cpp-backend'
        --help|-h) print_help; exit 0 ;;
        *) echo "Error: Unknown parameter passed."; print_help; exit 1 ;;
    esac
    shift
done

# Check if 'env' field is provided and is either 'train' | 'inference' | 'RL'
if [ -z "$env" ]; then
    echo "Error: env field is required. Please specify either 'train' | 'inference' | 'RL'."
    exit 1
fi

# Check the value of env
if [ "$env" != "train" ] && [ "$env" != "inference" ] && [ "$env" != "RL" ]; then
    echo "Error: env must be 'train' | 'inference' | 'RL'."
    exit 1
fi

python -m pip install --upgrade pip

# If the environment is "train" or "inference", install the base dependency package
if [ "$env" == "train" ] || [ "$env" == "inference" ]; then
    pip install -r ./requirements/requirements-base.txt
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
    if [ "$env" == "train" ] || [ "$env" == "inference" ]; then
        # Use the current Python version of the system (e.g. 3.12)
        conda create --name "flagscale-${env}" python=$(python --version | awk '{print $2}' | cut -d '.' -f 1,2) -y
    fi

    if [ "$env" == "RL" ]; then
        # Enforce the use of Python 3.10 for RL environments
        conda create --name "flagscale-${env}" python==3.10 -y
    fi
fi

# Activate the target Conda environment
conda activate flagscale-${env}

if [ "$env" == "train" ] || [ "$env" == "inference" ]; then
    # This command updates `setuptools` to the latest version, ensuring compatibility and access to the latest features for Python package management.
    pip install --upgrade setuptools

    # Navigate to requirements directory and install basic dependencies
    pip install -r ./requirements/requirements-common.txt

    # TransformerEngine
    # Megatron-LM requires TE >= 2.1.0.
    git clone --recursive https://github.com/NVIDIA/TransformerEngine.git
    cd TransformerEngine
    git checkout 5bee81e
    pip install .
    cd ..
    rm -r ./TransformerEngine

    # cudnn frontend
    pip install nvidia-cudnn-cu12==9.7.1.26
    CMAKE_ARGS="-DCMAKE_POLICY_VERSION_MINIMUM=3.5" pip install nvidia-cudnn-frontend
    python -c "import torch; print('cuDNN version:', torch.backends.cudnn.version());"
    python -c "from transformer_engine.pytorch.utils import get_cudnn_version; get_cudnn_version()"

    # Megatron-LM requires flash-attn >= 2.1.1, <= 2.7.3
    cu=$(nvcc --version | grep "Cuda compilation tools" | awk '{print $5}' | cut -d '.' -f 1)
    torch=$(pip show torch | grep Version | awk '{print $2}' | cut -d '+' -f 1 | cut -d '.' -f 1,2)
    cp=$(python3 --version | awk '{print $2}' | awk -F. '{print $1$2}')
    cxx=$(g++ --version | grep 'g++' | awk '{print $3}' | cut -d '.' -f 1)
    flash_attn_version="2.8.0.post2"
    wget https://github.com/Dao-AILab/flash-attention/releases/download/v${flash_attn_version}/flash_attn-${flash_attn_version}+cu${cu}torch${torch}cxx${cxx}abiFALSE-cp${cp}-cp${cp}-linux_x86_64.whl
    pip install --no-cache-dir flash_attn-${flash_attn_version}+cu${cu}torch${torch}cxx${cxx}abiFALSE-cp${cp}-cp${cp}-linux_x86_64.whl
    rm flash_attn-${flash_attn_version}+cu${cu}torch${torch}cxx${cxx}abiFALSE-cp${cp}-cp${cp}-linux_x86_64.whl

    # From Megatron-LM log
    pip install "git+https://github.com/Dao-AILab/flash-attention.git@v2.7.2#egg=flashattn-hopper&subdirectory=hopper"
    python_path=`python -c "import site; print(site.getsitepackages()[0])"`
    mkdir -p $python_path/flashattn_hopper
    wget -P $python_path/flashattn_hopper https://raw.githubusercontent.com/Dao-AILab/flash-attention/v2.7.2/hopper/flash_attn_interface.py

    # If env equals 'train'
    if [ "${env}" == "train" ]; then
        # Unpatch
        python tools/patch/unpatch.py --backend Megatron-LM

        # Navigate to requirements directory and install training dependencies
        pip install -r ./requirements/train/megatron/requirements-cuda.txt

        # apex train
        git clone https://github.com/NVIDIA/apex
        cd apex
        pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --global-option="--use-ninja" --config-settings '--build-option=--cpp_ext' --config-settings '--build-option=--cuda_ext' ./
        cd ..
        rm -r ./apex

        python -m nltk.downloader -d /root/nltk_data punkt

        # Used for automatic fault tolerance
        # Set the path to the target Python file
        SITE_PACKAGES_DIR=$(python3 -c "import site; print(site.getsitepackages()[0])")
        FILE="$SITE_PACKAGES_DIR/torch/distributed/elastic/agent/server/api.py"
        torch_version=`python -c "import torch; print(torch.__version__)"`
        echo "torch_version: $torch_version"

        # Replace the following code with torch version 2.5.1
        if [[ $torch_version == *"2.5.1"* ]];then
            # Check and replace line 893
            LINE_893=$(sed -n '893p' "$FILE")
            EXPECTED_893='                if num_nodes_waiting > 0:'

            if [[ "$LINE_893" != "$EXPECTED_893" ]]; then
                echo "Error: Line 893 in $FILE does not exactly match '                if num_nodes_waiting > 0:' ."
                exit 1
            else
                echo "Line 893 is correct. Proceeding with replacement."
                # Directly replace the line without using regex
                sed -i '893s|.*|                if num_nodes_waiting > 0 and self._remaining_restarts > 0:|' "$FILE"
                echo "Success: Line 893 replaced."
            fi

            # Check and replace line 902
            LINE_902=$(sed -n '902p' "$FILE")
            EXPECTED_902='                    self._restart_workers(self._worker_group)'

            if [[ "$LINE_902" != "$EXPECTED_902" ]]; then
                echo "Error: Line 902 does not match '                    self._restart_workers(self._worker_group)'."
                exit 1
            else
                echo "Line 902 is correct. Proceeding with replacement."
                # Directly replace the line without using regex
                sed -i '902s|.*|                    self._remaining_restarts -= 1; self._restart_workers(self._worker_group)|' "$FILE"
                echo "Success: Line 902 replaced."
            fi
        fi

        # Replace the following code with torch version 2.6.0
        if [[ $torch_version == *"2.6.0"* ]] || [[ $torch_version == *"2.7.0"* ]];then
            # Check and replace line 908
            LINE_908=$(sed -n '908p' "$FILE")
            EXPECTED_908='                if num_nodes_waiting > 0:'

            if [[ "$LINE_908" != "$EXPECTED_908" ]]; then
                echo "Error: Line 908 in $FILE does not exactly match '                if num_nodes_waiting > 0:'."
                exit 1
            else
                echo "Line 908 is correct. Proceeding with replacement."
                # Directly replace the line without using regex
                sed -i '908s|.*|                if num_nodes_waiting > 0 and self._remaining_restarts > 0:|' "$FILE"
                echo "Success: Line 908 replaced."
            fi

            # Check and replace line 917
            LINE_917=$(sed -n '917p' "$FILE")
            EXPECTED_917='                    self._restart_workers(self._worker_group)'

            if [[ "$LINE_917" != "$EXPECTED_917" ]]; then
                echo "Error: Line 917 does not match '                    self._restart_workers(self._worker_group)'."
                exit 1
            else
                echo "Line 917 is correct. Proceeding with replacement."
                # Directly replace the line without using regex
                sed -i '917s|.*|                    self._remaining_restarts -= 1; self._restart_workers(self._worker_group)|' "$FILE"
                echo "Success: Line 917 replaced."
            fi
        fi

        # For FlagRelease
        pip install --no-build-isolation git+https://github.com/FlagOpen/FlagGems.git@release_v1.0.0
    fi

    # If env equals 'inference'
    if [ "${env}" == "inference" ]; then
        # Navigate to requirements directory and install basic dependencies
        pip install -r ./requirements/inference/requirements.txt

        # Unpatch
        python tools/patch/unpatch.py --backend vllm
        python tools/patch/unpatch.py --backend llama.cpp
        python tools/patch/unpatch.py --backend omniinfer

        # Build vllm
        # Navigate to requirements directory and install inference dependencies
        pip install -r ./third_party/vllm/requirements/build.txt
        pip install -r ./third_party/vllm/requirements/cuda.txt
        pip install -r ./third_party/vllm/requirements/common.txt
        pip install "git+https://github.com/state-spaces/mamba.git@v2.2.4"
        pip install -r ./third_party/vllm/requirements/dev.txt

        MAX_JOBS=$(nproc) pip install --no-build-isolation -v ./third_party/vllm/.

        # Navigate to requirements directory and install serving dependencies
        pip install -r ./requirements/serving/requirements.txt

        # Build llama.cpp
        cd ./third_party/llama.cpp
        rm -rf ./build
        case "$llama_cpp_backend" in
            cpu|metal|cpu_and_metal)
                cmake -B build
                cmake --build build --config Release -j8
                ;;
            blas|openblas)
                cmake -B build -DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS
                cmake --build build --config Release
                ;;
            blis)
                # You can skip this step if  in oneapi-basekit docker image, only required for manual installation
                source /opt/intel/oneapi/setvars.sh
                cmake -B build -DGGML_BLAS=ON -DGGML_BLAS_VENDOR=Intel10_64lp -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx -DGGML_NATIVE=ON
                cmake --build build --config Release
                ;;
            cuda|gpu)
                cmake -B build -DGGML_CUDA=ON -DLLAMA_CURL=OFF
                cmake --build build --config Release
                ;;
            musa)
                cmake -B build -DGGML_MUSA=ON
                cmake --build build --config Release
                ;;
            vulkan_mingw64)
                cmake -B build -DGGML_VULKAN=ON
                cmake --build build --config Release
                ;;
            cann)
                cmake -B build -DGGML_CANN=on -DCMAKE_BUILD_TYPE=release
                cmake --build build --config release
                ;;
            arm_kleidi)
                cmake -B build -DGGML_CPU_KLEIDIAI=ON
                cmake --build build --config Release
                ;;
            hip|vulkan_w64devkit|vulkan_msys2|opencl_android|opencl_windows_arm64)
                echo "auto build unsupport: $1, follow the README.md to build manually:"
                echo "https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md"
                exit 1
                ;;  
            *)
                echo "unknown backend: $1"
                print_help
                exit 1
                ;;
        esac

        cd ../..
        # Build omniinfer
        if [ "${omni_infer}" == "1" ]; then
            # process repo
            find ./third_party/omniinfer -type f -exec dos2unix {} +
            find ./third_party/omniinfer -type f -path '*.sh' -exec chmod a+x {} \;

            # unpatch vllm
            cd ./third_party/omniinfer/infer_engines/
            git clone https://github.com/vllm-project/vllm.git
            git checkout 65334ef3
            bash bash_install_code.sh
            cd ../../..

            # install dependencies
            pip install -r ./third_party/omniinfer/tests/requirements.txt

            # build whl for vllm
            mkdir -p ./third_party/omniinfer/build/dist
            cd ./third_party/omniinfer/infer_engines/vllm
            VLLM_TARGET_DEVICE=empty python setup.py bdist_wheel
            mv dist/vllm* ../../build/dist

            # build whl for omniinfer
            cd ../..
            pip install build
            python -m build
            mv dist/omni_i* ./build/dist

            # build whl for omniinfer omni_placement
            cd ./omni/accelerators/placement
            python setup.py bdist_wheel
            mv dist/omni_placement* ../../../build/dist

            # install 3 whl
            cd ../../../build/dist
            pip install omni_i*.whl
            pip install vllm*.whl
            pip install omni_placement*.whl

            cd ../../../..
        fi

        # For FlagRelease
        pip install --no-build-isolation git+https://github.com/FlagOpen/FlagGems.git@release_v1.0.0
    fi
fi

if [ "$env" == "RL" ]; then
    # install verl
    git clone https://github.com/volcengine/verl.git
    cd verl
    pip install --no-deps -e .
    bash scripts/install_vllm_sglang_mcore.sh
    cd ..
    rm -rf ./verl

    # Install dependencies
    pip install cryptography
fi

# Clean all conda caches
conda clean --all -y