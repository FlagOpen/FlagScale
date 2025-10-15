#!/bin/bash

set -x

ulimit -n 65535 && source /root/miniconda3/bin/activate flagscale-inference

if [ -z "$PYTHONPATH" ]; then
    export PYTHONPATH=/home/FlagScale/vllm:/home/FlagScale
else
    export PYTHONPATH="$PYTHONPATH:/home/FlagScale/vllm:/home/FlagScale"
fi

export CUDA_DEVICE_MAX_CONNECTIONS=1 && export no_proxy=127.0.0.1,localhost
ulimit -n 65535 && source /root/miniconda3/bin/activate flagscale-inference
mkdir -p /home/FlagScale/tests/Nvidia_GPU/tests/functional_tests/test_cases/serve/utils/models/serve_logs
mkdir -p /home/FlagScale/tests/Nvidia_GPU/tests/functional_tests/test_cases/serve/utils/models/serve_logs/pids

cd /home/FlagScale

cmd="CUDA_DEVICE_MAX_CONNECTIONS=1 no_proxy=127.0.0.1,localhost python flagscale/serve/run_serve.py --config-path=/home/FlagScale/tests/Nvidia_GPU/tests/functional_tests/test_cases/serve/utils/models/serve_logs/scripts/serve.yaml --log-dir=/home/FlagScale/tests/Nvidia_GPU/tests/functional_tests/test_cases/serve/utils/models/serve_logs"

echo '=========== launch task ==========='
nohup bash -c "$cmd; sync" >> /home/FlagScale/tests/Nvidia_GPU/tests/functional_tests/test_cases/serve/utils/models/serve_logs/host_0_localhost.output 2>&1 & echo $! > /home/FlagScale/tests/Nvidia_GPU/tests/functional_tests/test_cases/serve/utils/models/serve_logs/pids/host_0_localhost.pid

