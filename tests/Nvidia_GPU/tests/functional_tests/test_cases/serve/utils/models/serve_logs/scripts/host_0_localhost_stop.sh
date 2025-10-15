#!/bin/bash

set -x

ulimit -n 65535 && source /root/miniconda3/bin/activate flagscale-inference

export CUDA_DEVICE_MAX_CONNECTIONS=1 && export no_proxy=127.0.0.1,localhost
ulimit -n 65535 && source /root/miniconda3/bin/activate flagscale-inference
pkill -f 'run_inference_engine'
pkill -f 'run_fs_serve_vllm'
pkill -f 'vllm serve'
pkill -f multiprocessing


