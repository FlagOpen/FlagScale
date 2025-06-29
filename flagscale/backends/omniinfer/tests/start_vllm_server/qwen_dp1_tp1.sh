#!/bin/bash
set -e

export GLOO_SOCKET_IFNAME=enp23s0f3
export VLLM_USE_V1=1
export VLLM_WORKER_MULTIPROC_METHOD=fork
export VLLM_ENABLE_MC2=0
export USING_LCCL_COM=0
export VLLM_LOGGING_LEVEL=DEBUG
export ASCEND_RT_VISIBLE_DEVICES=0

source ~/.bashrc || true

MODEL_PATH=$1
python3 -m vllm.entrypoints.openai.api_server \
    --host 0.0.0.0 \
    --port 8300 \
    --model ${MODEL_PATH} \
    --data-parallel-size 1 \
    --tensor-parallel-size 1 \
    --dtype bfloat16 \
    --max-model-len 4096 \
    --trust_remote_code \
    --gpu_memory_utilization 0.9 \
    --enforce-eager \
    --block_size 128 \
    --served-model-name qwen \
    --distributed-executor-backend mp \
    --max-num-batched-tokens 20000 \
    --max-num-seqs 128
