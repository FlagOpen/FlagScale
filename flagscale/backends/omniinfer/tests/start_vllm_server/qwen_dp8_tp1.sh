#!/bin/bash

# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

set -e

MODEL_PATH=$1
python3 -m vllm.entrypoints.openai.api_server \
    --host 0.0.0.0 \
    --port 8000 \
    --model ${MODEL_PATH} \
    --data-parallel-size 8 \
    --tensor-parallel-size 1 \
    --dtype bfloat16 \
    --max-model-len 10240 \
    --trust_remote_code \
    --gpu_memory_utilization 0.95 \
    --enforce-eager \
    --block_size 128 \
    --served-model-name qwen \
    --distributed-executor-backend mp \
    --max-num-batched-tokens 200000 \
    --max-num-seqs 128