#!/bin/bash

# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

set -e

MODEL_PATH=$1
vllm serve ${MODEL_PATH} --served-model-name=deepseek --trust-remote-code --max-model-len=8192 --gpu-memory-utilization=0.95 --data-parallel-size 2 --data-parallel-size-local 2  -tp=4 --enable-expert-parallel --block_size 128