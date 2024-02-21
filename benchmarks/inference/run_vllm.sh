#!/bin/bash

MODEL=BAAI/AquilaChat2-7B
TYPE=latency #throughout/latency

FlagScale_HOME=<xxxx>
LOG_FILE=./log.vllm.$TYPE

cd $FlagScale_HOME/benchmarks/inference;
export CUDA_DEVICE_MAX_CONNECTIONS=1;
export CUDA_VISIBLE_DEVICES=0;

nohup python benchmark_vllm_$TYPE.py \
        --num-requests 10 \
        --temperature 0.9 \
        --top-p 0.9 \
        --top-k 200 \
        --prompt-len 64 \
        --generate-len 64 \
        --seed 42 \
        --model $MODEL\
        --tensor-parallel-size 1 \
        --pipeline-parallel-size 1 > $LOG_FILE 2>&1 &
