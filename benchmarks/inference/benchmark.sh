#!/bin/bash

MODEL_INFO=$1
CHECKPOINT=$2
MASTER_PORT=$3
DEVICES=$4
TYPE=$5

# Define files related to tokenizer
VOCAB_FILE=../../examples/aquila/tokenizer/vocab.json
MERGE_FILE=../../examples/aquila/tokenizer/merges.txt
SPECIAL_TOKENS_FILE=../../examples/aquila/tokenizer/special_tokens.txt

nproc=$(awk -F',' '{print NF}' <<< "$DEVICES")
DISTRIBUTED_ARGS="
    --nproc_per_node $nproc \
    --nnodes 1 \
    --master_addr localhost \
    --master_port $MASTER_PORT
"

INFER_ARGS="
    --tensor-model-parallel-size 1\
    --pipeline-model-parallel-size 1 \
    --make-vocab-size-divisible-by 128 \
    --disable-bias-linear \
    --use-flash-attn
"

MIXED_PRECISION_ARGS="
    --bf16 \
"

DATA_ARGS="
    --tokenizer-type AquilaTokenizer \
    --vocab-file $VOCAB_FILE \
    --merge-file $MERGE_FILE \
    --special-tokens-file $SPECIAL_TOKENS_FILE \
"

NETWORK_ARGS="
    --model-info $MODEL_INFO\
    --num-layers 24 \
    --hidden-size 2048 \
    --hidden-dim-multiplier 1. \
    --num-attention-heads 16 \
    --seq-length 2048 \
    --max-position-embeddings 2048 \
    --normalization RMSNorm \
    --use-rotary-position-embeddings \
    --no-position-embedding \
    --swiglu \
    --multiple-of 128 \
    --untie-embeddings-and-output-weights \
    --norm-epsilon 1e-6 \
    --norm-init-weight 0.8441 \
"

CHECKPOINTING_ARGS="
    --load $CHECKPOINT
"

# TODO: add 'TYPE == serving'
if [[ "$TYPE" == "throughout" ]]; then
    BENCHMARK_ARGS="
        --micro-batch-size 1 \
        --num-requests 20 \
        --temperature 0.9 \
        --top_p 0.9 \
        --top_k 200 \
        --prompt-len 32 \
        --generate-len 128 \
        --dataset-path 'test.jsonl' \
        --seed 42
    "
    cmd="
    export CUDA_DEVICE_MAX_CONNECTIONS=1;
    export CUDA_VISIBLE_DEVICES=$DEVICES;

    torchrun $DISTRIBUTED_ARGS test_throughout.py \
                $INFER_ARGS \
                $MIXED_PRECISION_ARGS \
                $DATA_ARGS \
                $NETWORK_ARGS \
                $CHECKPOINTING_ARGS \
                $BENCHMARK_ARGS
    "
elif [[ "$TYPE" == "latency" ]]; then
    BENCHMARK_ARGS="
        --micro-batch-size 2 \
        --num-iters 10 \
        --temperature 0.9 \
        --top_p 0.9 \
        --top_k 200 \
        --prompt-len 32 \
        --generate-len 128 \
        --dataset-path 'test.jsonl' \
        --seed 42
    "
    cmd="
    export CUDA_DEVICE_MAX_CONNECTIONS=1;
    export CUDA_VISIBLE_DEVICES=$DEVICES;

    torchrun $DISTRIBUTED_ARGS test_latency.py \
                  $INFER_ARGS \
                  $MIXED_PRECISION_ARGS \
                  $DATA_ARGS \
                  $NETWORK_ARGS \
                  $CHECKPOINTING_ARGS \
                  $BENCHMARK_ARGS
    "
else
    cmd="
    echo 'Please set right benchmark TYPE.'
    "
fi

echo $cmd
eval $cmd
