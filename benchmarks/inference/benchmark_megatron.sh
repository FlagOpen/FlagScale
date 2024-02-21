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

# 2GPUS-TensorParallel
INFER_ARGS="
    --tensor-model-parallel-size 2\
    --pipeline-model-parallel-size 1 \
    --make-vocab-size-divisible-by 64 \
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

# 33B CONFIG
NETWORK_ARGS="
    --model-info $MODEL_INFO\
    --num-layers 60 \
    --hidden-size 6144 \
    --hidden-dim-multiplier 1.3 \
    --num-attention-heads 48 \
    --group-query-attention \
    --num-query-groups 8 \
    --seq-length 4096 \
    --max-position-embeddings 4096 \
    --normalization RMSNorm \
    --use-rotary-position-embeddings \
    --no-position-embedding \
    --swiglu \
    --multiple-of 4096 \
    --untie-embeddings-and-output-weights \
    --norm-epsilon 1e-5 \
    --norm-init-weight 0.3 \
"

CHECKPOINTING_ARGS="
    --load $CHECKPOINT
"

if [[ "$TYPE" == "throughout" ]]; then
    BENCHMARK_ARGS="
        --micro-batch-size 1 \
        --num-requests 10 \
        --temperature 1.0 \
        --top-p 0.9 \
        --top-k 200 \
        --prompt-len 64 \
        --generate-len 64 \
        --seed 42
    "
    cmd="
    export CUDA_DEVICE_MAX_CONNECTIONS=1;
    export CUDA_VISIBLE_DEVICES=$DEVICES;

    torchrun $DISTRIBUTED_ARGS benchmark_megatron_throughout.py \
                $INFER_ARGS \
                $MIXED_PRECISION_ARGS \
                $DATA_ARGS \
                $NETWORK_ARGS \
                $CHECKPOINTING_ARGS \
                $BENCHMARK_ARGS
    "
elif [[ "$TYPE" == "latency" ]]; then
    BENCHMARK_ARGS="
        --micro-batch-size 1 \
        --num-iters 10 \
        --temperature 1.0 \
        --top-p 0.9 \
        --top-k 200 \
        --prompt-len 64 \
        --generate-len 64 \
        --seed 42
    "
    cmd="
    export CUDA_DEVICE_MAX_CONNECTIONS=1;
    export CUDA_VISIBLE_DEVICES=$DEVICES;

    torchrun $DISTRIBUTED_ARGS benchmark_megatron_latency.py \
                  $INFER_ARGS \
                  $MIXED_PRECISION_ARGS \
                  $DATA_ARGS \
                  $NETWORK_ARGS \
                  $CHECKPOINTING_ARGS \
                  $BENCHMARK_ARGS
    "
elif [[ "$TYPE" == "serving" ]]; then
    BENCHMARK_ARGS="
        --micro-batch-size 1 \
    "
    cmd="
    export CUDA_DEVICE_MAX_CONNECTIONS=1;
    export CUDA_VISIBLE_DEVICES=$DEVICES;

    torchrun $DISTRIBUTED_ARGS megatron_server.py \
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
