#!/bin/bash

export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_SOCKET_IFNAME=bond0
export NCCL_IB_DISABLE=0
export NCCL_IB_CUDA_SUPPORT=1
export NCCL_IB_GID_INDEX=0
export NCCL_IB_HCA=mlx5_0,mlx5_1
export NCCL_DEBUG=DEBUG
export OMP_NUM_THREADS=4
export CUDA_DEVICE_MAX_CONNECTIONS=1
export GLOO_SOCKET_IFNAME=bond0

WORLD_SIZE=4
TASK="AQUILA"
VALID_DATA=<Specify lambada path>
CHECKPOINT=<Specify checkpoints path>
VOCAB_FILE=../examples/aquila/tokenizer/vocab.json
MERGE_FILE=../examples/aquila/tokenizer/merges.txt
SPECIAL_TOKENS_FILE=../examples/aquila/tokenizer/special_tokens.txt


DISTRIBUTED_ARGS="
    --nproc_per_node $WORLD_SIZE \
    --nnodes 1 \
    --node_rank 0 \
    --master_addr localhost \
    --master_port 6144
"

TRAINING_ARGS="
    --task $TASK \
    --eval-metric loss \
    --tensor-model-parallel-size 4 \
    --pipeline-model-parallel-size 1 \
    --micro-batch-size 1 \
    --disable-bias-linear \
    --use-flash-attn
"

MIXED_PRECISION_ARGS="
    --bf16 \
    --attention-softmax-in-fp32 \
"

DATA_ARGS="
    --valid-data $VALID_DATA \
    --tokenizer-type AquilaTokenizer \
    --make-vocab-size-divisible-by 64 \
    --vocab-file $VOCAB_FILE \
    --vocab-size 100008\
    --merge-file $MERGE_FILE \
    --special-tokens-file $SPECIAL_TOKENS_FILE \
"

NETWORK_ARGS="
    --num-layers 80 \
    --hidden-size 8192 \
    --num-attention-heads 64 \
    --group-query-attention \
    --num-query-groups 8 \
    --hidden-dim-multiplier 1.3 \
    --seq-length 4096 \
    --max-position-embeddings 4096 \
    --norm-epsilon 1e-5 \
    --norm-init-weight 0.25 \
    --use-rotary-position-embeddings \
    --no-position-embedding \
    --swiglu \
    --multiple-of 4096 \
    --normalization RMSNorm \
    --untie-embeddings-and-output-weights
"

INITIALIZATION_ARGS="
    --init-method-std 0.0149 \
    --seed 42
"

REGULARIZATION_ARGS="
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --weight-decay 0.1 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --clip-grad 1.0
"

LEARNING_RATE_ARGS=""

CHECKPOINTING_ARGS="
    --load $CHECKPOINT\
    --no-initialization \
    --use-cpu-initialization \
    --no-load-optim \
    --no-load-rng
"

LOGGING_ARGS="
    --log-interval 1
"

cmd="torchrun $DISTRIBUTED_ARGS ./tasks/main.py \
              $TRAINING_ARGS \
              $MIXED_PRECISION_ARGS \
              $DATA_ARGS \
              $NETWORK_ARGS \
              $INITIALIZATION_ARGS \
              $REGULARIZATION_ARGS \
              $LEARNING_RATE_ARGS \
              $CHECKPOINTING_ARGS \
              $LOGGING_ARGS
    "
echo $cmd
eval $cmd
