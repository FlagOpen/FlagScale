#!/bin/bash

# Runs the "70B" parameter model

export CUDA_DEVICE_MAX_CONNECTIONS=1

GPUS_PER_NODE=8
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6000
NUM_NODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NUM_NODES))

CHECKPOINT_PATH=$0 #<Specify path>
TENSORBOARD_LOGS_PATH=$1 #<Specify path>
TOKENIZER_PATH=$2 #<Specify path to file>/tokenizer.model
DATA_PATH=$3 #<Specify path and file prefix>_text_document

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE 
    --nnodes $NUM_NODES 
    --master_addr $MASTER_ADDR 
    --master_port $MASTER_PORT
)

LLAMA_MODEL_ARGS=(
    --num-layers 80 
    --hidden-size 8192
    --ffn-hidden-size 28672 
    --num-attention-heads 64 
    --seq-length 4096 
    --max-position-embeddings 4096
    --group-query-attention
    --num-query-groups 8
    --tokenizer-type Llama2Tokenizer
    --tokenizer-model $TOKENIZER_PATH
    --swiglu
    --use-flash-attn
    --normalization RMSNorm
    --use-rotary-position-embeddings
    --no-position-embedding
    --disable-bias-linear
)

TRAINING_ARGS=(
    --micro-batch-size 1 
    --global-batch-size 1024 
    --train-iters 500000 
    --weight-decay 1e-2
    --use-distributed-optimizer
    --clip-grad 1.0 
    --fp16
    --lr 0.00015
    --lr-decay-style cosine 
    --min-lr 1.0e-5
    --lr-warmup-fraction .01 
    --lr-decay-iters 320000
    --adam-beta1 0.9
    --adam-beta2 0.95
    --attention-dropout 0
    --hidden-dropout 0
    --untie-embeddings-and-output-weights
    --multiple-of 4096
    --sequence-parallel
    --distributed-backend nccl
)

MODEL_PARALLEL_ARGS=(
	--tensor-model-parallel-size 8 
	--pipeline-model-parallel-size 4 
)

DATA_ARGS=(
    --data-path $DATA_PATH 
    --split 949,50,1
)

EVAL_AND_LOGGING_ARGS=(
    --log-interval 100
    --save-interval 10000 
    --eval-interval 1000 
    --save $CHECKPOINT_PATH 
    --load $CHECKPOINT_PATH 
    --eval-iters 10
    --tensorboard-dir $TENSORBOARD_LOGS_PATH 
)

torchrun ${DISTRIBUTED_ARGS[@]} pretrain_llama.py \
    ${LLAMA_MODEL_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]}
