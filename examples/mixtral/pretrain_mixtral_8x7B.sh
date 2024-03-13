#!/bin/bash

# Runs Mixtral 8x7B model on 16 A100 GPUs

PROJ_HOME=$1
EXPNAME=$2
HOSTFILE=$3
DATA_PATH=$4

VOCAB_FILE=../examples/aquila/tokenizer/vocab.json
MERGE_FILE=../examples/aquila/tokenizer/merges.txt
SPECIAL_TOKENS_FILE=../examples/aquila/tokenizer/special_tokens.txt
TOKENIZER_PATH=<xxxx>

CHECKPOINT_PATH=$PROJ_HOME/checkpoints/$EXPNAME
mkdir -p $CHECKPOINT_PATH
LOG_PATH=$PROJ_HOME/logs/$EXPNAME
mkdir -p $LOG_PATH
rm -rf $LOG_PATH/*
TB_PATH=$PROJ_HOME/tboard/$EXPNAME
mkdir -p $TB_PATH
WB_PATH=$PROJ_HOME/wandb/$EXPNAME
mkdir -p $WB_PATH

# export NODE_ADDR=$(ifconfig bond0|grep inet|grep -v 127.0.0.1|grep -v inet6|awk '{print $2;}'|tr -d "addr:"|head -n 1)
# export GPUS_PER_NODE=$(awk '{$1=$1;print}' $HOSTFILE|awk -F" |=" '{ranks[$1]=$NF;}END{print ranks["'$NODE_ADDR'"];}')
# export NNODES=$(awk '{$1=$1;print}' $HOSTFILE | wc -l)
# export MASTER_ADDR=$(head -n1 $HOSTFILE | awk '{print $1;}')
# export NODE_RANK=$(awk '{ranks[$1]=(FNR-1);}END{print ranks["'$NODE_ADDR'"];}' $HOSTFILE)
# export MASTER_PORT=23456

# Preapre the environment related configuration
source ../examples/aquila/env.sh

GPUS_PER_NODE=8
# Change for multinode config
MASTER_ADDR=${MASTER_ADDR:-"localhost"}
MASTER_PORT=${MASTER_PORT:-"6000"}
NNODES=${NNODES:-"1"}
NODE_RANK=${RANK:-"0"}
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

# export NCCL_DEBUG=INFO

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE
    --nnodes $NNODES
    --node_rank $NODE_RANK
    --master_addr $MASTER_ADDR
    --master_port $MASTER_PORT
    --log_dir $LOG_PATH
    --redirects 3
    --tee 3
)

MODEL_ARGS=(
    --use-mcore-models
    --disable-bias-linear
    --seq-length 2048
    --max-position-embeddings 32768
    --num-layers 4
    --hidden-size 4096
    --ffn-hidden-size 14336
    --num-attention-heads 32
    --init-method-std 0.01
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --normalization RMSNorm
    --position-embedding-type rope
    --swiglu
    --untie-embeddings-and-output-weights
    --group-query-attention
    --num-query-groups 8
    --no-masked-softmax-fusion
    --no-position-embedding
)

MOE_ARGS=(
    --num-experts 8
    --expert-model-parallel-size 2 # 4
    --moe-router-load-balancing-type aux_loss # options: aux_loss, sinkhorn, None. Default is aux_loss.
    --moe-router-topk 2
    --moe-aux-loss-coeff 1e-2
#    --moe-grouped-gemm 
)

DATA_ARGS=(
    --data-path $DATA_PATH
    --tokenizer-type QwenTokenizer
    --tokenizer-path $TOKENIZER_PATH
    --vocab-file $VOCAB_FILE
    --vocab-size 151851
    --make-vocab-size-divisible-by 64
    --merge-file $MERGE_FILE
    --special-tokens-file $SPECIAL_TOKENS_FILE
    --split 1
)

TRAINING_ARGS=(
    --micro-batch-size 1
    --global-batch-size 128
    --lr 1e-4
    --train-iters 500000
    --lr-decay-iters 320000
    --lr-decay-style cosine
    --min-lr 1.0e-5
    --weight-decay 0.1
    --lr-warmup-iters 500
    --clip-grad 1.0
    --bf16
)

MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size 2
    --pipeline-model-parallel-size 2
    --sequence-parallel
    --use-distributed-optimizer
)

LOGGING_ARGS=(
    --log-interval 1 \
    --save-interval 10000 \
    --eval-interval 1000 \
    --eval-iters 10 \
    --save $CHECKPOINT_PATH \
    --load $CHECKPOINT_PATH \
    --tensorboard-dir "${CHECKPOINT_PATH}/tensorboard" \
    --no-load-optim \
    --no-load-rng
)

if [ -n "${WANDB_API_KEY}" ]; then
    LOGGING_ARGS+=(
        --wandb-project ${WANDB_PROJECT:-"Mixtral-Finetuning"}
        --wandb-exp-name ${WANDB_NAME:-"Mixtral_8x7B"} 
    )
fi

torchrun ${DISTRIBUTED_ARGS[@]} pretrain_gpt.py \
    ${MODEL_ARGS[@]} \
    ${MOE_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${LOGGING_ARGS[@]}
