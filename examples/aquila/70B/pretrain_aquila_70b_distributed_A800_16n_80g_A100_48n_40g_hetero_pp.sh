#!/bin/bash

PROJ_HOME=$1
EXPNAME=$2
HOSTFILE=$3
DATA_PATH=$4

# Preapre the environment related configuration
source examples/aquila/env.sh

# Define files related to tokenizer
VOCAB_FILE=examples/aquila/tokenizer/vocab.json
MERGE_FILE=examples/aquila/tokenizer/merges.txt
SPECIAL_TOKENS_FILE=examples/aquila/tokenizer/special_tokens.txt

# Build some paths for the current training
CHECKPOINT_PATH=$PROJ_HOME/checkpoints/$EXPNAME
mkdir -p $CHECKPOINT_PATH
LOG_PATH=$PROJ_HOME/logs/$EXPNAME
mkdir -p $LOG_PATH
cp $0 $LOG_PATH/
TB_PATH=$PROJ_HOME/tboard/$EXPNAME
mkdir -p $TB_PATH
WB_PATH=$PROJ_HOME/wandb/$EXPNAME
mkdir -p $WB_PATH


DISTRIBUTED_ARGS="
    --nproc_per_node $NODE_DEVICES \
    --nnodes $NUM_NODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT 
"

HETERO_ARGS="
    --hetero-mode pp \
    --hetero-current-device-type $NODE_TYPE \
    --hetero-device-types A800 A100 \
    --hetero-pipeline-stages 1 20 3 20 20 20 \
"

TRAINING_ARGS="
    --train-samples 488281250 \
    --rampup-batch-size 32 32 2000000 \
    --eval-iters 0 \
    --eval-interval 2000 \
    --tensor-model-parallel-size 8 \
    --pipeline-model-parallel-size 4 \
    --micro-batch-size 1 \
    --global-batch-size 1024 \
    --disable-bias-linear \
    --use-flash-attn \
    --sequence-parallel \
    --use-distributed-optimizer
"

MIXED_PRECISION_ARGS="
    --bf16 \
    --attention-softmax-in-fp32 \
    --accumulate-allreduce-grads-in-fp32
"

DATA_ARGS="
    --data-path $DATA_PATH \
    --tokenizer-type AquilaTokenizer \
    --vocab-file $VOCAB_FILE \
    --vocab-size 100008\
    --make-vocab-size-divisible-by 64 \
    --merge-file $MERGE_FILE \
    --special-tokens-file $SPECIAL_TOKENS_FILE \
    --data-impl mmap \
    --split 1
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

LEARNING_RATE_ARGS="
    --lr 1.5e-4 \
    --lr-decay-style cosine \
    --lr-warmup-samples 500000 \
    --min-lr 1.5e-5
"

CHECKPOINTING_ARGS="
    --save-interval 500 \
    --rampup-save-interval 5000 \
    --save $CHECKPOINT_PATH \
    --load $CHECKPOINT_PATH
"

LOGGING_ARGS="
    --log-interval 1 \
    --tensorboard-dir $TB_PATH \
    --tensorboard-log-interval 1 \
    --wandb-save-dir $WB_PATH
"

cmd="torchrun $DISTRIBUTED_ARGS pretrain_gpt.py \
              $HETERO_ARGS \
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
