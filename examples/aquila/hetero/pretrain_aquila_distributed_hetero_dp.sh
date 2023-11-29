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
rm -rf $LOG_PATH
mkdir -p $LOG_PATH
cp $0 $LOG_PATH/
TB_PATH=$PROJ_HOME/tboard/$EXPNAME
mkdir -p $TB_PATH
WB_PATH=$PROJ_HOME/wandb/$EXPNAME
mkdir -p $WB_PATH

DISTRIBUTED_ARGS="
    --nnodes $NUM_NODES \
    --node_rank $NODE_RANK \
    --nproc_per_node $NODE_DEVICES \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
"
    # --log_dir $LOG_PATH --redirects 3 --tee 3

# DISTRIBUTED_ARGS="
#     --nnodes $NUM_NODES \
#     --rdzv_id "hetero" \
#     --nproc_per_node $NODE_DEVICES \
#     --rdzv-backend=c10d \
#     --rdzv-endpoint=$MASTER_ADDR:$MASTER_PORT
# "

HETERO_ARGS="
    --hetero-mode dp \
    --hetero-device-types A800 A100 \
    --hetero-current-device-type $NODE_TYPE \
    --hetero-micro-batch-sizes 2 3 2 1 \
"

TRAINING_ARGS="
    --train-samples 40000 \
    --eval-iters 0 \
    --tensor-model-parallel-size 2 \
    --pipeline-model-parallel-size 2 \
    --global-batch-size 32  \
    --disable-bias-linear
"

MIXED_PRECISION_ARGS="
    --bf16 \
    --embedding-weights-in-fp32 \
    --attention-softmax-in-fp32 \
    --accumulate-allreduce-grads-in-fp32
"

DATA_ARGS="
    --data-path $DATA_PATH \
    --tokenizer-type AquilaTokenizer \
    --vocab-file $VOCAB_FILE \
    --vocab-size 100008\
    --merge-file $MERGE_FILE \
    --special-tokens-file $SPECIAL_TOKENS_FILE \
    --data-impl mmap \
    --split 1
"

NETWORK_ARGS="
    --num-layers 8 \
    --hidden-size 4096 \
    --num-attention-heads 32 \
    --seq-length 2048 \
    --max-position-embeddings 2048 \
    --layernorm-epsilon 1e-5 \
    --use-rotary-position-embeddings \
    --rotary-position-embeddings-in-fp32 \
    --no-position-embedding \
    --swiglu \
    --multiple-of 256 \
    --apply-layernorm-rms \
    --untie-embeddings-and-output-weights
"

INITIALIZATION_ARGS="
    --init-method-std 0.02 \
    --seed 1234 
"

REGULARIZATION_ARGS="
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --weight-decay 0.1 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --clip-grad 0.0
"

LEARNING_RATE_ARGS="
    --lr 2.0e-3 \
"
    # --min-lr 2.0e-6 \
    # --lr-decay-style cosine \
    # --lr-warmup-samples 1000 

CHECKPOINTING_ARGS="
    --load $CHECKPOINT_PATH
"
    # --save-interval 200000 \
    # --save $CHECKPOINT_PATH \

LOGGING_ARGS="
    --log-interval 1 \
"
    # --wandb-dir $WB_PATH \
    # --tensorboard-dir $TB_PATH \
    # --tensorboard-log-interval 1 

ENV_ARGS=""

cmd="$ENV_ARGS torchrun $DISTRIBUTED_ARGS pretrain_gpt.py \
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
