#!/bin/bash

######
# Note that this script is for continuous training from the Aquila-7B model,
# which was first trained by BMTrain. So if you want to start from scratch, please remove the
# "--rotary-interleaved-patch" argument and change the mixed-percision and learning rate 
# arguments as needed.
######

# Please change the following envrioment variables
# base on the cluster configuration
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_DISABLE=0
export NCCL_IB_CUDA_SUPPORT=1
export NCCL_IB_GID_INDEX=0
export NCCL_IB_HCA=mlx5_0,mlx5_3
export NCCL_DEBUG=debug
export OMP_NUM_THREADS=4

set -u
  PROJ_HOME=$1
  EXPNAME=$2
  HOSTFILE=$3
  DATA_PATH=$4
set +u

CHECKPOINT_PATH=$PROJ_HOME/checkpoints/$EXPNAME
mkdir -p $CHECKPOINT_PATH
VOCAB_FILE=examples/aquila/tokenizer/vocab.json
MERGE_FILE=examples/aquila/tokenizer/merges.txt
SPECIAL_TOKENS_FILE=examples/aquila/tokenizer/special_tokens.txt
LOG_PATH=$PROJ_HOME/logs/$EXPNAME
mkdir -p $LOG_PATH
cp $0 $LOG_PATH/
TB_PATH=$PROJ_HOME/tboard/$EXPNAME
mkdir -p $TB_PATH
WB_PATH=$PROJ_HOME/wandb/$EXPNAME
mkdir -p $WB_PATH

# Change for multinode config
export NODE_ADDR=$(ifconfig -a|grep inet|grep -v 127.0.0.1|grep -v inet6|awk '{print $2;}'|tr -d "addr:"|head -n 1)
export GPUS_PER_NODE=$(awk '{$1=$1;print}' $HOSTFILE|awk -F" |=" '{ranks[$1]=$NF;}END{print ranks["'$NODE_ADDR'"];}')
export NNODES=$(awk '{$1=$1;print}' $HOSTFILE | wc -l)
export MASTER_ADDR=$(head -n1 $HOSTFILE | awk '{print $1;}')
export NODE_RANK=$(awk '{ranks[$1]=(FNR-1);}END{print ranks["'$NODE_ADDR'"];}' $HOSTFILE)
export MASTER_PORT=12346
WORLD_SIZE=$(($GPUS_PER_NODE * $NNODES))

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

TRAINING_ARGS="
    --train-samples 1002539063 \
    --eval-iters 0 \
    --tensor-model-parallel-size 1 \
    --pipeline-model-parallel-size 1 \
    --micro-batch-size 2 \
    --global-batch-size 1728 \
    --disable-bias-linear \
    --use-distributed-optimizer \
    --use-flash-attn
"

MIXED_PRECISION_ARGS="
    --fp16 \
    --initial-loss-scale 522893 \
    --min-loss-scale 1.0 \
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
    --split 1
"

NETWORK_ARGS="
    --num-layers 32 \
    --hidden-size 4096 \
    --num-attention-heads 32 \
    --seq-length 2048 \
    --max-position-embeddings 2048 \
    --norm-epsilon 1e-5 \
    --use-rotary-position-embeddings \
    --no-position-embedding \
    --swiglu \
    --multiple-of 256 \
    --normalization RMSNorm \
    --rotary-interleaved-patch \
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
    --clip-grad 1.0
"

LEARNING_RATE_ARGS="
    --lr 2.0e-5 \
    --min-lr 2.0e-6 \
    --lr-decay-style cosine \
    --lr-warmup-samples 3076172
"

CHECKPOINTING_ARGS="
    --save-interval 2000 \
    --save $CHECKPOINT_PATH \
    --load $CHECKPOINT_PATH
"

LOGGING_ARGS="
    --log-interval 1 \
    --wandb-save-dir $WB_PATH \
    --tensorboard-dir $TB_PATH \
    --tensorboard-log-interval 1 
"

cmd="torchrun $DISTRIBUTED_ARGS pretrain_gpt.py \
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


