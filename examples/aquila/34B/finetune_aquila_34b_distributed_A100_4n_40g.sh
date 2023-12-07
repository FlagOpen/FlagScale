#!/bin/bash

export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_DISABLE=0
export NCCL_IB_CUDA_SUPPORT=1
export NCCL_IB_GID_INDEX=0
export NCCL_IB_HCA=mlx5_2,mlx5_5
export NCCL_DEBUG=INFO
export OMP_NUM_THREADS=4
export CUDA_DEVICE_MAX_CONNECTIONS=1
export GLOO_SOCKET_IFNAME=eth0

# The format of train and valid data for finetuning is jsonl. 
set -u
  PROJ_HOME=$1
  EXPNAME=$2
  LOAD_EXPNAME=$3
  HOSTFILE=$4
  TRAIN_DATA_PATH=$5
  VALID_DATA_PATH=$6
set +u

CHECKPOINT_PATH=$PROJ_HOME/checkpoints/$EXPNAME
LOAD_CHECKPOINT_PATH=$PROJ_HOME/checkpoints/$LOAD_EXPNAME
echo "LOAD_CHECKPOINT_PATH", $LOAD_CHECKPOINT_PATH
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
export NNODES=$(cat $HOSTFILE | wc -l)
export MASTER_ADDR=$(head -n1 $HOSTFILE | awk '{print $1;}')
export NODE_RANK=$(awk '{ranks[$1]=(FNR-1);}END{print ranks["'$NODE_ADDR'"];}' $HOSTFILE)
export MASTER_PORT=12345
WORLD_SIZE=$(($GPUS_PER_NODE * $NNODES))

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

# Change iters based on dataset size, epoch num and global bsz
TRAINING_ARGS="
    --train-iters 14700 \
    --dataloader-type cyclic \
    --eval-iters 0 \
    --eval-interval 20 \
    --tensor-model-parallel-size 4 \
    --pipeline-model-parallel-size 4 \
    --make-vocab-size-divisible-by 64 \
    --micro-batch-size 1 \
    --global-batch-size 128 \
    --disable-bias-linear \
    --sequence-parallel \
    --recompute-granularity 'full' \
    --recompute-method 'uniform' \
    --use-distributed-optimizer
"

MIXED_PRECISION_ARGS="
    --bf16 \
    --attention-softmax-in-fp32 \
    --accumulate-allreduce-grads-in-fp32
"

DATA_ARGS="
    --train-data-path $TRAIN_DATA_PATH \
    --valid-data-path $VALID_DATA_PATH \
    --tokenizer-type AquilaTokenizer \
    --vocab-file $VOCAB_FILE \
    --vocab-size 100008\
    --special-tokens-file $SPECIAL_TOKENS_FILE \
    --merge-file $MERGE_FILE
"

NETWORK_ARGS="
    --num-layers 60 \
    --hidden-size 6144 \
    --num-attention-heads 48 \
    --group-query-attention \
    --num-query-groups 8 \
    --hidden-dim-multiplier 1.3 \
    --seq-length 4096 \
    --max-position-embeddings 4096 \
    --norm-epsilon 1e-5 \
    --norm-init-weight 0.3 \
    --use-rotary-position-embeddings \
    --no-position-embedding \
    --swiglu \
    --multiple-of 4096 \
    --normalization RMSNorm \
    --untie-embeddings-and-output-weights
"

INITIALIZATION_ARGS="
    --init-method-std 0.02 \
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
    --lr 9.65e-6 \
    --lr-decay-style linear \
    --lr-warmup-fraction 0.1 \
    --min-lr 0.0
"

CHECKPOINTING_ARGS="
    --save-interval 2000 \
    --save $CHECKPOINT_PATH \
    --load $LOAD_CHECKPOINT_PATH
    --no-load-optim \
    --no-load-rng \
    --finetune
"

LOGGING_ARGS="
    --log-interval 1 \
    --tensorboard-dir $TB_PATH \
    --tensorboard-log-interval 1 \
    --wandb-save-dir $WB_PATH
"

cmd="torchrun $DISTRIBUTED_ARGS finetune_aquila.py \
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


