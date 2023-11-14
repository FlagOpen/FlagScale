export XACC=1

export AQUILA_INFERENCE_BADBMM="true"
export AQUILA_INFERENCE_BMM="true"
export AQUILA_TRAIN_XPU="true"

export XDNN_FC_GEMM_DTYPE="int16"

#fc autotune
export XPU_FC_AUTOTUNE_FILE="fc_autotune_aquila_34b.log"

export OMP_NUM_THREADS=4

export AQUILA_INFERENCE_BADBMM="true"
export AQUILA_INFERENCE_BMM="true"
export AQUILA_TRAIN_XPU="true"

# use ccix
export BKCL_CCIX_RING=1
export BKCL_TREE_THRESHOLD=1

# bckl not use L3
export BKCL_CCIX_BUFFER_GM=1

ulimit -c 0
export XMLIR_F_XPU_ENABLED_BOOL=true
export ALLREDUCE_ASYNC=false
export ALLGATHER_ASYNC=false
export ALLREDUCE_FUSION=0
export BKCL_TIMEOUT=1800
export BKCL_SOCKET_IFNAME=ibs11

ROOT_DIR=$( dirname -- "$( readlink -f -- "$0"; )"; )
DATA_PATH=${ROOT_DIR}/../../../megatron/data

CHECKPOINT_PATH=$PROJ_HOME/checkpoints/$EXPNAME
LOAD_CHECKPOINT_PATH=$PROJ_HOME/checkpoints/$LOAD_EXPNAME
echo "LOAD_CHECKPOINT_PATH", $LOAD_CHECKPOINT_PATH
mkdir -p $CHECKPOINT_PATH
VOCAB_FILE=${ROOT_DIR}/../../../examples/aquila/tokenizer/vocab.json
MERGE_FILE=${ROOT_DIR}/../../../examples/aquila/tokenizer/merges.txt
SPECIAL_TOKENS_FILE=${ROOT_DIR}/../../../examples/aquila/tokenizer/special_tokens.txt
LOG_PATH=$PROJ_HOME/logs/$EXPNAME
mkdir -p $LOG_PATH
cp $0 $LOG_PATH/
TB_PATH='./aquila34b_tp8_pp1_fp32_tensorboard'
mkdir -p $TB_PATH
WB_PATH=$PROJ_HOME/wandb/$EXPNAME
mkdir -p $WB_PATH


export XPU_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
GPUS_PER_NODE=`echo "$XPU_VISIBLE_DEVICES" | awk -F, '{print NF}'`

# Change for multinode config
MASTER_ADDR=${MASTER_ADDR:-"192.167.5.6"}

NNODES=${1:-12}
NODE_RANK=${2:-$NODE_RANK}
MASTER_PORT=${MASTER_PORT:-23461}
echo $NNODES
echo $NODE_RANK
echo $MASTER_ADDR
echo $MASTER_PORT


DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

TRAINING_ARGS="
    --train-iters 1 \
    --dataloader-type cyclic \
    --eval-iters 0 \
    --eval-interval 20 \
    --make-vocab-size-divisible-by 64 \
    --micro-batch-size 1 \
    --global-batch-size 128 \
    --sequence-parallel \
    --findmax-opt \
    --recompute-granularity selective \
    --disable-bias-linear \
"


MIXED_PRECISION_ARGS="
    --attention-softmax-in-fp32 \
    --embedding-weights-in-fp32 \
    --no-gradient-accumulation-fusion \
    --rotary-position-embeddings-in-fp32 \
    --accumulate-allreduce-grads-in-fp32
"

DATA_ARGS="
    --train-data-path $DATA_PATH/train_convo_samples.jsonl \
    --valid-data-path $DATA_PATH/val_convo_samples.jsonl \
    --tokenizer-type AquilaTokenizer \
    --vocab-file $VOCAB_FILE \
    --vocab-size 100008\
    --special-tokens-file $SPECIAL_TOKENS_FILE \
    --merge-file $MERGE_FILE
"

NETWORK_ARGS="
    --num-layers 60 \
    --tensor-model-parallel-size 8 \
    --pipeline-model-parallel-size  12 \
    --hidden-size 6144 \
    --num-attention-heads 48 \
    --hidden-dim-multiplier 1.3 \
    --seq-length 4096 \
    --max-position-embeddings 4096 \
    --layernorm-epsilon 1e-5 \
    --layernorm-init-weight 0.3 \
    --use-rotary-position-embeddings \
    --no-position-embedding \
    --swiglu \
    --distributed-backend nccl \
    --multiple-of 4096 \
    --apply-layernorm-rms \
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
    --load $LOAD_CHECKPOINT_PATH
    --no-load-optim \
    --no-load-rng \
    --finetune
"

LOGGING_ARGS="
    --log-interval 1 \
    --tensorboard-dir $TB_PATH \
    --tensorboard-log-interval 1 \
    --wandb-dir $WB_PATH
"

PYTHONPATH=${ROOT_DIR}/../../../megatron/data
PYTHONPATH="$ROOT_DIR/../../../":$PYTHONPATH \
              python -m torch.distributed.launch $DISTRIBUTED_ARGS \
              $ROOT_DIR/../../../finetune_aquila.py \
              $TRAINING_ARGS \
              $MIXED_PRECISION_ARGS \
              $DATA_ARGS \
              $NETWORK_ARGS \
              $INITIALIZATION_ARGS \
              $REGULARIZATION_ARGS \
              $LEARNING_RATE_ARGS \
              $CHECKPOINTING_ARGS \
              $LOGGING_ARGS
