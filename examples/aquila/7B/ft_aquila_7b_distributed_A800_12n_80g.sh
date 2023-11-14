export XACC=1

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
export BKCL_SOCKET_IFNAME=bond0

ROOT_DIR=$( dirname -- "$( readlink -f -- "$0"; )"; )
DATA_PATH=${ROOT_DIR}/../../../megatron/data
CHECKPOINT_PATH=/XMLIR/zhiyuan/baidu/xpu/XMLIR/baidu/hac-aiacc/Megatron/checkpoints/aquilaChat-7b-tp8-pp1-dp1-fp32_random
CHECKPOINT_SAVE_PATH=${ROOT_DIR}/checkpoints/megatron_aquila_7b_checkpoint_tp8_pp1_fp32_save
mkdir -p $CHECKPOINT_SAVE_PATH
VOCAB_FILE=${ROOT_DIR}/../../../examples/aquila/tokenizer/vocab.json
MERGE_FILE=${ROOT_DIR}/../../../examples/aquila/tokenizer/merges.txt
SPECIAL_TOKENS_FILE=${ROOT_DIR}/../../../examples/aquila/tokenizer/special_tokens.txt
TENSORBOARD_PATH='./aquila7b_tp8_pp1_fp32_tensorboard'
LOG_PATH=$PROJ_HOME/logs/$EXPNAME
mkdir -p $LOG_PATH
cp $0 $LOG_PATH/

WB_PATH=$PROJ_HOME/wandb/$EXPNAME
mkdir -p $WB_PATH

export XPU_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
GPUS_PER_NODE=`echo "$XPU_VISIBLE_DEVICES" | awk -F, '{print NF}'`

# Change for multinode config
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-23461}
NNODES=${1:-1}
NODE_RANK=${2:-0}

WORLD_SIZE=$(($GPUS_PER_NODE * $NNODES))

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
    --save-interval 1000 \
    --eval-interval 500 \
    --tensor-model-parallel-size 8 \
    --pipeline-model-parallel-size 1 \
    --make-vocab-size-divisible-by 8
    --micro-batch-size 2 \
    --global-batch-size 16 \
    --disable-bias-linear
"


MIXED_PRECISION_ARGS="
    --loss-scale 0 \
    --initial-loss-scale 16 \
    --embedding-weights-in-fp32 \
    --attention-softmax-in-fp32 \
    --no-gradient-accumulation-fusion \
    --no-async-tensor-model-parallel-allreduce \
    --accumulate-allreduce-grads-in-fp32
"


DATA_ARGS="
    --train-data-path $DATA_PATH/train_convo_samples.jsonl \
    --valid-data-path $DATA_PATH/val_convo_samples.jsonl \
    --data-impl mmap \
    --tokenizer-type AquilaTokenizer \
    --vocab-file $VOCAB_FILE \
    --merge-file $MERGE_FILE \
    --special-tokens-file $SPECIAL_TOKENS_FILE \
    --vocab-size 100008
"


NETWORK_ARGS="
    --num-layers 32 \
    --hidden-size 4096 \
    --num-attention-heads 32 \
    --seq-length 2048 \
    --ffn-hidden-size 11008 \
    --max-position-embeddings 2048 \
    --layernorm-epsilon 1e-5 \
    --use-rotary-position-embeddings \
    --rotary-position-embeddings-in-fp32 \
    --no-position-embedding \
    --no-bias-gelu-fusion \
    --disable-bias-linear \
    --swiglu \
    --multiple-of 256 \
    --apply-layernorm-rms \
    --sequence-parallel \
    --findmax-opt \
    --distributed-backend nccl \
    --no-query-key-layer-scaling \
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
    --optimizer adam \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --clip-grad 1.0
"

LEARNING_RATE_ARGS="
    --lr 2.0e-5 \
    --min-lr 2.0e-6 \
    --lr-decay-style cosine \
    --lr-decay-iters 80 \
    --lr-warmup-fraction .01 \
    --memory-saving
"

CHECKPOINTING_ARGS="
    --save-interval 2000 \
    --save $CHECKPOINT_SAVE_PATH \
    --load $CHECKPOINT_PATH \
    --no-load-optim \
    --no-load-rng \
    --finetune
"

LOGGING_ARGS="
    --log-interval 1 \
    --tensorboard-dir $TENSORBOARD_PATH \
    --tensorboard-queue-size 5 "


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

