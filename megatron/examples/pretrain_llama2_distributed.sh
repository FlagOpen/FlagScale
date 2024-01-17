#! /bin/bash

# Runs the "70B" parameter model
export CUDA_DEVICE_MAX_CONNECTIONS=1

GPUS_PER_NODE=8
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

CHECKPOINT_PATH=<Specify path>
TENSORBOARD_LOGS_PATH=<Specify path>
TOKENIZER_PATH=<Specify path to file>/tokenizer.model
DATA_PATH=<Specify path and file prefix>_text_document

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE \
                  --nnodes $NNODES \
                  --node_rank $NODE_RANK \
                  --master_addr $MASTER_ADDR \
                  --master_port $MASTER_PORT"

LLAMA2_70B_ARGS="--num-layers 80 \
                 --hidden-size 8192 \
                 --ffn-hidden-size 28672 \
                 --num-attention-heads 64 \
                 --micro-batch-size 1 \
                 --global-batch-size 1024 \
                 --seq-length 4096 \
                 --group-query-attention \
                 --num-query-groups 8 \
                 --max-position-embeddings 4096 \
                 --lr 0.00015 \
                 --min-lr 1.0e-5 \
                 --clip-grad 1.0 \
                 --weight-decay 1e-2 \
                 --train-iters 500000 \
                 --lr-decay-iters 320000 \
                 --lr-decay-style cosine \
                 --lr-warmup-fraction .01 \
                 --tokenizer-type Llama2Tokenizer \
                 --tokenizer-model $TOKENIZER_PATH \
                 --swiglu \
                 --use-flash-attn \
                 --normalization RMSNorm \
                 --adam-beta1 0.9 \
                 --adam-beta2 0.95 \
                 --use-rotary-position-embeddings \
                 --no-position-embedding \
                 --disable-bias-linear \
                 --attention-dropout 0 \
                 --hidden-dropout 0 \
                 --use-distributed-optimizer \
                 --untie-embeddings-and-output-weights \
                 --multiple-of 4096 \
                 --sequence-parallel \
                 --fp16"

OUTPUT_ARGS="--log-interval 1 \
             --save-interval 1000 \
             --eval-interval 100 \
             --eval-iters 10 \
             --tensorboard-dir ${CHECKPOINT_PATH}"

OTHER_ARGS="--save $CHECKPOINT_PATH \
            --data-path $DATA_PATH \
            --split 949,50,1 \
            --distributed-backend nccl"

torchrun $DISTRIBUTED_ARGS pretrain_llama.py \
       $LLAMA2_70B_ARGS \
       $OUTPUT_ARGS \
       $OTHER_ARGS
