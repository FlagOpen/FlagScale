#!/bin/bash

export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_DISABLE=0
export NCCL_IB_CUDA_SUPPORT=1
export NCCL_IB_GID_INDEX=0
export NCCL_IB_HCA=mlx5_0,mlx5_3
export NCCL_DEBUG=debug
export OMP_NUM_THREADS=4

WORLD_SIZE=1
TASK="AQUILA"
VALID_DATA=<Specify lambada path>
CHECKPOINT=<Specify checkpoints path>
VOCAB_FILE=examples/aquila/tokenizer/vocab.json
MERGE_FILE=examples/aquila/tokenizer/merges.txt
SPECIAL_TOKENS_FILE=examples/aquila/tokenizer/special_tokens.txt

DISTRIBUTED_ARGS="--nproc_per_node $WORLD_SIZE \
                  --nnodes 1 \
                  --node_rank 0 \
                  --master_addr localhost \
                  --master_port 6000"

python -m torch.distributed.launch $DISTRIBUTED_ARGS ./tasks/main.py \
               --task $TASK \
               --eval-metric loss \
               --valid-data $VALID_DATA \
    	       --tokenizer-type AquilaTokenizer \
               --vocab-file $VOCAB_FILE \
               --merge-file $MERGE_FILE \
               --special-tokens-file $SPECIAL_TOKENS_FILE  \
               --load $CHECKPOINT \
               --use-flash-attn \
               --tensor-model-parallel-size 1 \
               --pipeline-model-parallel-size 1 \
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
               --untie-embeddings-and-output-weights \
               --disable-bias-linear \
               --log-interval 1 \
               --bf16 \
               --make-vocab-size-divisible-by 64 \
               --micro-batch-size 1 \
               --no-load-optim \
               --no-load-rng
