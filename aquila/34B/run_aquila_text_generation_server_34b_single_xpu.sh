#!/bin/bash
# This example will start serving the 345M model.
DISTRIBUTED_ARGS="--nproc_per_node 2 \
                  --nnodes 1 \
                  --node_rank 0 \
                  --master_addr localhost \
                  --master_port 6000"

VOCAB_FILE=../aquila/tokenizer/vocab.json
MERGE_FILE=../aquila/tokenizer/merges.txt
SPECIAL_TOKENS_FILE=../aquila/tokenizer/special_tokens.txt

export XPU_FC_AUTOTUNE_FILE="fc_autotune.log"


export XMLIR_D_XPU_L3_SIZE=0
export CUDA_DEVICE_MAX_CONNECTIONS=1
export LD_PRELOAD="/libtcmalloc_minimal.so"
export XDNN_FC_GEMM_DTYPE="float32"
export XDNN_USE_FAST_SWISH="true"
export XMLIR_XDNN_PYTORCH_CHECK_ENABLE_FALLBACK_BOOL=false

export BKCL_CCIX_RING=1
export BKCL_TREE_THRESHOLD=1

torchrun $DISTRIBUTED_ARGS tools/run_text_generation_uvicorn_server_single_thread.py   \
       --make-vocab-size-divisible-by 64 \
       --apply-layernorm-rms \
       --tensor-model-parallel-size 2  \
       --pipeline-model-parallel-size 1  \
       --num-layers 2  \
       --hidden-size 6144  \
       --hidden-dim-multiplier 1.3 \
       --disable-bias-linear \
       --layernorm-epsilon 1e-5 \
       --layernorm-init-weight 0.3 \
       --num-attention-heads 48  \
       --group-query-attention \
       --num-query-groups 8 \
       --max-position-embeddings 4096  \
       --use-rotary-position-embeddings \
       --no-position-embedding \
       --swiglu \
       --multiple-of 4096 \
       --untie-embeddings-and-output-weights \
       --tokenizer-type AquilaTokenizer  \
       --fp16  \
       --micro-batch-size 1  \
       --seq-length 4096  \
       --out-seq-length 3000  \
       --temperature 1.0  \
       --vocab-file $VOCAB_FILE  \
       --merge-file $MERGE_FILE  \
       --special-tokens-file $SPECIAL_TOKENS_FILE  \
       --top_p 0.9  \
       --seed 42 \
       --no-gradient-accumulation-fusion \
       --no-bias-gelu-fusion \
       --disable-bias-linear \
       --server-port 5060
