#!/bin/bash
# This example will start serving the 345M model.
DISTRIBUTED_ARGS="--nproc_per_node 1 \
                  --nnodes 1 \
                  --node_rank 0 \
                  --master_addr localhost \
                  --master_port 6000"

# CHECKPOINT=/data/ayl/workspace/aquila-7b/checkpoints/aquila_7b_date_23-06-30_time_14-59-24
CHECKPOINT=/share/project/ayl/FlagScale/tmp/
# CHECKPOINT=/share/project/ayl/FlagScale/tmp/iter_0030000/mp_rank_00
VOCAB_FILE=examples/aquila/tokenizer/vocab.json
MERGE_FILE=examples/aquila/tokenizer/merges.txt

export CUDA_DEVICE_MAX_CONNECTIONS=1


torchrun $DISTRIBUTED_ARGS tools/run_text_generation_uvicorn_server.py   \
       --make-vocab-size-divisible-by 64 \
       --use-flash-attn \
       --apply-layernorm-rms \
       --sequence-parallel \
       --tensor-model-parallel-size 1  \
       --pipeline-model-parallel-size 1  \
       --num-layers 60  \
       --hidden-size 6144  \
       --hidden-dim-multiplier 1.3 \
       --load ${CHECKPOINT}  \
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
       --bf16  \
       --micro-batch-size 1  \
       --seq-length 4096  \
       --out-seq-length 3000  \
       --temperature 1.0  \
       --vocab-file $VOCAB_FILE  \
       --merge-file $MERGE_FILE  \
       --top_p 0.9  \
       --seed 42

       #--apply-layernorm-rms \
