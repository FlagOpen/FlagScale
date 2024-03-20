# sparse model change the parallel config
python convert.py \
    --model-type mixtral \
    --loader transformers \
    --saver megatron \
    --load-dir Mixtral-8x7B-v0.1 \
    --save-dir output \
    --target-tensor-parallel-size 2 \
    --target-pipeline-parallel-size 2 \
    --target-expert-parallel-size 2 \
    --target-params-dtype fp32 \
    --megatron-path <xxx>


# dense model convert to sparse model, mlp weight copy to all experts weight
# padding vocab_size with default value 64
python convert.py \
    --model-type mistral mixtral \
    --loader transformers \
    --saver megatron \
    --load-dir Mistral-7B-v0.1 \
    --save-dir output \
    --target-tensor-parallel-size 2 \
    --target-pipeline-parallel-size 2 \
    --target-expert-parallel-size 2 \
    --target-params-dtype fp32 \
    --target-num-experts 8 \
    --true-vocab-size 151851 \
    --megatron-path <xxx>
