python convert.py \
    --model-type mixtral \
    --loader transformer \
    --saver megatron \
    --load-dir Mixtral-8x7B-v0.1 \
    --save-dir output \
    --target-tensor-parallel-size 2 \
    --target-pipeline-parallel-size 2 \
    --target-expert-parallel-size 2 \
    --megatron-path <xxx>
