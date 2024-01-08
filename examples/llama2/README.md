# LLaMA2 MODEL

## Table of contents
- [1. Training Setup](#1-training-setup)
- [2. Configurations](#2-configurations)

## 1. Training setup
<a id="markdown-training-setup" name="training-setup"></a>

To run the model using a docker container run it as follows
```
PYTORCH_IMAGE=nvcr.io/nvidia/pytorch:23.09-py3
CHECKPOINT_PATH="" #<Specify path>
TENSORBOARD_LOGS_PATH=""#<Specify path>
TOKENIZER_PATH="" #<Specify path to file>/tokenizer.model
DATA_PATH="" #<Specify path and file prefix>_text_document

docker run \
  --gpus=all \
  --ipc=host \
  --workdir /workspace/megatron-lm \
  -v /path/to/data:/path/to/data \
  -v /path/to/megatron-lm:/workspace/megatron-lm \
  megatron-lm nvcr.io/nvidia/pytorch:23.09-py3 \
  bash /examples/llama2/train_llama2_70b_distributed.sh $CHECKPOINT_PATH $TENSORBOARD_LOGS_PATH $TOKENIZER_PATH $DATA_PATH "

```
NOTE: Depending on the environment you are running it the above command might like slightly different.


## 2. Configurations
<a id="markdown-configurations" name="configurations"></a>
The example in this folder shows you how to run 70B model. There are other configs you could run as well

### 7B 
```
       --num-layers 32 \
       --hidden-size 4096 \
       --num-attention-heads 32 \
       --seq-length 4096 \
       --ffn-hidden-size 11008 \
       --tensor-model-parallel-size 1 \
       --pipeline-model-parallel-size 1 \
```

### 13B 
```
       --num-layers 40 \
       --hidden-size 5120 \
       --num-attention-heads 40 \
       --seq-length 4096 \
       --ffn-hidden-size 13824 \
       --tensor-model-parallel-size 2 \
       --pipeline-model-parallel-size 1 \
```
