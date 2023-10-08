## Introduction

[FlagScale](https://github.com/FlagOpen/FlagScale.git) is a Large Language Model (LLM) toolkit based on the [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) project, which supports the LLMs from Beijing Academy of Artificial Intelligence (BAAI). Our primary goal is to utilize the computation resources efficiently for LLMs without sacrificing the numerical stability and model effectiveness. In the future, we will support different LLMs on various hardware architectures. 

The reason why we start from Megatron-LM is that it can achieve a very high-level resource utilization by combining the most comprehensive distributed training and accelerating techniques, especially for training LLMs beyond ten-billions of parameters. 

## Highlights
FlagScale provides developers with the actual configurations, optimization schemes and hyper-parameter settings for LLM training from BAAI. It also assists developers in rapidly establishing a basic yet complete pipeline for LLM, including training, fine-tuning, inference and serving. It has several features as follows:

- Provide the training schemes of the Aquila models form BAAI which can guaranteed training convergence
- Support the model weight conversion to Huggingface and the distributed optimizer repartition
- Keep timely synchronization with the upstream Megatron-LM project

## News and Updates

* 2023.10.11 We release the initial version by supporting the Aquila models, and also provide the used training schemes for [Aquila-7B](./examples/aquila/7B/pretrain_aquila_7b_distributed_A800_12n_80g.sh) and [Aquila-34B](./examples/aquila/33B/pretrain_aquila_33b_distributed_A100_64n_40g.sh).

## Quick Start

We highly recommend developers to follow the [Megatron-LM Usage](./README_original.md#contents). Here we provide instructions for Aquila LLMs:

### Setup 

1. Install the Megatron-LM dependencies as the [original link](./README_original.md#setup)

2. Install the requirements for FlagScale
```
git clone git@gitee.com:baai-opensp/FlagScale.git 
cd FlagScale
pip install -r requirements.txt
```

### Pretrain the aquila model

1. Change to the aquila directory 

```
cd FlagScale/examples/aquila
```
2. Start a distributed training job 

```
bash dist_start.sh
```
Before running `dist_start.sh`, you should provide the required information: 
  * `FlagScale_HOME`: the directory of the FlagScale
  * `PROJ_HOME`: the directory for saving checkpoints, tensorboards and other information
  * `EXPNAME`: the name of the current training experiment
  * `DATA_PATH`: the path of the training datasets following the [Megatron-LM format](./README_original.md#data-preprocessing)
  * `HOSTFILE`: the hostfile of the nodes for the current training 

3. Stop a distributed training job

```
bash dist_stop.sh
```
Before running `dist_stop.sh`, you should provide the required information: 
  * `HOSTFILE`: the hostfile of the nodes for the current training 


### From FlagScale to HuggingFace

1. Change to the FlagScale directory

```
cd FlagScale 
```

2. Merge the multiple checkpoints to a single checkpoint (if needed)
```
python tools/checkpoint_util.py --model-type GPT \
        --load-dir ${LOAD_DIR} --save-dir ${SAVE_DIR} \
        --true-vocab-size 100008 --vocab-file ${FlagScale_HOME}/examples/aquila/tokenizer/vocab.json \
        --megatron-path ${FlagScale_HOME} --target-tensor-parallel-size 1 --target-pipeline-parallel-size 1
```
Please set the following variables before running the command:
  * `LOAD_DIR`: the directory for loading the original checkpoint
  * `SAVE_DIR`: the directory for saving the merged checkpoint
  * `FlagScale_HOME`: the directory of FlagScale

3. Convert the merged checkpoint to the Huggingface format 
```
export PYTHONPATH=${FlagScale_HOME}:$PYTHONPATH

python scripts/convert_megatron_unsharded_to_huggingface.py \
        --input-dir ${SAVE_DIR}/iter_${ITERATION}/mp_rank_00/ \
        --output-dir ${SAVE_DIR}/iter_${ITERATION}_hf \
        --num-layers 64 --hidden-size 6144 \
        --num-attention-heads 48 --group-query-attention --num-query-groups 8 \
        --data-type bf16 --multiple-of 4096 --hidden-dim-multiplier 1.3
```
Please set the following variables before running the command:
  * `FlagScale_HOME`: the directory of FlagScale
  * `SAVE_DIR`: the directory for loading the merged checkpoint
  * `ITERATION`: the iteration number from `latest_checkpointed_iteration.txt` in `SAVE_DIR` and need to be padded zeros to 7 digits.

Besides, you may need to change the model configurations such as `num_layers`, `hidden_size` and so on. 

### Repartition the distributed optimizer [optional] 

When using the distributed optimizer, you can use the following tool to repartition the distributed optimizer if the parallel schemes is changed during the training.

1. Change to the FlagScale directory

```
cd FlagScale 
```

2. Repartition the model weight

```
python tools/checkpoint_util_lite.py --conversion-type weight --model-type GPT --load-dir ${LOAD_DIR} --save-dir ${SAVE_DIR} \ 
    --true-vocab-size 100008 --vocab-file ${FlagScale_HOME}/examples/aquila/tokenizer/vocab.json --megatron-path  ${FlagScale_HOME} \
    --target-tensor-parallel-size ${TP} --target-pipeline-parallel-size ${PP} 
```
Please set the following variables before running the command:
  * `LOAD_DIR`: the directory for loading the original checkpoint
  * `SAVE_DIR`: the directory for saving the converted checkpoint
  * `FlagScale_HOME`: the directory of FlagScale
  * `TP`: the target tensor parallel size
  * `PP`: the target pipeline parallel size 


3. Repartition the distributed optimizer 
```
python tools/checkpoint_util_lite.py --conversion-type optimizer --model-type GPT --load-dir ${LOAD_DIR} --save-dir ${SAVE_DIR} \ 
    --true-vocab-size 100008 --vocab-file ${FlagScale_HOME}/examples/aquila/tokenizer/vocab.json --megatron-path  ${FlagScale_HOME} \
    --target-tensor-parallel-size ${TP} --target-pipeline-parallel-size ${PP} 
```
Please set the following variables before running the command **as these used in the model weight conversion**:
  * `LOAD_DIR`: the directory for loading the original checkpoint
  * `SAVE_DIR`: the directory for saving the converted checkpoint
  * `FlagScale_HOME`: the directory of FlagScale
  * `TP`: the target tensor parallel size
  * `PP`: the target pipeline parallel size 


## Future work

We will work with the community together on the following items:

* Release the actual used training schemes for more models from BAAI 
* Add customized optimizations and integrate techniques from other excellent open-source projects like DeepSpeed and vLLM etc. 
* Support LLMs with different model structures 
* Support the model training with more hardware architectures

## License
This project is mainly based on the [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) project and is licensed under the [Apache License (Version 2.0)](https://github.com/FlagOpen/FlagScale/blob/main/LICENSE). This project also contains other third-party components under other open-source licenses. See the [LICENSE](https://github.com/FlagOpen/FlagScale/blob/main/LICENSE) file for more information.
