## Introduction

[FlagScale](https://github.com/FlagOpen/FlagScale.git) is a comprehensive toolkit for large-scale models, developed with the support of the Beijing Academy of Artificial Intelligence (BAAI). It builds upon open-source projects such as [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) and [vllm](https://github.com/vllm-project/vllm). 

Our primary objective with FlagScale is to optimize the use of computational resources for large models, while maintaining numerical stability and model effectiveness. Currently, FlagScale is in its early stages of development. We are actively collaborating with the community to enhance its capabilities, with the aim to support a variety of large models across diverse hardware architectures.

## Highlights
FlagScale provides developers with the actual configurations, optimization schemes and hyper-parameter settings for the large model training from BAAI. It also assists developers in rapidly establishing a basic yet complete pipeline for LLM, including training, fine-tuning, inference and serving. It has several features as follows:

- Provide the training schemes of the Aquila models form BAAI which can guaranteed training convergence
- Support the model weight conversion to Huggingface and the distributed optimizer repartition
- Keep timely synchronization with the upstream projects

## News and Updates

* 2024.4.11 🔥 We release the new version ([v0.3](https://github.com/FlagOpen/FlagScale/tree/release/v0.3)): 
  * Accomplish the heterogeneous hybrid training of the Aquila2-70B-Expr model on a cluster utilizing a combination of NVIDIA and Iluvatar chips.
  * Provide the training of the Aquila2 series across a variety of AI chips from six distinct manufacturers.

* 2023.11.30 We release the new version (v0.2): 
  * Provide the actually used training scheme for [Aquila2-70B-Expr](./examples/aquila/70B), including the parallel strategies, optimizations and hyper-parameter settings.
  * Support heterogeneous training on chips of different generations with the same architecture or compatible architectures, including NVIDIA GPUs and Iluvatar CoreX chips. 
  * Support training on chinese domestic hardware, including Iluvatar CoreX and Baidu KUNLUN chips.

* 2023.10.11 We release the initial version (v0.1) by supporting the Aquila models, and also provide our actually used training schemes for [Aquila2-7B](./examples/aquila/7B/pretrain_aquila_7b_distributed_A800_12n_80g.sh) and [Aquila2-34B](./examples/aquila/34B/pretrain_aquila_34b_distributed_A100_64n_40g.sh), including the parallel strategies, optimizations and hyper-parameter settings.

## Quick Start

We highly recommend developers to follow the [Megatron-LM Usage](./megatron/README.md#contents). Here we provide instructions for Aquila LLMs:

### Setup 

1. Install the Megatron-LM dependencies as the [original link](./megatron/README.md#setup)

2. Install the requirements for FlagScale
```
git clone git@gitee.com:baai-opensp/FlagScale.git 
cd FlagScale
pip install -r requirements.txt
```

### Pretrain the Aquila model

1. Start a distributed training job 

```
python run.py --config-path ./examples/aquila/conf --config-name config
```

FlagScale leverages [Hydra](https://github.com/facebookresearch/hydra) for configuration management. The YAML configuration is structured into four key sections:

  * `experiment`: Defines the experiment directory, backend, and other related environmental configurations.
  * `system`: Details execution parameters, such as parallel strategies and precision of operations.
  * `model`: Describes the model's architecture along with its associated hyperparameters.
  * `data`: Specifies configurations related to the data used by the model.

All valid configurations correspond to the arguments used in Megatron-LM, with hyphens (-) replaced by underscores (_). For a complete list of available configurations, please refer to the Megatron-LM arguments source [file](./megatron/megatron/arguments.py).

To kickstart the training process, consider using the existing YAML files in the [examples](./examples/aquila/conf) folder as a template. Simply copy and modify these files to suit your needs. Please note the following important configurations:

  * `exp_dir`: the directory for saving checkpoints, tensorboards and other logging information.
  * `hostfile`: the hostfile file path for the current training, which consists of a list of hostnames and slot counts. For example:
    ```
    hostnames-1/IP-1 slots=8
    hostnames-2/IP-2 slots=8
    ```
    These hostnames or IPs represent machines accessible via passwordless SSH and the slots specify the number of GPUs available on that machine.

  * `data_path`: the path of the training datasets following the [Megatron-LM format](./megatron/README.md#data-preprocessing). For quickly running the pretraining process, we also provide a small processed data ([bin](https://model.ks3-cn-beijing.ksyuncs.com/nlpdata/pile_wikipedia_demo.bin) and [idx](https://model.ks3-cn-beijing.ksyuncs.com/nlpdata/pile_wikipedia_demo.idx)) from the [Pile](https://pile.eleuther.ai/) dataset.

2. Stop a distributed training job

```
python run.py --config-path ./examples/aquila/conf --config-name config action=stop
```

### Do the heterogenous training 

Please checkout the [v0.3](https://github.com/FlagOpen/FlagScale/tree/release/v0.3) branch first and follow the instructions below.

It is very simple to do the heterogeneous training on chips of different generations with the same architecture or compatible architectures. You only need to follow the steps below and everything else just remains the same as the above homogeneous training. In addition, you can also refer to the examples [1](./examples/aquila/34B/pretrain_aquila_34b_distributed_A800_16n_80g_A100_48n_40g_hetero_pp.sh), [2](./examples/aquila/34B/pretrain_aquila_34b_distributed_A800_16n_80g_A100_48n_40g_hetero_dp.sh), [3](./examples/aquila/70B/pretrain_aquila_70b_distributed_A800_16n_80g_A100_48n_40g_hetero_pp.sh) for better understanding.

1. Extend the hostfile

   Before doing the heterogenous training, you should extend the hostfile by adding the device types. You are free to choose the identifier strings for these device types, but please ensure they are not duplicated. 

    ```
    hostnames-1/IP-1 slots=8 typeA
    hostnames-2/IP-2 slots=8 typeB
    ```

2. Add the heterogeneous configuration
   * If you choose the heterogenous pipeline parallelism mode, please set the following configurations: 
      * `hetero-mode`: specify the heterogenous training mode `pp`.
      * `hetero-current-device-type`: specify the device type of the current node.
      * `hetero-device-types`: specify all the device types used in this training.
      * `hetero-pipeline-stages`: specify the stage splitting configuration. For example, given `2 4 4 3 5 5 5`, the total pipeline parallel size is `2 + 3 = 5`, the total number of the model layers is `4 + 4 + 5 + 5 + 5 = 23`, the pipeline parallel size for the first device type in the `hetero-device-types` list is `2` and the pipeline parallel size for the second device type in the `hetero-device-types` is list `3`. 

   * If you choose the heterogenous data parallelism mode, please set the following configurations:
      * `hetero-mode`: specify the heterogenous training mode `dp`.
      * `hetero-current-device-type`: specify the device type of the current node.
      * `hetero-device-types`: specify all the device types used in this training.
      * `hetero-micro-batch-sizes`: specify the micro batch size splitting configuration. For example, given `2 1 3 2`, the total data parallel size is `2 + 3 = 5` and the micro batch size for each training iteration is `2 * 1 + 3 * 2 = 8`, the data parallel size for the first device type in the `hetero-device-types` list is `2` and the data parallel size for the second device type in the `hetero-device-types` is `3` list. 
      * **Remove** the `micro-batch-size` configuration because `hetero-micro-batch-sizes` works as the same purpose.  

### From FlagScale to HuggingFace

1. Change to the FlagScale directory

```
cd FlagScale/megatron 
```

2. Merge the multiple checkpoints to a single checkpoint (if needed)
```
python tools/checkpoint_util.py --model-type GPT \
        --load-dir ${LOAD_DIR} --save-dir ${SAVE_DIR} \
        --true-vocab-size 100008 --vocab-file ${FlagScale_HOME}/examples/aquila/tokenizer/vocab.json \
        --megatron-path ${FlagScale_HOME} --target-tensor-parallel-size 1 --target-pipeline-parallel-size 1
```
Please set the following variables before running the command:
  * `LOAD_DIR`: the directory for loading the original checkpoint.
  * `SAVE_DIR`: the directory for saving the merged checkpoint.
  * `FlagScale_HOME`: the directory of FlagScale.

3. Convert the merged checkpoint to the Huggingface format 
```
export PYTHONPATH=${FlagScale_HOME}:$PYTHONPATH

python scripts/convert_megatron_unsharded_to_huggingface.py \
        --input-dir ${SAVE_DIR}/iter_${ITERATION}/mp_rank_00/ \
        --output-dir ${SAVE_DIR}/iter_${ITERATION}_hf \
        --num-layers 60 --hidden-size 6144 \
        --num-attention-heads 48 --group-query-attention --num-query-groups 8 \
        --data-type bf16 --multiple-of 4096 --hidden-dim-multiplier 1.3
```
Please set the following variables before running the command:
  * `FlagScale_HOME`: the directory of FlagScale.
  * `SAVE_DIR`: the directory for loading the merged checkpoint.
  * `ITERATION`: the iteration number from `latest_checkpointed_iteration.txt` in `SAVE_DIR` and need to be padded zeros to 7 digits.

Note that the above configuration is for converting Aquila-34B and you may need to change the model configurations such as `num_layers` and`hidden_size` as needed.  

### Serve a model

1. Change to the FlagScale directory

``` python
cd FlagScale/megatron
```

2. Merge the multiple checkpoints to a single checkpoint (as needed)
```
python tools/checkpoint_util.py --model-type GPT \
        --load-dir ${LOAD_DIR} --save-dir ${SAVE_DIR} \
        --true-vocab-size 100008 --vocab-file ${FlagScale_HOME}/examples/aquila/tokenizer/vocab.json \
        --megatron-path ${FlagScale_HOME} --target-tensor-parallel-size 1 --target-pipeline-parallel-size 1
```
Please set the following variables before running the command:
  * `LOAD_DIR`: the directory for loading the original checkpoint.
  * `SAVE_DIR`: the directory for saving the merged checkpoint.
  * `FlagScale_HOME`: the directory of FlagScale.

3. Serve the Aquila2 model by the below script. Here we take the Aquila2-34B as an example and assume you have an A800-80G GPU.
``` 
python ../examples/aquila/34B/inference_auto.py \
       --server-port ${SERVER_PORT} --master-process ${MASTER_PORT} \
       --device "0" --iteration -1 --checkpoint-path "${CKPT_DIR}" \
       --model-info "Aquila-34b"
```
Please set the following variables before running the command:
  * `SERVER_PORT`: the server port for serving the model.
  * `MASTER_PORT`: the port of the master process.
  * `CKPT_DIR`: the directory for loading the merged checkpoint.

4. After you have served an Aquila model successfully, you can send a request to do the testing. 
```
python tools/test/test_api_flask.py
```

### Repartition the distributed optimizer [optional] 

When using the distributed optimizer, you can use the following tool to repartition the distributed optimizer if the parallel schemes is changed during the training.

1. Change to the FlagScale directory

```
cd FlagScale/megatron
```

2. Repartition the model weight

```
python tools/checkpoint_util_lite.py --conversion-type weight --model-type GPT --load-dir ${LOAD_DIR} --save-dir ${SAVE_DIR} \ 
    --true-vocab-size 100008 --vocab-file ${FlagScale_HOME}/examples/aquila/tokenizer/vocab.json --megatron-path  ${FlagScale_HOME} \
    --target-tensor-parallel-size ${TP} --target-pipeline-parallel-size ${PP} 
```
Please set the following variables before running the command:
  * `LOAD_DIR`: the directory for loading the original checkpoint.
  * `SAVE_DIR`: the directory for saving the converted checkpoint.
  * `FlagScale_HOME`: the directory of FlagScale.
  * `TP`: the target tensor parallel size.
  * `PP`: the target pipeline parallel size. 


3. Repartition the distributed optimizer 
```
python tools/checkpoint_util_lite.py --conversion-type optimizer --model-type GPT --load-dir ${LOAD_DIR} --save-dir ${SAVE_DIR} \ 
    --true-vocab-size 100008 --vocab-file ${FlagScale_HOME}/examples/aquila/tokenizer/vocab.json --megatron-path  ${FlagScale_HOME} \
    --target-tensor-parallel-size ${TP} --target-pipeline-parallel-size ${PP} 
```
Please set the following variables before running the command **as these used in the model weight conversion**:
  * `LOAD_DIR`: the directory for loading the original checkpoint.
  * `SAVE_DIR`: the directory for saving the converted checkpoint.
  * `FlagScale_HOME`: the directory of FlagScale.
  * `TP`: the target tensor parallel size.
  * `PP`: the target pipeline parallel size. 


## Future work

We will work with the community together on the following items:

* Release the actual used training schemes for more models from BAAI 
* Add customized optimizations and integrate techniques from other excellent open-source projects like DeepSpeed and vLLM etc. 
* Support LLMs with different model structures 
* Support the model training with more hardware architectures

## License
This project is mainly based on the [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) project and is licensed under the [Apache License (Version 2.0)](https://github.com/FlagOpen/FlagScale/blob/main/LICENSE). This project also contains other third-party components under other open-source licenses. See the [LICENSE](https://github.com/FlagOpen/FlagScale/blob/main/LICENSE) file for more information.
