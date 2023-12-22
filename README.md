## Introduction

[FlagScale](https://github.com/FlagOpen/FlagScale.git) is a Large Language Model (LLM) toolkit based on the [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) project, which supports the LLMs from Beijing Academy of Artificial Intelligence (BAAI). Our primary goal is to utilize the computation resources efficiently for LLMs without sacrificing the numerical stability and model effectiveness. For now, FlagScale is still in its early stage and we will work with the community together to support different LLMs on various hardware architectures. 

The reason why we start from Megatron-LM is that it can achieve a very high-level resource utilization by combining the most comprehensive distributed training and accelerating techniques, especially for training LLMs beyond ten-billions of parameters. 

## Highlights
FlagScale provides developers with the actual configurations, optimization schemes and hyper-parameter settings for LLM training from BAAI. It also assists developers in rapidly establishing a basic yet complete pipeline for LLM, including training, fine-tuning, inference and serving. It has several features as follows:

- Provide the training schemes of the Aquila models form BAAI which can guaranteed training convergence
- Support the model weight conversion to Huggingface and the distributed optimizer repartition
- Keep timely synchronization with the upstream Megatron-LM project

## News and Updates

* 2023.11.30 We release the new version (v0.2): 
  * Provide the actually used training scheme for [Aquila2-70B-Expr](./aquila/70B), including the parallel strategies, optimizations and hyper-parameter settings.
  * Support heterogeneous training on chips of different generations with the same architecture or compatible architectures, including NVIDIA GPUs and Iluvatar CoreX chips. 
  * Support training on chinese domestic hardwares, including Iluvatar CoreX and Baidu KUNLUN chips.

* 2023.10.11 We release the initial version (v0.1) by supporting the Aquila models, and also provide our actually used training schemes for [Aquila2-7B](./aquila/7B/pretrain_aquila_7b_distributed_A800_12n_80g.sh) and [Aquila2-34B](./aquila/34B/pretrain_aquila_34b_distributed_A100_64n_40g.sh), including the parallel strategies, optimizations and hyper-parameter settings.

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

### Pretrain the Aquila model (new)
1. Change to the aquila directory 

```
cd FlagScale
```
2. Start a distributed training job 

```
python run.py --config <config_path> --extra_config <extra_config_path> --action <action> --stop-key <stop_key>
```

* `--config <config_path>`: Specifies the path to the main configuration JSON file. This file contains key-value pairs that configure the experiment. These pairs mirror the arguments used in Megatron-LM, with hyphens (`-`) replaced by underscores (`_`). For a comprehensive list of configurations, refer to the [source file](./megatron/megatron/arguments.py). The key-value pairs can be organized into different sections as needed. These [pre-defined configuration](predefined_args_megatron.json) also should be provided for the experiment:
  * `experiment`: Contains configuration options for the experiment setup.
    * `exp_name`: Defines the experiment's name. A directory with this name will be created under `log_dir` to store all experiment artifacts. Default is `default`.
    * `log_dir`: Specifies the directory for storing experiment logs. Default is the `logs` directory of the current working directory.
    * `hostfile`: Specifies a file containing a list of hosts for running the experiment. Default is `localhost`.
    * `no_shared_fs`: A boolean flag indicating whether the file system is shared between hosts. Default is `false`.
    * `ssh_port`: Specifies the port for SSH connections when running the experiment across multiple hosts. Default is `22`.
    * `auto_mkdir`: A boolean flag indicating whether directories (for save, load, tensorboard, and wandb) should be automatically created. Default is `true`.
    * `shell_cmds`: Specifies shell commands to be executed before running the experiment. Defualt is `""`.
  * `launch`: Contains configuration options for launching the experiment using `torchrun`.
  * `env_vars`: Contains environment variables to be set for the experiment.

* `--extra_config <extra_config_path>` (optional): Specifies the path to an additional configuration file. This JSON file can contain additional configuration options not present in the main configuration file. Options in this file will override those in the main configuration file if they have the same keys.

* `--action <action>` (optional): Specifies the action to perform. Default is `run` if none is specified. Possible values are:
  * `generate`: Generates the experiment shell script based on the provided configuration files and stops. The generated script is saved to `log_dir`.
  * `run`: Generates the experiment shell script and runs the experiment immediately. Both the generated script and the experiment log are saved to `log_dir`. The experiment runs in the background.
  * `stop`: Stops the experiment. The `--stop-key` argument need be provided with this action.

* `--stop_key <stop_key>` (optional): Used with the `stop` action to specify the process name of the experiment process to be stopped. Default is `torchrun` if none is specified.

Here are some examples of how to use these arguments:

```shell
# Run the experiment only with the base configuration
python run.py \
  --config aquila/7B/pretrain_aquila_7b_distributed.json

# Run the experiment only with the base configuration and the extra configuration
python run.py \
  --config aquila/7B/pretrain_aquila_7b_distributed.json \
  --extra-config apu_demo/scripts/pretrain_aquila_7b_distributed_extra.json

# Generate the experiment script from the base configuration and the extra configuration
python run.py \
  --config aquila/7B/pretrain_aquila_7b_distributed.json \
  --extra-config apu_demo/scripts/pretrain_aquila_7b_distributed_extra.json \
  --action generate 

# Stop the experiment
python run.py \
  --config aquila/7B/pretrain_aquila_7b_distributed.json \
  --action stop 
```


### Pretrain the Aquila model (old)

1. Change to the aquila directory 

```
cd FlagScale/aquila
```
2. Start a distributed training job 

```
bash dist_start.sh
```
Before running `dist_start.sh`, you should provide the required information: 
  * `FlagScale_HOME`: the directory of the FlagScale.
  * `PROJ_HOME`: the directory for saving checkpoints, tensorboards and other information.
  * `EXPNAME`: the name of the current training experiment.
  * `DATA_PATH`: the path of the training datasets following the [Megatron-LM format](./README_original.md#data-preprocessing). For quickly running the pretraining process, we also provide a small processed data ([bin](https://model.ks3-cn-beijing.ksyuncs.com/nlpdata/pile_wikipedia_demo.bin) and [idx](https://model.ks3-cn-beijing.ksyuncs.com/nlpdata/pile_wikipedia_demo.idx)) from the [Pile](https://pile.eleuther.ai/) dataset.
  * `HOSTFILE`: the hostfile of the nodes for the current training, which consists of a list of hostnames and slot counts. For example:
    ```
    hostnames-1/IP-1 slots=8
    hostnames-2/IP-2 slots=8
    ```
    These hostnames or IPs represent machines accessible via passwordless SSH and the slots specify the number of GPUs available on that machine.
  * `SCRIPT_FILE`: the actually used training script of the current job where you can change the specific configurations as needed. For example, you should change `--train-samples` to match the small demo dataset we provided above. 

3. Stop a distributed training job

```
bash dist_stop.sh
```
Before running `dist_stop.sh`, you should provide the required information: 
  * `HOSTFILE`: the hostfile of the nodes for the current training. 

### Do the heterogenous training 

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
cd FlagScale 
```

2. Merge the multiple checkpoints to a single checkpoint (if needed)
```
python tools/checkpoint_util.py --model-type GPT \
        --load-dir ${LOAD_DIR} --save-dir ${SAVE_DIR} \
        --true-vocab-size 100008 --vocab-file ${FlagScale_HOME}/aquila/tokenizer/vocab.json \
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
cd FlagScale
```

2. Merge the multiple checkpoints to a single checkpoint (as needed)
```
python tools/checkpoint_util.py --model-type GPT \
        --load-dir ${LOAD_DIR} --save-dir ${SAVE_DIR} \
        --true-vocab-size 100008 --vocab-file ${FlagScale_HOME}/aquila/tokenizer/vocab.json \
        --megatron-path ${FlagScale_HOME} --target-tensor-parallel-size 1 --target-pipeline-parallel-size 1
```
Please set the following variables before running the command:
  * `LOAD_DIR`: the directory for loading the original checkpoint.
  * `SAVE_DIR`: the directory for saving the merged checkpoint.
  * `FlagScale_HOME`: the directory of FlagScale.

3. Serve the Aquila2 model by the below script. Here we take the Aquila2-34B as an example and assume you have an A800-80G GPU.
``` 
python examples/aquila/34B/inference_auto.py \
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
cd FlagScale 
```

2. Repartition the model weight

```
python tools/checkpoint_util_lite.py --conversion-type weight --model-type GPT --load-dir ${LOAD_DIR} --save-dir ${SAVE_DIR} \ 
    --true-vocab-size 100008 --vocab-file ${FlagScale_HOME}/aquila/tokenizer/vocab.json --megatron-path  ${FlagScale_HOME} \
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
    --true-vocab-size 100008 --vocab-file ${FlagScale_HOME}/aquila/tokenizer/vocab.json --megatron-path  ${FlagScale_HOME} \
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
