## Introduction

[FlagScale](https://github.com/FlagOpen/FlagScale.git) is a Large Language Model (LLM) toolkit based on the [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) project, which supports the Aquila LLMs from BAAI. Our primary goal is to utilize the computation resources efficiently for LLMs without sacrificing the numerical stability and model effectiveness. In the future, we will support **different LLMs on various hardware architectures**. 

The reason why we start from Megatron-LM is that it can achieve a very high-level resource utilization by combining the most comprehensive distributed training and accelerating techniques, especially for training LLMs beyond ten-billions of parameters. 

## Highlights
FlagScale provides developers with the actual configurations, optimization schemes and hyper-parameter settings for LLM training from BAAI. It also assists developers in rapidly establishing a fundamental yet complete pipeline for LLM, including training, fine-tuning, inference and serving. It has several features as follows:

- Support pre-training, fine-tuning, inference and serving scripts for the Aquila LLMs
- Provide the training schemes of the Aquila model which can guaranteed training convergence
- Support model weight conversion to Huggingface and repartition distributed optimizer
- Ensure timely synchronization with the upstream Megatron-LM project

## Quick Start

We highly recommend developers to follow the [Megatron-LM Usage](./README_original.md#contents). Here we provide instructions for Aquila-33b model: 

### Launch a training task

```
bash examples/aquila/dist_start.sh
```

### Stop a training task

```
bash examples/aquila/dist_stop.sh
```

### Convert a checkpoint before the inference and serving

```
python tools/checkpoint_util.py  --model-type GPT --load-dir <training_ckpt_dir>  --save-dir <inference_ckpt_dir> \
    --true-vocab-size 100008 --vocab-file examples/aquila/tokenizer/vocab.json --megatron-path <FlagScale_dir> \
    --target-tensor-parallel-size 1 --target-pipeline-parallel-size 1
```

### Launch a inference and serving task

```
python examples/aquila/33B/inference_auto.py --server-port <server_port> --master-process <master_port> --device "0" \
    --iteration <iter_num> --checkpoint-path <inference_ckpt_dir> --model-info "Aquila-33b"
```

### Repartition the distributed optimizer

When using the distributed optimzier, you can use the following tool to repartition the distributed optimizer if the data parallel degree is changed.

```
python tools/checkpoint_util_lite.py --conversion-type weight --model-type GPT --load-dir <load_ckpt_dir> --save-dir <save_ckpt_dir> \ 
    --true-vocab-size 100008 --vocab-file examples/aquila/tokenizer/vocab.json --megatron-path <FlagScale_dir> \
    --target-tensor-parallel-size <tp_degree> --target-pipeline-parallel-size <pp_degree>

python tools/checkpoint_util_lite.py --conversion-type optimizer --model-type GPT --load-dir <load_ckpt_dir> --save-dir <save_ckpt_dir> \
    --true-vocab-size 100008 --vocab-file examples/aquila/tokenizer/vocab.json --megatron-path <FlagScale_dir> \
    --target-tensor-parallel-size <tp_degree> --target-pipeline-parallel-size <pp_degree>
```

### From FlagScale to HuggingFace

```
python scripts/convert_megatron_unsharded_to_huggingface.py
```

## Future work

We will work with the community together on the following items:

* Release the actual used training schemes for the larger Aquila models 
* Add customized optimizations and integrate techniques from other excellent open-source projects like DeepSpeed and vLLM etc. 
* Support LLMs with different model structures 
* Support the model training with more hardware architectures

## License
This project is mainly based on the [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) project and is licensed under the [Apache License (Version 2.0)](https://github.com/FlagOpen/FlagScale/blob/main/LICENSE). This project also contains other third-party components under other open-source licenses. See the [LICENSE](https://github.com/FlagOpen/FlagScale/blob/main/LICENSE) file for more information.