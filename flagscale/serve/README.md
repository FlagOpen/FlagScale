## Introduce

We introduce support for deploying large models with FlagScale, leveraging the Ray framework for efficient orchestration and scalability. Currently, this implementation supports the Qwen model, enabling users to easily deploy and manage large-scale machine learning services.

Future Key features include:

- Easy distributed Serve on base of eamless integration with Ray.
- Optimized resource management for large model inference.
- Simplified deployment process for the LLM and Multimodal models.

This enhancement will significantly improve the usability of FlagScale for large model deployment scenarios.

## Setup

[Install vLLM](../../README.md#setup)

## Prepare Model

[Prepare Qwen data](https://www.modelscope.cn/models/Qwen/Qwen2.5-7B-Instruct/summary)

```shell
pip install modelscope
modelscope download --model Qwen/Qwen2.5-7B-Instruct --local_dir /models/
```

## Serve run

```shell
cd FlagScale
python run.py --config-path ./examples/qwen/conf --config-name config_qwen2.5_7b action=run
```

## Serve call

```shell
curl http://127.0.0.1:4567/v1/chat/completions -H "Content-Type: application/json" -d '{
        "model": "/models/Qwen2.5-7B-Instruct",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Introduce Bruce Lee in details."}
        ]
    }'
```

## Serve stop

```shell
cd FlagScale
python run.py --config-path ./examples/qwen/conf --config-name config_qwen2.5_7b action=stop
```

## logs

Since serve is the distributed mode, the logs are stored separately. \
The default logs of are loacated in `/tmp/ray/session_latest/logs`.\
The log of each work is named as `worker-[worker_id]-[job_id]-[pid].[out|err]`.

## Config Template

Flagscale.serve will support multiple scenarios. For better performance and usage, Flagscale.serve will optimize for specific scenarios, and these optimizations can be applied through different configurations.

### Config one or more models
Now, you can config all args needed by one or more models in config file. You can also config the number of GPUs used by each model.\
Serve
```shell
python run.py --config-path ./examples/qwen/conf --config-name config_qwen2.5_multiple_models action=run
```
Call

```shell
curl http://127.0.0.1:9010/generate -H "Content-Type: application/json" -d '{
        "prompt": "Introduce Bruce Lee in details."
    }'
```
### Command Line Mode with vLLM

If origin model is excuted in command line mode with vLLM, we can use Flagscale.serve to deploy it easily.

```shell
vllm serve /models/Qwen2.5-7B-Instruct --tensor-parallel-size=1 --gpu-memory-utilization=0.9 --max-model-len=32768 --max-num-seqs=256 --port=4567 --trust-remote-code --enable-chunked-prefill
```

All the args remain the same as vLLM. Note that action args without value, like trust-remote-code and enable-chunked-prefill, are located in **action-args** block in config file.

```YAML
model_args:
  vllm_model:
    model-tag: /models/Qwen2.5-7B-Instruct
    tensor-parallel-size: 1
    gpu-memory-utilization: 0.9
    max-model-len: 32768
    max-num-seqs: 256
    port: 4567
    action-args:
      - trust-remote-code
      - enable-chunked-prefill

deploy:
  command-line-mode: true
  models:
    vllm_model:
      num_gpus: 1
```

### How to config serve parameters
***deploy*** block is used to specify the parameters of serve. The ***models*** block is used to specify the parameters of each model decorated by "serve.remote".
