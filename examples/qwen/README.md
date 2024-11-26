## Introduce
This pull request introduces support for deploying large models with FlagScale, leveraging the Ray framework for efficient orchestration and scalability. Currently, this implementation supports the Qwen model, enabling users to easily deploy and manage large-scale machine learning services.

Future Key features include:

- Easy distributed Serve on base of eamless integration with Ray.
- Optimized resource management for large model inference.
- Simplified deployment process for the LLM and Multimodal models.

This enhancement will significantly improve the usability of FlagScale for large model deployment scenarios.


## Setup
[Install vLLM](../../../README.md#setup)

[Prepare Qwen data](https://www.modelscope.cn/models/Qwen/Qwen2.5-7B-Instruct/summary)

```shell
pip install modelscope
modelscope download --model Qwen/Qwen2.5-7B-Instruct --local_dir /models/
```


## Serve run
```shell
cd FlagScale
python run.py --config-path examples/qwen/ --config-name config action=run
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
python run.py --config-path examples/qwen/ --config-name config action=stop
```
