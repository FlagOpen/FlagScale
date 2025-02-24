[<img src="flagopen.png">](https://flagopen.baai.ac.cn/)

## Latest News
- **[2024/11]** Released [v0.6.0](https://github.com/FlagOpen/FlagScale/tree/release/v0.6.0):
  - Introduced general multi-dimensional heterogeneous parallelism and CPU-based communication between different chips.
  - Added the full support for LLaVA-OneVision, achieving SOTA results on the [Infinity-MM](https://arxiv.org/abs/2410.18558) dataset.
  - Open-sourced the optimized CFG implementation and accelerated the generation and understanding tasks for [Emu3](https://arxiv.org/abs/2409.18869).
  - Implemented the auto-tuning feature and enhanced the CI/CD system.
- **[2024/4]** Released [v0.3](https://github.com/FlagOpen/FlagScale/tree/release/v0.3): Achieved heterogeneous hybrid training of the Aquila2-70B-Expr model on a cluster using both NVIDIA and Iluvatar chips. Adapted the Aquila2 series to AI chips from six different manufacturers.
- **[2023/11]** Released [v0.2](https://github.com/FlagOpen/FlagScale/tree/v0.2): Introduced training support for Aquila2-70B-Expr, enabling heterogeneous training across chips with the same or compatible architectures.
- **[2023/10]** Released [v0.1](https://github.com/FlagOpen/FlagScale/tree/v0.1): Supported Aquila models with optimized training schemes for Aquila2-7B and Aquila2-34B, including parallel strategies, optimizations, and hyper-parameter settings.

## About

[FlagScale](https://github.com/FlagOpen/FlagScale.git) is a comprehensive toolkit designed to support the entire lifecycle of large models, developed with the backing of the Beijing Academy of Artificial Intelligence (BAAI). It builds on the strengths of several prominent open-source projects, including [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) and [vllm](https://github.com/vllm-project/vllm), to provide a robust, end-to-end solution for managing and scaling large models.

The primary objective of FlagScale is to enable seamless scalability across diverse hardware architectures while maximizing computational resource efficiency and enhancing model performance. By offering essential components for model development, training, and deployment, FlagScale seeks to establish itself as an indispensable toolkit for optimizing both the speed and effectiveness of large model workflows.

FlagScale is also a part of [FlagAI-Open](https://flagopen.baai.ac.cn/), an open-source initiative by BAAI that aims to foster an open-source ecosystem for AI technologies. It serves as a platform where developers, researchers, and AI enthusiasts can collaborate on various AI projects, contribute to the development of cutting-edge AI solutions, and share their work with the global community.

## Quick Start

FlagScale leverages [Hydra](https://github.com/facebookresearch/hydra) for configuration management. The configurations are organized into two levels: an outer experiment-level YAML file and an inner task-level YAML file.

- The experiment-level YAML file defines the experiment directory, backend engine, task type, and other related environmental configurations.
- The task-level YAML file specifies the model, dataset, and parameters for specific tasks such as training or inference.

All valid configurations in the task-level YAML file correspond to the arguments used in backend engines such as Megatron-LM and vllm, with hyphens (-) replaced by underscores (_). For a complete list of available configurations, please refer to the backend engine documentation. Simply copy and modify the existing YAML files in the [examples](./examples) folder to get started.

### Setup
We recommend using the latest release of [NGC's PyTorch container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch) for setup.

1. Clone the repository:
    ```sh
    git clone https://github.com/FlagOpen/FlagScale.git
    ```

2. Install the requirements:
    ```sh
    cd FlagScale/install
    ./install-requirements.sh --env train
    ./install-requirements.sh --env inference
    ```
    The above instructions create two conda environments: `flagscale-train` and `flagscale-inference`, which contain the dependency environments for training and inference, respectively.

3. Install the packages with customized extensions:
    ```sh
    cd vllm
    pip install .

    pip install -e ./megatron-energon
    cp -r megatron-energon/src/megatron/energon megatron/megatron
    ```

### DeepSeek-R1 Serve

We support the model serving of DeepSeek R1 and have implemented the flagscale serve command for one-click deployment.
Only configure two YAML files, then use the `flagscale serve` command to serve.

1. Configure the yaml files:
```
Flagscale/
├── examples/
│   └── deepseek_r1/
│       └── config_deepseek_r1.yaml # set hostfile
│       └── serve/
│           └── deepseek_r1.yaml # set model parameters and server port
```
2. Install FlagScale CLI:
    ```
    cd FlagScale
    pip install .
    ```

3. One-click serve:
    ```
    flagscale serve deepseek_r1
    ```

4. When custom service parameters, users can run:
    ```
    flagscale serve <model_name> <MODEL_CONFIG_YAML>
    ```

### Run a Task

FlagScale provides a unified runner for various tasks, including training，inference and serve. Simply specify the configuration file to run the task with a single command. The runner will automatically load the configurations and execute the task. The following example demonstrates how to run a distributed training task.

#### Train

1. Start the distributed training job:
    ```sh
    python run.py --config-path ./examples/aquila/conf --config-name config action=run
    ```
    The `data_path` in the demo is the path of the training datasets following the [Megatron-LM format](./megatron/README.md#data-preprocessing). For quickly running the pretraining process, we also provide a small processed data ([bin](https://model.ks3-cn-beijing.ksyuncs.com/nlpdata/pile_wikipedia_demo.bin) and [idx](https://model.ks3-cn-beijing.ksyuncs.com/nlpdata/pile_wikipedia_demo.idx)) from the [Pile](https://pile.eleuther.ai/) dataset.

2. Stop the distributed training job:
    ```sh
    python run.py --config-path ./examples/aquila/conf --config-name config action=stop
    ```

#### Serve

1. Start the server:
    ```sh
    python run.py --config-path ./examples/qwen/conf --config-name config_qwen2.5_7b action=run
    ```
2. Stop the server:
    ```sh
    python run.py --config-path ./examples/qwen/conf --config-name config_qwen2.5_7b action=stop
    ```
For more details, please refer to [Quick Start](./flagscale/serve/README.md).

## License

This project is licensed under the [Apache License (Version 2.0)](https://github.com/FlagOpen/FlagScale/blob/main/LICENSE). This project also contains other third-party components under other open-source licenses. See the [LICENSE](https://github.com/FlagOpen/FlagScale/blob/main/LICENSE) file for more information.
