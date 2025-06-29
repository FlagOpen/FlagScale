## vLLM Mock Model 
Mock model for vllm testing without running the actual model, to verify (v) the pre- and postprocessing:

![image](./docs/mock_model.png 'mock_model.png')

The replay of captured outputs can be performed and is faster than running the actual model. Moreover, for replay it may suffice to use less NPUs than needed for the actual model.


## Automated Install
Checkout branch omni_infer_mock_model_pr (or omni_infer_v1 when it is merged). Run one of two patch scripts for either using NPUs (normal mode), or not using NPUs at all (no-NPU mode, does not support capturing or normal model execution). Usually, you want the normal mode:
```bash
cd omni_infer/infer_engines
bash_install_mock.sh
```
or alternatively if tests without NPUs are required:
```bash
cd omni_infer/infer_engines
bash_install_mock_without_npus.sh
```
Now you can run the scripts in the omni_infer/tests/test_framework/vllm_mock/scripts folder, see also further below.

NOTE: If get ValueError: infer_schema, just comment out registration of apply_w8a8_block_fp8_linear in vllm/model_executor/layers/quantization/utils/fp8_utils.py

## Manual Install
First, install for example vllm v0.8.5 and vllm_ascend v0.8.5, e.g. from https://codehub-g.huawei.com/DataScience/omni_infer/wiki?categoryId=221789&sn=WIKI202505086766436 or specifically
```bash
git clone --recurse-submodules -b omni_infer_v1 ssh://git@codehub-dg-g.huawei.com:2222/DataScience/omni_infer.git && \
cd omni_infer/infer_engines/vllm && \
git checkout master && \
git pull && \
git checkout 1ff751ca17a58ad3877f25eca440d74a97d6cc64 && \
sed -i 's/^xgrammar[[:space:]]*==[[:space:]]*0.1.19/xgrammar == 0.1.18/' requirements/common.txt
VLLM_TARGET_DEVICE=empty pip install -e . && \
cd ../vllm_ascend && \
git checkout master && \
git pull && \
git checkout dc1e55bbd0200929a9ccc3e2f5d5db31131a49fd && \
pip install -e .
```

In case of multi-node, you may have to apply the patch vllm_commit_id_1ff751ca17a58ad3877f25eca440d74a97d6cc64.patch.

Other versions may also work. In particular, vllm v0.9.0 also works (tested: https://codehub-g.huawei.com/DataScience/omni_infer/wiki?categoryId=221789&sn=WIKI202505206889113 )

Perform the following manual patch into vllm_ascend:

- Copy mock.py from models/mock.py to vllm_ascend/models/mock.py
- Register / Overwrite your desired models as mock models in vllm_ascend/models/init.py , e.g., add the following end of register_models() in init.py:
```python
    from vllm_ascend.models.mock import mock_model_class_factory

    ModelRegistry.register_model(
        "DeepseekV2ForCausalLM",
        mock_model_class_factory(CustomDeepseekV2ForCausalLM))
        
    from vllm.model_executor.models.qwen2 import Qwen2ForCausalLM
    ModelRegistry.register_model(
        "Qwen2ForCausalLM",
        mock_model_class_factory(Qwen2ForCausalLM))
    
    ModelRegistry.register_model(
        "DeepseekV3ForCausalLM",
        mock_model_class_factory(CustomDeepseekV3ForCausalLM))
```

## Configs
The mock model has a random output mode, capture mode and replay mode, as well as PD separation (KV_CACHE_MODE) and no-NPU support, with the configs set by environment variables (there are some more variables, check in mock.py for your needs):

```python
# COMMENT OUT the ones not in use. IMPORTANT: Make sure you use temperature=0!
# os.environ["KV_CACHE_MODE"] = "KV_CACHE"  # for PD separation
# os.environ["CAPTURE_MODE"] = "CAPTURE"  # capture inputs and outputs to cache
# os.environ["REPLAY_MODE"] = "REPLAY"  # replay inputs and outputs from cache
os.environ["RANDOM_MODE"] = "RANDOM"
# os.environ["FORWARD_TIME"] = "0"
# os.environ["SIMULATE_ELAPSED_TIME"] = "SIMULATE"  # replay model output after waiting for approx. captured time.
os.environ["MOCK_CAPTURE_DIR"] = "/home/kc/capture/"  # saving folder for logs of inputs and outputs
os.environ["MOCK_CAPTURE_FILE"] = ".mock"
os.environ["MOCK_CAPTURE_FILE_LOCK"] = ".lock"
```

- When CAPTURE_MODE (set), the model outputs for each prompt (identified by their prompt token ids) are captured to MOCK_CAPTURE_FILE in MOCK_CAPTURE_DIR.
- When instead REPLAY_MODE (set), the MOCK_CAPTURE_FILE in MOCK_CAPTURE_DIR is read into a cache, and whenever a request with matching prompt token ids comes in, the corresponding position outputs are replayed.
- KV_CACHE_MODE needed for PD separation, captures or replays the KV cache on the prefill nodes.
- RANDOM_MODE is very fast and outputs random outputs regardless of input, with FORWARD_TIME simulated time for computing output.
- SIMULATE_ELAPSED_TIME on REPLAY_MODE will simulate time for computing output as it took on the NPUs.
- PREFILL_PROCESS must be set for the P node of PD separation
- MOCK_COMPUTE_LOGITS allows also mocking logits if needed (may be slow because logits are high-dimensional) 
- TORCH_COMPILE_MODE_MOCK can be set for using torch graph compile mode, in case automatic detection fails

## Run (Offline Mode)
NOTE: Regardless of mock model, seems DP>1 fails in (21st May) v0.9.0 state. If it works for you, go ahead. Otherwise, please use online modes below.

Set capture_mode and not replay_mode, register the mock model instead of the original model in vllm_ascend/models/init.py and then run (with temperature 0 and use tp=2, otherwise accuracy drops due to an unrelated issue) to capture / replay the outputs or produce random ones. See the following scripts:
```bash
python ./scripts/random_mock_model_tp2.py
python ./scripts/capture_mock_model_tp2.py
python ./scripts/replay_mock_model_tp2.py
```

## Run (Single-Node Online API-Server)
To run in serving mode on a single node, register the mock model and then serve the mock model in either capture mode or replay mode, similar to offline mode. See:
```bash
python ./scripts/vllm_serve.py
```
and send some prompts to the server
```bash
bash ./scripts/prompt.sh
```

## Run (Multi-Node Online API-Server)
You may be required to apply the patch, see installation above.

To run multi-node, simultaneously run ./scripts/vllm_serve_multinode_master.sh and ./scripts/vllm_serve_multinode_slave.sh on nodes 0 and 1 respectively. Make sure to change the IPs in the script to the master node IP, and have access to the same files / copies of the same files with the same path. The server will start up on the master node. 

Note: Make sure to change configs on both machines / in both master & slave scripts. Make sure also to change all of the desired environment variables before calling ray start in the script!

## Run (API-Servers  PD Separation)
To run PD separation on one node (1P1D), run ./scripts/offline_inference_pd.py :

- (If needed: Checkout branch yyx_dev_pd_seperate in vllm and vllm_ascend or see wiki for newest versions)
- Generate a rank table using the following:
```bash
export LLM_TOOLS_PATH=/home/ma-user/AscendCloud/AscendCloud-LLM/llm_tools
python ${LLM_TOOLS_PATH}/PD_separate/pd_ranktable_tools.py --mode gen --prefill-server-list 0,1 --decode-server-list 2,3 --api-server --save-dir ./save_dir
```
- Rename P/D to "server-0" and "server-1" under the last two "server_id" entries
- Add the rank table path and NPU ids you set above to ./scripts/offline_inference_pd.py
- Run the script ./scripts/offline_inference_pd.py 

In particular, note that you need to set the PREFILL_PROCESS environment variable for the prefill node, and KV_CACHE_MODE for both P/D nodes. Apart from that, it suffices to set the environment variables as usual, see the script for reference. 

Note: For PD separation, capture / replay on decode node works, but will not verify the KV cache input on the decode node due to numerical inaccuracy issues from layer normalization, since the KV cache will look different on every run.

Please see the following config as an example:
```python
os.environ["RANDOM_MODE"] = "1"  # replay inputs and outputs from the cache
# os.environ["SIMULATE_ELAPSED_TIME"] = "1"  # replay model output after waiting for approx. captured time.
os.environ["KV_CACHE_MODE"] = "1"  # capture inputs and outputs, for use with the replaying mode.
# os.environ["CAPTURE_MODE"] = "1"  # capture inputs and outputs, for use with the replaying mode.
# os.environ["REPLAY_MODE"] = "1"  # capture inputs and outputs, for use with the replaying mode.
os.environ["MOCK_CAPTURE_DIR"] = "/home/kc/capture/"  # saving folder for logs of inputs and outputs, ensure this exists
os.environ["MOCK_CAPTURE_FILE"] = ".mock_cache_pd"
os.environ["MOCK_CAPTURE_FILE_LOCK"] = ".lock"
```

## Run (Single-Node Online API-Server without NPU)
To run without NPUs, in case of manual install, apply the required patches to vllm_ascend.
```bash
cd ../vllm_ascend
git apply ../vllm_mock/vllm_ascend_no_npu.patch
```
For installation, if you do not have anything installed and no NPU, do the following:
1. Install CANN toolkit and CANN nnal, e.g., https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.1.RC1/Ascend-cann-toolkit_8.1.RC1_linux-x86_64.run and https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.1.RC1/Ascend-cann-nnal_8.1.RC1_linux-x86_64.run
2. Set the environments for CANN, and maybe also install CANN driver, unsure if needed, e.g. (wrong architecture but works), https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/Ascend%20HDK/Ascend%20HDK%2024.1.RC2/Ascend-hdk-910b-npu-driver_24.1.rc2_linux-aarch64.run
3. Install vllm, e.g.
git clone --recurse-submodules -b omni_infer_v1 ssh://git@codehub-dg-g.huawei.com:2222/DataScience/omni_infer.git
cd omni_infer/infer_engines/
sh bash_install_mock_no_npus.sh
cd vllm
SETUPTOOLS_SCM_PRETEND_VERSION=0.9.0 VLLM_TARGET_DEVICE=empty pip install -e .
4. pip3 install torch==2.5.1+cpu --index-url https://download.pytorch.org/whl/cpu
5. pip3 install torch-npu==2.5.1
6. pip install ray decorator cloudpickle tornado absl-py ml-dtypes attrs psutil scipy jinja2 latex2sympy2 word2number timeout_decorator "numpy<2" setuptools_scm cmake msgpack pybind11 quart
7. CMAKE_MODULE_PATH=/root/venv/lib/python3.10/site-packages/torch/share/cmake/Torch/ CMAKE_PREFIX_PATH=/root/venv/lib/python3.10/site-packages/torch/share/cmake/Torch/ pip install -e . --no-build-isolation

Then start the API server (set ASCEND_RT_VISIBLE_DEVICES as if you had enough NPUs, and also set env variable NO_NPU_MOCK):
```bash
python ./scripts/vllm_serve_no_npu.py
```
and send some prompts to the server
```bash
bash ./scripts/prompt.sh
```


## Saving memory manually
To save memory when running with NPUs, you need to stop loading the full model. The easiest way to do this is to simply use the no-NPU patch, see omni_infer/infer_engines/bash_install_mock_no_npus.sh. Alternatively (not without NPUs, to avoid the patch), change the number of layers to, e.g., 1: In the config.json of the model directory, write "num_hidden_layers": 1

Then, you can manually adjust load_weights in the vllm model_executor or vllm model file to ignore non-existing params, e.g., before param = params_dict[name]:
```python
if name not in params_dict:
    continue
```

## Limitations
- Sometimes the accuracy of model seems to drop, in particular for TP=1. It appears an unrelated issue, you can easily test by removing the mock model in init.py.
- For PD separation, capture / replay on decode node works, but will not verify the KV cache input on the decode node due to numerical inaccuracy issues from layer normalization, since the KV cache will look different on every run.