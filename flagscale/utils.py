import importlib
import os
import sys

_vLLM_replace_modules = {
    "vllm.sampling_params": "flagscale/inference/core/sampling_params.py",
    "vllm.sequence": "flagscale/inference/core/sequence.py",
    "vllm.core.block_manager": "flagscale/inference/core/block_manager.py",
    "vllm.core.scheduler": "flagscale/inference/core/scheduler.py",
    "vllm.engine.llm_engine": "flagscale/inference/core/llm_engine.py",
    "vllm.engine.output_processor.single_step": "flagscale/inference/core/single_step.py",
    "vllm.inputs.data": "flagscale/inference/core/data.py",
    "vllm.inputs.preprocess": "flagscale/inference/core/preprocess.py",
    "vllm.model_executor.sampling_metadata": "flagscale/inference/core/sampling_metadata.py",
    "vllm.model_executor.layers.logits_processor": "flagscale/inference/core/logits_processor.py",
    "vllm.worker.model_runner": "flagscale/inference/core/model_runner.py",
}
_hook_modules = {"transformer_engine"} | set(_vLLM_replace_modules.keys())


class Empty:
    def __init__(self, *args):
        pass


class CustomModuleFinder:

    def find_module(self, fullname, path=None):
        if fullname in _hook_modules:
            return CustomModuleLoader()


class CustomModuleLoader:
    device_type = os.environ.get("DEVICE_TYPE", None)

    def load_module(self, fullname):

        if fullname in _vLLM_replace_modules:
            replace_module_path = _vLLM_replace_modules[fullname]
            spec = importlib.util.spec_from_file_location(fullname, replace_module_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            sys.modules[fullname] = module

        if fullname in sys.modules:
            return sys.modules[fullname]

        finder = sys.meta_path.pop(0)
        module = importlib.import_module(fullname)

        if self.device_type == "iluvatar":
            if fullname == "transformer_engine":
                module.common.recipe = Empty()
                module.common.recipe.DelayedScaling = Empty()

        sys.meta_path.insert(0, finder)
        return module


def flatten_dict_to_args(config_dict, ignore_keys=[]):
    args = []
    for key, value in config_dict.items():
        if key in ignore_keys:
            continue
        key = key.replace("_", "-")
        if isinstance(value, dict):
            args.extend(flatten_dict_to_args(value, ignore_keys))
        elif isinstance(value, list):
            args.append(f"--{key}")
            for v in value:
                args.append(f"{v}")
        elif isinstance(value, bool):
            if value:
                args.append(f"--{key}")
        else:
            args.append(f"--{key}")
            args.append(f"{value}")
    return args
