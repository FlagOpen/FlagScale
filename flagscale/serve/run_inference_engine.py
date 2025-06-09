import logging as logger
import os
import subprocess
import sys

from flagscale.serve.args_mapping.mapping import ARGS_CONVERTER

# Compatible with both command-line execution and source code execution.
try:
    import flag_scale
except Exception as e:
    pass

from flagscale import serve
from flagscale.utils import flatten_dict_to_args


def vllm_serve(args):
    common_args = args.get("engine_args", {})
    vllm_args = args.get("engine_args_specific", {}).get("vllm", {})
    vllm_args.update(common_args)
    command = ["vllm", "serve"]
    if vllm_args.get("model", None):
        command.append(vllm_args["model"])
        other_args = flatten_dict_to_args(vllm_args, ["model"])
        command.extend(other_args)
    else:
        raise ValueError("Either model must be specified in vllm_model.")

    # Start the subprocess
    logger.info(f"[Serve]: Starting vllm serve with command: {' '.join(command)}")

    process = subprocess.Popen(command, stdout=sys.stdout, stderr=sys.stderr)
    pid = os.getpid()
    logger.info(f"[Serve]: Current vLLM PID: {pid} ")

    stdout, stderr = process.communicate()
    logger.info(f"[Serve]: Standard Output: {stdout}")
    logger.info(f"[Serve]: Standard Error: {stderr}")

    return process.returncode


def llama_cpp_serve(args):
    common_args = args.get("engine_args", {})
    llama_cpp_args = args.get("engine_args_specific", {}).get("llama_cpp", {})

    command = ["./third_party/llama.cpp/build/bin/llama-server"]
    if common_args.get("model", None):
        converted_args = ARGS_CONVERTER.convert("llama_cpp", common_args)
        command.extend(["--model", converted_args["model"]])
        common_args_flatten = flatten_dict_to_args(converted_args, ["model"])
        command.extend(common_args_flatten)
        llama_cpp_args_flatten = flatten_dict_to_args(llama_cpp_args, ["model"])
        command.extend(llama_cpp_args_flatten)
    else:
        raise ValueError("Either model must be specified in vllm_model.")

    # Start the subprocess
    logger.info(f"[Serve]: Starting llama-cpp serve with command: {' '.join(command)}")

    process = subprocess.Popen(command, stdout=sys.stdout, stderr=sys.stderr)
    pid = os.getpid()
    logger.info(f"[Serve]: Current Llama PID: {pid} ")

    stdout, stderr = process.communicate()
    logger.info(f"[Serve]: Standard Output: {stdout}")
    logger.info(f"[Serve]: Standard Error: {stderr}")

    return process.returncode


def sglang_serve(args):
    common_args = args.get("engine_args", {})
    sglang_args = args.get("engine_args_specific", {}).get("sglang", {})

    command = ["python", "-m", "sglang.launch_server"]
    if common_args.get("model", None):
        converted_args = ARGS_CONVERTER.convert("sglang", common_args)
        command.extend(["--model-path", converted_args["model_path"]])
        common_args_flatten = flatten_dict_to_args(converted_args, ["model"])
        command.extend(common_args_flatten)
        sglang_args_flatten = flatten_dict_to_args(sglang_args, ["model"])
        command.extend(sglang_args_flatten)
    else:
        raise ValueError("Either model must be specified in vllm_model.")

    # Start the subprocess
    logger.info(f"[Serve]: Starting sglang serve with command: {' '.join(command)}")

    process = subprocess.Popen(command, stdout=sys.stdout, stderr=sys.stderr)
    pid = os.getpid()
    logger.info(f"[Serve]: Current Llama PID: {pid} ")

    stdout, stderr = process.communicate()
    logger.info(f"[Serve]: Standard Output: {stdout}")
    logger.info(f"[Serve]: Standard Error: {stderr}")

    return process.returncode


def main():
    serve.load_args()
    serve_config = serve.task_config.get("serve", [])
    if not serve_config:
        raise ValueError(f"No 'serve' configuration found in task config: {serve.task_config}")

    model_config = None
    for item in serve_config:
        if item.get("serve_id", None) == "vllm_model":
            model_config = item
            break

    if model_config is None:
        raise ValueError(f"No 'vllm_model' configuration found in task config: {serve.task_config}")

    engine = model_config.get("engine", None)

    if engine == "vllm":
        return_code = vllm_serve(model_config)
    elif engine == "llama_cpp":
        return_code = llama_cpp_serve(model_config)
    elif engine == "sglang":
        return_code = sglang_serve(model_config)
    else:
        raise ValueError(
            f"Unsupported inference engine: {engine}, current config {serve.task_config}"
        )

    logger.info(f"[Serve]: {engine} serve exited with return code: {return_code}")


if __name__ == "__main__":
    main()
