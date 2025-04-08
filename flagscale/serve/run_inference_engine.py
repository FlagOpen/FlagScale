import logging as logger
import os
import subprocess
import sys

# Compatible with both command-line execution and source code execution.
try:
    import flag_scale
except Exception as e:
    pass

from flagscale import serve
from flagscale.utils import flatten_dict_to_args


def vllm_serve(args):
    vllm_args = args.get("engine_args", {})
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


def main():
    serve.load_args()
    serve_config = serve.task_config.get("serve", [])
    if not serve_config:
        raise ValueError(
            f"No 'serve' configuration found in task config: {serve.task_config}"
        )

    model_config = None
    for item in serve_config:
        if item.get("serve_id", None) == "vllm_model":
            model_config = item
            break

    if model_config is None:
        raise ValueError(
            f"No 'vllm_model' configuration found in task config: {serve.task_config}"
        )

    engine = model_config.get("engine", None)

    if engine == "vllm":
        return_code = vllm_serve(model_config)
    else:
        raise ValueError(
            f"Unsupported inference engine: {engine}, current config {serve.task_config}"
        )

    logger.info(f"[Serve]: {engine} serve exited with return code: {return_code}")


if __name__ == "__main__":
    main()
