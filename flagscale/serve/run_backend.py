import logging as logger
import os
import subprocess
import sys

from flagscale import serve


def vllm_serve(args):
    vllm_args = args["serve"]["model_args"]["vllm_model"]

    command = ["vllm", "serve"]
    command.append(vllm_args["model-tag"])
    for item in vllm_args:
        if item not in {"model-tag", "action-args"}:
            command.append(f"--{item}={vllm_args[item]}")
    for arg in vllm_args["action-args"]:
        command.append(f"--{arg}")

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
    backend = serve.task_config.experiment.task.get("backend", None)
    print(
        "================== task_config ====================",
        serve.task_config,
        flush=True,
    )
    if backend == "vllm":
        return_code = vllm_serve(serve.task_config)
    else:
        raise ValueError(f"Unsupported backend: {backend}")

    logger.info(f"[Serve]: {backend} serve exited with return code: {return_code}")


if __name__ == "__main__":
    main()
