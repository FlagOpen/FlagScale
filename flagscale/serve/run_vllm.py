import argparse
import datetime
import inspect
import json
import logging as logger
import os
import subprocess
import sys

import ray
from dag_utils import check_and_get_port
from omegaconf import OmegaConf

# Compatible with both command-line execution and source code execution.
try:
    import flag_scale
except Exception as e:
    pass

from flagscale import serve

timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")


@serve.remote(name="vllm_model")
def vllm_model(args):

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
    runtime_context = ray.get_runtime_context()
    worker_id = runtime_context.get_worker_id()
    job_id = runtime_context.get_job_id()
    logger.info(
        f"[Serve]: Current Job ID: {job_id} , \n[Serve]: ******** Worker ID: {worker_id} ********\n\n"
    )
    link_dir = os.path.join(
        args.log_dir, f"session_latest_{timestamp}", f"worker-{worker_id}-"
    )
    logger.info(
        f"\n\n[Serve]: **********************        {inspect.currentframe().f_code.co_name} Worker log path\
        ********************** \n[Serve]: {link_dir} \n\n"
    )

    process = subprocess.Popen(command, stdout=sys.stdout, stderr=sys.stderr)
    pid = os.getpid()
    logger.info(f"[Serve]: Current vLLM PID: {pid} ")

    stdout, stderr = process.communicate()
    logger.info(f"[Serve]: Standard Output: {stdout}")
    logger.info(f"[Serve]: Standard Error: {stderr}")

    return process.returncode


def vllm_multiple_nodes_serve(args):
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
    runtime_context = ray.get_runtime_context()
    worker_id = runtime_context.get_worker_id()
    job_id = runtime_context.get_job_id()
    logger.info(
        f"[Serve]: Current Job ID: {job_id} , \n[Serve]: ******** Worker ID: {worker_id} ********\n\n"
    )
    link_dir = os.path.join(
        args.log_dir, f"session_latest_{timestamp}", f"worker-{worker_id}-"
    )
    logger.info(
        f"\n\n[Serve]: **********************        {inspect.currentframe().f_code.co_name} Worker log path\
        ********************** \n[Serve]: {link_dir} \n\n"
    )

    process = subprocess.Popen(command, stdout=sys.stdout, stderr=sys.stderr)
    pid = os.getpid()
    logger.info(f"[Serve]: Current vLLM PID: {pid} ")

    stdout, stderr = process.communicate()
    logger.info(f"[Serve]: Standard Output: {stdout}")
    logger.info(f"[Serve]: Standard Error: {stderr}")

    return process.returncode


def main():
    hostfile = serve.task_config.experiment.runner.get("hostfile", None)
    if hostfile:
        return_code = vllm_multiple_nodes_serve(serve.task_config)
    else:
        # Note: Custom log dir here may cause "OSError: AF_UNIX path length cannot exceed 107 bytes:"
        ray.init(
            log_to_driver=True,
            logging_config=ray.LoggingConfig(encoding="TEXT", log_level="INFO"),
        )
        link_dir = os.path.join(
            serve.task_config.log_dir, f"session_latest_{timestamp}"
        )
        tar_dir = ray._private.worker.global_worker.node._logs_dir
        os.symlink(tar_dir, link_dir)

        result = vllm_model.remote(serve.task_config)
        return_code = ray.get(result)

    logger.info(f"[Serve]: vLLM serve exited with return code: {return_code}")


if __name__ == "__main__":
    main()
