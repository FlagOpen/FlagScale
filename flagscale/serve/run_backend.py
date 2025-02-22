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

from flagscale import serve

timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")


# @serve.remote(name="vllm_model")
# def vllm_model(args):

#     vllm_args = args["serve"]["model_args"]["vllm_model"]

#     command = ["vllm", "serve"]
#     command.append(vllm_args["model-tag"])
#     for item in vllm_args:
#         if item not in {"model-tag", "action-args"}:
#             command.append(f"--{item}={vllm_args[item]}")
#     for arg in vllm_args["action-args"]:
#         command.append(f"--{arg}")

#     # Start the subprocess
#     logger.info(f"[Serve]: Starting vllm serve with command: {' '.join(command)}")
#     runtime_context = ray.get_runtime_context()
#     worker_id = runtime_context.get_worker_id()
#     job_id = runtime_context.get_job_id()
#     logger.info(
#         f"[Serve]: Current Job ID: {job_id} , \n[Serve]: ******** Worker ID: {worker_id} ********\n\n"
#     )
#     link_dir = os.path.join(
#         args.log_dir, f"session_latest_{timestamp}", f"worker-{worker_id}-"
#     )
#     logger.info(
#         f"\n\n[Serve]: **********************        {inspect.currentframe().f_code.co_name} Worker log path\
#         ********************** \n[Serve]: {link_dir} \n\n"
#     )

#     process = subprocess.Popen(command, stdout=sys.stdout, stderr=sys.stderr)
#     pid = os.getpid()
#     logger.info(f"[Serve]: Current vLLM PID: {pid} ")

#     stdout, stderr = process.communicate()
#     logger.info(f"[Serve]: Standard Output: {stdout}")
#     logger.info(f"[Serve]: Standard Error: {stderr}")

#     return process.returncode


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
    backend = serve.task_config.experiment.runner.get("backend", None)
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
