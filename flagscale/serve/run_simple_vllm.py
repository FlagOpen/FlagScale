import os
import sys
import datetime
import subprocess
import argparse
import logging as logger
from omegaconf import OmegaConf
import ray


timestamp =  datetime.datetime.now().strftime("%Y%m%d%H%M%S")

@ray.remote(num_gpus=1)
def start_vllm_serve(args, log_dir):

    vllm_args = args["serve"]["llm"]

    command = ["vllm", "serve"]
    command.append(vllm_args["model-tag"])
    for item in vllm_args:
        if item not in {"model-tag", "action-args"}:
            command.append(f"--{item}={vllm_args[item]}")
    for arg in vllm_args["action-args"]:
        command.append(f"--{arg}")

    # Start the subprocess

    logger.info(f"Starting vllm serve with command: {' '.join(command)}")
    runtime_context = ray.get_runtime_context()
    worker_id = runtime_context.get_worker_id()
    job_id = runtime_context.get_job_id()
    logger.info(f"***************** Current Worker ID: {worker_id} *****************")
    logger.info(f"***************** Current Job ID: {job_id} *****************")
    link_dir = os.path.join(log_dir, f"session_latest_{timestamp}", "logs", f"worker-{worker_id}-")
    logger.info(f"***************** Current Worker log path: {link_dir} *****************")
    process = subprocess.Popen(command, stdout=sys.stdout, stderr=sys.stderr)
    pid = os.getpid()
    logger.info(f"***************** Current vLLM PID: {pid} *****************")

    stdout, stderr = process.communicate()

    logger.info(f"Standard Output: {stdout}")
    logger.info(f"Standard Error: {stderr}")

    return process.returncode


def main():
    parser = argparse.ArgumentParser(description="Start vllm serve with Ray")

    parser.add_argument(
        "--config-path", type=str, required=True, help="Path to the model"
    )
    parser.add_argument(
        "--log-dir", type=str, required=True, help="Path to the model"
    )
    args = parser.parse_args()

    config = OmegaConf.load(args.config_path)
    logger.info(f"--------------------- config {config} -------------------------")
    # Note: Custom the log dir here may cause "OSError: AF_UNIX path length cannot exceed 107 bytes:"
    ray.init(log_to_driver=True, logging_config=ray.LoggingConfig(encoding="TEXT", log_level="INFO"))
    link_dir = os.path.join(args.log_dir, f"session_latest_{timestamp}")
    # TODO: Default path in ray will be replaced by api here.
    os.symlink("/tmp/ray/session_latest", link_dir)
    result = start_vllm_serve.remote(config, args.log_dir)

    return_code = ray.get(result)

    logger.info(f"vllm serve exited with return code: {return_code}")


if __name__ == "__main__":
    main()
