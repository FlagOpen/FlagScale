import os, sys
import logging
import yaml
import ray
import subprocess
import argparse
#from flagscale.logger import logger
import logging as logger

#ray.init(log_to_driver=True, logging_level=logging.INFO)
ray.init(log_to_driver=True, logging_config=ray.LoggingConfig(encoding="TEXT", log_level="INFO"))

@ray.remote(num_gpus=1)
def start_vllm_serve(args):

    vllm_args = args["serve"]["vllm"]

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
    log_path = f"/tmp/ray/session_latest/logs/worker-{worker_id}-"
    logger.info(f"***************** Current Worker log path: {log_path} *****************")
    process = subprocess.Popen(command, stdout=sys.stdout, stderr=sys.stderr)
    # process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    pid = os.getpid()
    logger.info(f"***************** Current vLLM PID: {pid} *****************")

    stdout, stderr = process.communicate()

    logger.info(f"Standard Output: {stdout}")
    # logger.info(stdout.decode())
    logger.info(f"Standard Error: {stderr}")
    # logger.info(stderr.decode())

    return process.returncode


def main():
    parser = argparse.ArgumentParser(description="Start vllm serve with Ray")

    parser.add_argument(
        "--config-path", type=str, required=True, help="Path to the model"
    )

    args = parser.parse_args()
    with open(args.config_path, "r") as file:
        config = yaml.safe_load(file)
    result = start_vllm_serve.remote(config)

    return_code = ray.get(result)

    logger.info(f"vllm serve exited with return code: {return_code}")


if __name__ == "__main__":
    main()
