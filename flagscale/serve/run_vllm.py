import os
import sys
import datetime
import inspect
import subprocess
import argparse
import json
import logging as logger
from omegaconf import OmegaConf
import ray
from flagscale import serve
from dag_utils import check_and_get_port


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

def start_cluster(task_config):
    hostfile = task_config.serve.hostfile
    head_ip, head_port = next(
        (
            (node.master.ip, node.master.get("port", None))
            for node in hostfile.nodes
            if "master" in node
        ),
        (None, None),
    )
    if head_ip is None:
        raise ValueError(
            f"Failed to start Ray cluster using hostfile {hostfile} due to master node missing. Please ensure that the file exists and has the correct format."
        )
    if head_port is None:
        port = check_and_get_port()
    else:
        port = check_and_get_port(target_port=int(head_port))
    cmd = f"ray stop && ray start --head --port={port}"
    logger.info(f"head node command: {cmd}")
    head_result = subprocess.run(
        cmd,
        shell=True,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if head_result.returncode != 0:
        logger.warning(
            f"Head Node cmd {ssh_cmd} failed with return code {head_result.returncode}."
        )
        logger.warning(f"Output: {head_result.stdout}")
        logger.warning(f"Error: {head_result.stderr}")
        sys.exit(head_result.returncode)
    address = f"{head_ip}:{port}"

    for item in hostfile.nodes:
        if "node" in item:
            node = item.node
            if node.type == "gpu":
                node_cmd = (
                    f"ray stop && ray start --address={address} --num-gpus={node.slots}"
                )

            elif node.type == "cpu":
                node_cmd = (
                    f"ray stop && ray start --address={address} --num-cpus={node.slots}"
                )
            else:
                resource = json.dumps({node.type: node.slots}).replace(
                    '"', '\\"'
                )
                node_cmd = (
                    f"ray stop && ray start --address={address} --resources='{resource}'"
                )
            if task_config.experiment.get("cmds", "") and task_config.experiment.cmds.get(
                "before_start", ""
            ):
                before_start_cmd = task_config.experiment.cmds.before_start
                node_cmd = (
                    f"{before_start_cmd} && "
                    + node_cmd
                )

            if node.get("port", None):
                ssh_cmd = f'ssh -n -p {node.port} {node.ip} "{node_cmd}"'
            else:
                ssh_cmd = f'ssh -n {node.ip} "{node_cmd}"'

            if node.get("docker", None):
                ssh_cmd = f'ssh -n {node.ip} "docker exec {node.docker} /bin/bash -c \'{node_cmd}\'"'

            logger.info(f"worker node command: {cmd}")
            print("=========== ssh_cmd =============== ", ssh_cmd)

            result = subprocess.run(
                ssh_cmd,
                shell=True,
                check=True,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            if result.returncode != 0:
                logger.warning(
                    f"SSH command {ssh_cmd} failed with return code {result.returncode}."
                )
                logger.warning(f"Output: {result.stdout}")
                logger.warning(f"Error: {result.stderr}")
                sys.exit(result.returncode)

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
        start_cluster(serve.task_config)
        return_code = vllm_multiple_nodes_serve(serve.task_config)
    else:
        # Note: Custom log dir here may cause "OSError: AF_UNIX path length cannot exceed 107 bytes:"
        ray.init(
            log_to_driver=True,
            logging_config=ray.LoggingConfig(encoding="TEXT", log_level="INFO"),
        )
        link_dir = os.path.join(serve.task_config.log_dir, f"session_latest_{timestamp}")
        tar_dir = ray._private.worker.global_worker.node._logs_dir
        os.symlink(tar_dir, link_dir)

        result = vllm_model.remote(serve.task_config)
        return_code = ray.get(result)

    logger.info(f"[Serve]: vLLM serve exited with return code: {return_code}")


if __name__ == "__main__":
    main()