import os
import shlex
import asyncio
import psutil
import contextlib
import signal

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from flagscale.runner.runner_base import JobStatus, RunnerBase
from flagscale.runner.runner_utils import (
    get_free_port,
    get_nnodes,
    get_nproc_per_node,
    logger,
    parse_hostfile,
    run_local_command,
    run_scp_command,
    run_ssh_command,
    dummy_random_input,
    benchmark
)


def _get_args_vllm(config: DictConfig):
    # see the following link for more details
    # https://github.com/facebookresearch/hydra/discussions/2750
    config_dict = OmegaConf.to_container(config, resolve=True)

    # step2: restructuring the config
    # config_dict = config_dict["serve"]
    config_dict["serve"]["logging"].pop("log_dir")
    config_dict["serve"]["logging"].pop("scripts_dir")
    config_dict["serve"]["logging"].pop("pids_dir")
    if not config_dict["serve"].get("logging"):
        config_dict["serve"].pop("logging")

    # step3: dict -> yaml
    logging_config = config.serve.logging
    new_config = OmegaConf.create(config_dict)
    new_conf_file = os.path.join(logging_config.scripts_dir, f"serve.yaml")

    # step4: write the new yaml file to `outputs_dir/serve_logs/scripts/serve.yaml`
    with open(new_conf_file, "w") as f:
        OmegaConf.save(config=new_config, f=f.name, resolve=True)

    args = []
    args.append(f"--config-path={new_conf_file}")

    return args


def _update_config_serve(config: DictConfig):
    exp_dir = os.path.abspath(config.experiment.exp_dir)
    if not os.path.isdir(exp_dir):
        os.makedirs(exp_dir)
    assert os.path.isdir(exp_dir), f"Directory {exp_dir} does not exist."

    OmegaConf.set_struct(config, False)

    if config.get("logging", None) is None:
        config.serve.logging = DictConfig({})

    log_dir = os.path.join(exp_dir, f"serve_logs")
    scripts_dir = os.path.join(log_dir, "scripts")
    pids_dir = os.path.join(log_dir, "pids")

    config.serve.logging.log_dir = log_dir
    config.serve.logging.scripts_dir = scripts_dir
    config.serve.logging.pids_dir = pids_dir

    os.makedirs(config.serve.logging.scripts_dir, exist_ok=True)
    OmegaConf.set_struct(config, True)


def _generate_run_script_serve(
    config, host, node_rank, cmd, background=True, with_test=False
):
    nodes = config.serve.get("nodes", None)
    logging_config = config.serve.logging

    no_shared_fs = config.experiment.runner.get("no_shared_fs", False)
    if no_shared_fs:
        host_output_file = os.path.join(logging_config.log_dir, f"host.output")
    else:
        host_output_file = os.path.join(
            logging_config.log_dir, f"host_{node_rank}_{host}.output"
        )
    host_run_script_file = os.path.join(
        logging_config.scripts_dir, f"host_{node_rank}_{host}_run.sh"
    )
    host_pid_file = os.path.join(
        logging_config.pids_dir, f"host_{node_rank}_{host}.pid"
    )

    os.makedirs(logging_config.scripts_dir, exist_ok=True)

    root_dir = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    cmds_config = config.experiment.get("cmds", None)
    ssh_port = config.experiment.runner.get("ssh_port", None)
    docker_name = config.experiment.runner.get("docker", None)
    if cmds_config:
        before_start = cmds_config.get("before_start", "")
    else:
        before_start = ""
    cmd += f" --log-dir={logging_config.log_dir}"
    with open(host_run_script_file, "w") as f:
        f.write("#!/bin/bash\n\n")
        f.write("set -x\n")
        f.write(f"\n")
        f.write(f"{before_start}\n")
        f.write(f"\n")

        if nodes:
            master_ip = nodes[0][0]
            target_port = nodes[0][1].get("port")
            before_start_cmd = None
            if config.experiment.get("cmds", "") and config.experiment.cmds.get(
                "before_start", ""
            ):
                before_start_cmd = config.experiment.cmds.before_start

            f.write(f"# clean nodes \n")
            if len(nodes) > 1:
                for ip, node in nodes[1:]:
                    if not node.get("type", None):
                        raise ValueError(
                            f"Node type must be specified for node {node}. Available types are 'cpu', 'gpu', or a custom resource name."
                        )
                    if not node.get("slots", None):
                        raise ValueError(
                            f"Number of slots must be specified for node {node}. This can be done by setting the 'slots' attribute."
                        )
                    node_cmd = f"ray stop"

                    if before_start_cmd:
                        node_cmd = f"{before_start_cmd} && " + node_cmd

                    if ssh_port:
                        ssh_cmd = f'ssh -n -p {ssh_port} {ip} "{node_cmd}"'
                    else:
                        ssh_cmd = f'ssh -n {ip} "{node_cmd}"'

                    if docker_name:
                        ssh_cmd = f"ssh -n {ip} \"docker exec {docker_name} /bin/bash -c '{node_cmd}'\""
                    f.write(f"{ssh_cmd}\n")
            if before_start_cmd:
                f.write(f"{before_start_cmd} && ray stop\n")
            else:
                f.write(f"ray stop\n")
            f.write(f"\n")

            master_port = target_port if target_port else get_free_port()

            f.write(f"# start cluster\n")
            f.write(f"# master node\n")
            if before_start_cmd:
                f.write(
                    f"{before_start_cmd} && ray start --head --port={master_port}\n"
                )
            else:
                f.write(f"ray start --head --port={master_port}\n")

            if len(nodes) > 1:
                f.write(f"\n")
                f.write(f"# worker nodes\n")
                address = f"{master_ip}:{master_port}"
                for ip, node in nodes[1:]:
                    if not node.get("type", None):
                        raise ValueError(
                            f"Node type must be specified for node {node}. Available types are 'cpu', 'gpu', or a custom resource name."
                        )
                    if not node.get("slots", None):
                        raise ValueError(
                            f"Number of slots must be specified for node {node}. This can be done by setting the 'slots' attribute."
                        )
                    if node.type == "gpu":
                        node_cmd = (
                            f"ray start --address={address} --num-gpus={node.slots}"
                        )

                    elif node.type == "cpu":
                        node_cmd = (
                            f"ray start --address={address} --num-cpus={node.slots}"
                        )
                    else:
                        resource = json.dumps({node.type: node.slots}).replace(
                            '"', '\\"'
                        )
                        node_cmd = (
                            f"ray start --address={address} --resources='{resource}'"
                        )
                    if config.experiment.get("cmds", "") and config.experiment.cmds.get(
                        "before_start", ""
                    ):
                        before_start_cmd = config.experiment.cmds.before_start
                        node_cmd = f"{before_start_cmd} && " + node_cmd

                    if ssh_port:
                        ssh_cmd = f'ssh -n -p {ssh_port} {ip} "{node_cmd}"'
                    else:
                        ssh_cmd = f'ssh -n {ip} "{node_cmd}"'

                    if docker_name:
                        ssh_cmd = f"ssh -n {ip} \"docker exec {docker_name} /bin/bash -c '{node_cmd}'\""
                    f.write(f"{ssh_cmd}\n")

        f.write(f"mkdir -p {logging_config.log_dir}\n")
        f.write(f"mkdir -p {logging_config.pids_dir}\n")
        f.write(f"\n")
        f.write(f"cd {root_dir}\n")
        f.write(f"\n")
        f.write(f"export PYTHONPATH={root_dir}\n")
        f.write(f"\n")
        f.write(f'cmd="{cmd}"\n')
        f.write(f"\n")
        # TODO: need a option to control whether to append or overwrite the output file
        # Now, it always appends to the output file
        if background:
            f.write(
                f'nohup bash -c "$cmd; sync" >> {host_output_file} 2>&1 & echo $! > {host_pid_file}\n'
            )
        else:
            f.write(f'bash -c "$cmd; sync" >> {host_output_file} 2>&1\n')
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.chmod(host_run_script_file, 0o755)

    return host_run_script_file


def _generate_stop_script(config, host, node_rank):
    logging_config = config.serve.logging

    host_stop_script_file = os.path.join(
        logging_config.scripts_dir, f"host_{node_rank}_{host}_stop.sh"
    )

    host_pid_file = os.path.join(
        logging_config.pids_dir, f"host_{node_rank}_{host}.pid"
    )

    os.makedirs(logging_config.scripts_dir, exist_ok=True)

    cmds_config = config.experiment.get("cmds", None)
    if cmds_config:
        after_stop = cmds_config.get("after_stop", "")
    else:
        after_stop = ""
    with open(host_stop_script_file, "w") as f:
        f.write("#!/bin/bash\n\n")
        f.write("ray stop\n")
        f.write("pkill -f 'vllm'\n")
        f.write(f"{after_stop}\n")
        f.flush()
        os.fsync(f.fileno())
    os.chmod(host_stop_script_file, 0o755)

    return host_stop_script_file

def kill_process_tree(pid):
    try:
        parent = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return

    # Get all children recursively
    children = parent.children(recursive=True)

    # Send SIGKILL to all children first
    for child in children:
        with contextlib.suppress(ProcessLookupError):
            os.kill(child.pid, signal.SIGKILL)

    # Finally kill the parent
    with contextlib.suppress(ProcessLookupError):
        os.kill(pid, signal.SIGKILL)


class SSHServeRunner(RunnerBase):
    def __init__(self, config: DictConfig):
        super().__init__(config)
        self.task_type = getattr(self.config.experiment.task, "type", None)
        assert self.task_type == "serve", f"Unsupported task type: {self.task_type}"
        self.command_line_mode = getattr(
            self.config.serve.deploy, "command-line-mode", None
        )
        self._prepare()
        self.host = None
        self.port = self.config.serve.model_args.vllm_model.port

    def _prepare(self):
        _update_config_serve(self.config)
        self.user_args = _get_args_vllm(self.config)
        self.user_envs = self.config.experiment.get("envs", {})
        entrypoint = self.config.experiment.task.get("entrypoint", None)
        if self.command_line_mode:
            self.user_script = "flagscale/serve/run_vllm.py"
        elif isinstance(entrypoint, str) and entrypoint.endswith(".py"):
            self.user_script = entrypoint
        elif entrypoint is None:
            self.user_script = "flagscale/serve/run_serve.py"
        else:
            raise ValueError(
                f"Invalid config entrypoint: {entrypoint}, must be a python file path or null."
            )
        self.resources = parse_hostfile(
            self.config.experiment.runner.get("hostfile", None)
        )
        if self.resources:
            self.config.serve["nodes"] = list(self.resources.items())
        logger.info("\n************** configuration **************")
        logger.info(f"\n{OmegaConf.to_yaml(self.config)}")

    def _run_each(
        self,
        host,
        master_addr,
        master_port,
        nnodes,
        node_rank,
        nproc_per_node,
        with_test=False,
        dryrun=False,
    ):
        export_cmd = []
        for k, v in self.user_envs.items():
            export_cmd += [f"{k}={v}"]

        cmd = shlex.join(export_cmd + ["python"] + [self.user_script] + self.user_args)

        host_run_script_file = _generate_run_script_serve(
            self.config, host, node_rank, cmd, background=True, with_test=with_test
        )

        run_local_command(f"bash {host_run_script_file}", dryrun)

    def run(self, with_test=False, dryrun=False):
        num_visible_devices = None
        visible_devices = self.user_envs.get("CUDA_VISIBLE_DEVICES", None)
        if visible_devices is not None and isinstance(visible_devices, str):
            visible_devices = visible_devices.split(",")
            num_visible_devices = len(visible_devices)

        runner_config = self.config.experiment.runner

        # If hostfile is not provided, run the job on localhost
        nproc_from_args = runner_config.get("nproc_per_node", None)
        nproc_per_node = get_nproc_per_node(None, nproc_from_args, num_visible_devices)
        available_addr = runner_config.get("master_addr", "localhost")
        available_port = runner_config.get("master_port", get_free_port())
        self._run_each(
            "localhost",
            available_addr,
            available_port,
            1,
            0,
            nproc_per_node,
            with_test=with_test,
            dryrun=dryrun,
        )
        self.host = available_addr

    def _stop_each(self, host, node_rank):
        logging_config = self.config.serve.logging
        host_pid_file = os.path.join(
                logging_config.pids_dir, f"host_{node_rank}_{host}.pid"
            )
        with open(host_pid_file, "r") as f:
            pid=f.readlines()[0]
            pid = int(pid.strip())
        kill_process_tree(pid)


    def stop(self):
        if self.resources is None:
            self._stop_each("localhost", 0)
            return

        nnodes = get_nnodes(
            len(self.resources), self.config.experiment.runner.get("nnodes", None)
        )

        for node_rank, (host, _) in enumerate(self.resources.items()):
            if node_rank >= nnodes:
                break
            self._stop_each(host, node_rank)

    def _generate_query_script(self, host, node_rank):
        """Genetrate the query script for each host."""
        logging_config = self.config.serve.logging

        host_query_script_file = os.path.join(
            logging_config.scripts_dir, f"host_{node_rank}_{host}_query.sh"
        )

        host_pid_file = os.path.join(
            logging_config.pids_dir, f"host_{node_rank}_{host}.pid"
        )
        os.makedirs(logging_config.scripts_dir, exist_ok=True)

        with open(host_query_script_file, "w") as f:
            f.write("#!/bin/bash\n\n")
            f.write("if [ -f " + host_pid_file + " ]; then\n")
            f.write("    pid=$(cat " + host_pid_file + ")\n")
            f.write("    ps -p $pid -o state --no-headers\n")
            f.write("else\n")
            # TODO: This is a temporary fix. We need to find a better way to query the job.
            f.write(
                "    pid=$(ps aux | grep 'run_vllm' | grep -v grep | head -n 1 | awk '{print $2}')\n"
            )
            f.write("    ps -p $pid -o state --no-headers\n")
            f.write("fi\n")
            f.flush()
            os.fsync(f.fileno())
        os.chmod(host_query_script_file, 0o755)

        return host_query_script_file

    def _query_each(self, host, node_rank):
        "Query each node status."
        host_query_script_file = self._generate_query_script(host, node_rank)
        logging_config = self.config.serve.logging
        result = ""
        if host != "localhost":
            ssh_port = self.config.experiment.runner.get("ssh_port", 22)
            # Step 1: make sure the scripts_dir exists on the remote host
            run_ssh_command(
                host, f"mkdir -p {logging_config.scripts_dir}", ssh_port, query=True
            )
            # Step 2: copy the host_run_script_file to the remote host
            no_shared_fs = self.config.experiment.runner.get("no_shared_fs", False)
            if no_shared_fs:
                run_scp_command(
                    host, host_query_script_file, logging_config.scripts_dir, ssh_port
                )
            # Step 3: run the host_run_script_file on the remote host
            try:
                result = run_ssh_command(
                    host, f"bash {host_query_script_file}", ssh_port, query=True
                )
            except Exception as e:
                logger.error(f"Failed to query job status on {host}: {e}")
        else:
            try:
                result = run_local_command(f"bash {host_query_script_file}", query=True)
            except Exception as e:
                logger.error(f"Failed to query job status on {host}: {e}")
        result = result.stdout.rstrip() if result else ""
        return result

    def _query_status(self):
        "Query Job status."
        results = []
        if self.resources is None:
            result = self._query_each("localhost", 0)
            results.append(result)

        else:
            host_list = list(self.resources.keys())
            for host, _ in self.resources.items():
                node_rank = host_list.index(host)
                result = self._query_each(host, node_rank)
                results.append(result)
        if all((status != "" and status != "Z") for status in results):
            job_status = JobStatus.RUNNING
        elif all((status == "" or status == "Z") for status in results):
            job_status = JobStatus.COMPLETED_OR_IDLE
        else:
            job_status = JobStatus.TRANSITIONAL
        return job_status

    def _serve_alive(self):
        config = self.config
        model_name = config.serve.model_args.vllm_model["model-tag"]
        from openai import OpenAI

        # Modify OpenAI's API key and API base to use vLLM's API server.
        api_key = "EMPTY"
        api_url = f"http://{self.host}:{self.port}/v1"

        try:
            client = OpenAI(
                api_key=api_key,
                base_url=api_url,
            )
            messages = [{"role": "user", "content": "who are you?"}]
            response = client.chat.completions.create(model=model_name, messages=messages)
        except Exception as e:
            # logger.info(f"API {api_url} is not ready, please wait a moment")
            return False

        return True

    def _profile_serve(self):
        from vllm.transformers_utils.tokenizer import get_tokenizer
        model = self.config.serve.model_args.vllm_model["model-tag"]
        tokenizer_mode = "auto"
        trust_remote_code = "trust-remote-code" in self.config.serve.model_args.vllm_model["action-args"]
        tokenizer = get_tokenizer(model,
                            tokenizer_mode=tokenizer_mode,
                            trust_remote_code=trust_remote_code)
        dummy_input_requests = dummy_random_input(tokenizer=tokenizer, num_prompts=10)
        api_url = f"http://{self.host}:{self.port}/v1/completions"
        ### allow metric = [\"ttft\", \"tpot\", \"itl\", \"e2el\"]
        ### allow percentiles = [\"25,50,75\"]
        result = asyncio.run(
            benchmark(
            api_url,
            model=model,
            tokenizer=tokenizer,
            input_requests=dummy_input_requests,
            selected_percentile_metrics="ttft,tpot,itl,e2el".split(","),
            selected_percentiles=[float(99)]))
        return result
