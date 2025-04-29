import asyncio
import contextlib
import copy
import json
import os
import shlex
import shutil
import signal

import psutil

from omegaconf import DictConfig, OmegaConf

from flagscale.runner.runner_base import JobStatus, RunnerBase
from flagscale.runner.utils import (
    ResourceManager,
    benchmark,
    dummy_random_input,
    flatten_dict_to_args,
    get_free_port,
    get_nproc_per_node,
    logger,
    parse_hostfile,
    run_local_command,
)


def _get_multiple_free_ports(num=1, exclude_ports=[]):
    allocated_ports = []
    for i in range(num):
        port = get_free_port()
        while port in allocated_ports or port in exclude_ports:
            port = get_free_port()
        allocated_ports.append(port)
    return allocated_ports


def _get_args_vllm(config: DictConfig):
    # see the following link for more details
    # https://github.com/facebookresearch/hydra/discussions/2750
    config_dict = OmegaConf.to_container(config, resolve=True)

    # step2: restructuring the config
    # config_dict = config_dict["serve"]
    config_dict["logging"].pop("log_dir")
    config_dict["logging"].pop("scripts_dir")
    config_dict["logging"].pop("pids_dir")
    if not config_dict.get("logging"):
        config_dict.pop("logging")

    # step3: dict -> yaml
    logging_config = config.logging
    new_config = OmegaConf.create(config_dict)
    new_conf_file = os.path.join(logging_config.scripts_dir, f"serve.yaml")

    # step4: write the new yaml file to `outputs_dir/serve_logs/scripts/serve.yaml`
    with open(new_conf_file, "w") as f:
        OmegaConf.save(config=new_config, f=f.name, resolve=True)

    args = []
    args.append(f"--config-path={new_conf_file}")

    return args


def _reset_serve_port(config):
    model_port = None
    deploy_port = config.experiment.get("deploy", {}).get("port", None)
    OmegaConf.set_struct(config, False)

    for item in config.serve:
        if item.get("serve_id", None) == "vllm_model":
            if deploy_port:
                model_port = deploy_port
                item.engine_args["port"] = deploy_port
            else:
                model_port = item.engine_args.get("port", 8000)
            break
    OmegaConf.set_struct(config, True)
    if not model_port:
        logger.warning(f"No 'model_port' configuration found in task config: {config}")
    return model_port


def _get_inference_engine(config):
    serve_config = config.get("serve", [])
    if not serve_config:
        raise ValueError(f"No 'serve' configuration found in task config: {serve_config}")
    if serve_config and len(serve_config) > 1:
        logger.warning(f"Multiple 'serve' configurations found in task config: {serve_config}")

    engine = serve_config[0].get("engine", None)
    return engine


def _get_engine_args(config, model="vllm_model"):
    serve_config = config.get("serve", [])
    if not serve_config:
        raise ValueError(f"No 'serve' configuration found in task config: {serve_config}")
    engine_args = {}

    for item in serve_config:
        if item.get("serve_id", None) == model:
            engine_args = item.get("engine_args", {})
            break
    if not engine_args:
        raise ValueError(f"No 'engine_args' configuration found in task config: {serve_config}")

    return engine_args


def _update_config_serve(config: DictConfig):
    deploy_config = config.experiment.get("deploy", {})

    exp_dir = os.path.abspath(config.experiment.exp_dir)
    if not os.path.isdir(exp_dir):
        os.makedirs(exp_dir)
    assert os.path.isdir(exp_dir), f"Directory {exp_dir} does not exist."

    OmegaConf.set_struct(config, False)

    if deploy_config.get("prefill_decode_disaggregation", False):
        deploy_config["pd_proxy_port"] = get_free_port()

    if config.get("logging", None) is None:
        config.logging = DictConfig({})

    log_dir = os.path.join(exp_dir, f"serve_logs")
    scripts_dir = os.path.join(log_dir, "scripts")
    pids_dir = os.path.join(log_dir, "pids")

    config.logging.log_dir = log_dir
    config.logging.scripts_dir = scripts_dir
    config.logging.pids_dir = pids_dir

    os.makedirs(config.logging.scripts_dir, exist_ok=True)
    OmegaConf.set_struct(config, True)


def _generate_run_script_serve(config, host, node_rank, cmd, background=True, with_test=False):
    nodes = config.get("nodes", None)
    logging_config = config.logging

    no_shared_fs = config.experiment.runner.get("no_shared_fs", False)
    if no_shared_fs:
        host_output_file = os.path.join(logging_config.log_dir, f"host.output")
    else:
        host_output_file = os.path.join(logging_config.log_dir, f"host_{node_rank}_{host}.output")
    host_run_script_file = os.path.join(
        logging_config.scripts_dir, f"host_{node_rank}_{host}_run.sh"
    )
    host_pid_file = os.path.join(logging_config.pids_dir, f"host_{node_rank}_{host}.pid")

    os.makedirs(logging_config.scripts_dir, exist_ok=True)

    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    cmds_config = config.experiment.get("cmds", None)
    ssh_port = config.experiment.runner.get("ssh_port", 22)
    docker_name = config.experiment.runner.get("docker", None)
    if cmds_config:
        before_start_cmd = cmds_config.get("before_start", "")
    else:
        before_start_cmd = ""
    cmd += f" --log-dir={logging_config.log_dir}"
    try:
        import vllm

        vllm_path = os.path.dirname(vllm.__path__[0])
    except Exception as e:
        vllm_path = f"{root_dir}/vllm"
    deploy_config = config.experiment.get("deploy", {})
    envs = config.experiment.get("envs", {})
    with open(host_run_script_file, "w") as f:
        f.write("#!/bin/bash\n\n")
        f.write("set -x\n")
        f.write(f"\n")
        f.write(f"{before_start_cmd}\n")
        f.write(f"\n")

        f.write(f'if [ -z "$PYTHONPATH" ]; then\n')
        f.write(f"    export PYTHONPATH={vllm_path}:{root_dir}\n")
        f.write(f"else\n")
        f.write(f'    export PYTHONPATH="$PYTHONPATH:{vllm_path}:{root_dir}"\n')
        f.write(f"fi\n")
        f.write(f"\n")
        envs_str = " && ".join(f"export {key}={value}" for key, value in envs.items())

        if nodes:
            if deploy_config.get("prefill_decode_disaggregation", False):
                resource_manager = ResourceManager(nodes)
                master_ip = nodes[0][0]
                target_port = nodes[0][1].get("port")
                p_num = deploy_config.get("prefill_num", 1)
                d_num = deploy_config.get("decode_num", 1)
                ports_num = (p_num + d_num) * 2
                kv_related_ports = _get_multiple_free_ports(ports_num)
                pd_proxy_port = deploy_config.get("pd_proxy_port", None)
                if not pd_proxy_port:
                    raise ValueError(f"PD disaggregation requires a proxy port to be set.")

                engine_args = _get_engine_args(config)
                command_items = ["vllm", "serve"]
                command_items.append(engine_args["model"])
                other_args = flatten_dict_to_args(engine_args, ["model", "port"])
                command_items.extend(other_args)
                vllm_command = " ".join(command_items)
                if before_start_cmd:
                    vllm_command = f"{before_start_cmd} && " + vllm_command
                if envs_str:
                    vllm_command = f"{envs_str} && " + vllm_command
                p_address = deploy_config.get("prefill_address", "auto")
                d_address = deploy_config.get("decode_address", "auto")
                tensor_parallel_size = engine_args.get("tensor_parallel_size", 1)
                pipeline_parallel_size = engine_args.get("pipeline_parallel_size", 1)
                each_instance_card_num = tensor_parallel_size * pipeline_parallel_size
                default_log_dir = deploy_config.get(
                    "prefill_decode_log_dir", logging_config.log_dir
                )

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
                        node_cmd = f"mkdir -p {default_log_dir} && pkill -f vllm"

                        ssh_cmd = f'ssh -n -p {ssh_port} {ip} "{node_cmd}"'

                        if docker_name:
                            ssh_cmd = f"ssh -n -p {ssh_port} {ip} \"docker exec {docker_name} /bin/bash -c '{node_cmd}'\""
                        f.write(f"{ssh_cmd}\n")

                f.write("pkill -f 'run_inference_engine'\n")
                f.write("pkill -f 'run_fs_serve_vllm'\n")
                f.write("pkill -f 'vllm serve'\n")
                f.write("pkill -f 'run_disagg_xpyd_router'\n")
                f.write(f"mkdir -p {default_log_dir}\n")
                f.write(f"\n")

                f.write("echo '=========== launch prefill instance ==========='\n")

                for i in range(p_num):
                    kv_port = kv_related_ports.pop()
                    http_port = kv_related_ports.pop()
                    p_kv_config = {
                        "kv_connector": "P2pConnector",
                        "kv_role": "kv_producer",
                        "kv_port": str(kv_port),
                        "kv_connector_extra_config": {
                            "proxy_ip": master_ip,
                            "proxy_port": str(pd_proxy_port),
                            "http_port": str(http_port),
                        },
                    }
                    logger.info(
                        f"============= prefill instance {i}, p_kv_config: {p_kv_config} ============="
                    )
                    card_ids = resource_manager.get_available_card_ids(
                        address=p_address, num=each_instance_card_num
                    )
                    card_ids_str = ",".join(map(str, card_ids))
                    ids_env = f"export CUDA_VISIBLE_DEVICES={card_ids_str}"

                    p_kv_config_json = json.dumps(p_kv_config)
                    p_instance_log_path = os.path.join(default_log_dir, f"prefill_{i}.log")

                    if p_address != master_ip:
                        p_kv_config_formate_json = p_kv_config_json.replace('"', '\\"')
                        node_cmd = f"{ids_env} && {vllm_command} --port {http_port} --kv-transfer-config '\\''{p_kv_config_formate_json}'\\''"
                        if docker_name:
                            ssh_cmd = f"ssh -f -n -p {ssh_port} {ip} \"docker exec {docker_name} /bin/bash -c '{node_cmd} > {p_instance_log_path} 2>&1 &'\""
                        else:
                            ssh_cmd = f'ssh -f -n -p {ssh_port} {d_address} "{node_cmd} > {p_instance_log_path} 2>&1 &"'
                        f.write(f"{ssh_cmd}\n\n")
                    else:
                        p_cmd = f"{ids_env} && {vllm_command} --port {http_port} --kv-transfer-config '\\''{p_kv_config_json}'\\''"
                        f.write(f"p_{i}_cmd='{p_cmd}'\n")
                        f.write(f"\n")
                        f.write(
                            f'nohup bash -c "$p_{i}_cmd; sync" >> {p_instance_log_path} 2>&1 &\n\n'
                        )

                f.write("echo '=========== launch decode instance ==========='\n")

                for j in range(d_num):
                    kv_port = kv_related_ports.pop()
                    http_port = kv_related_ports.pop()
                    d_kv_config = {
                        "kv_connector": "P2pConnector",
                        "kv_role": "kv_consumer",
                        "kv_port": str(kv_port),
                        "kv_connector_extra_config": {
                            "proxy_ip": master_ip,
                            "proxy_port": str(pd_proxy_port),
                            "http_port": str(http_port),
                        },
                    }
                    logger.info(
                        f"============= decode instance {i}, d_kv_config: {d_kv_config} ============="
                    )
                    card_ids = resource_manager.get_available_card_ids(
                        address=d_address, num=each_instance_card_num
                    )
                    card_ids_str = ",".join(map(str, card_ids))
                    ids_env = f"export CUDA_VISIBLE_DEVICES={card_ids_str}"

                    d_kv_config_json = json.dumps(d_kv_config)
                    d_instance_log_path = os.path.join(default_log_dir, f"decode_{j}.log")

                    if d_address != master_ip:
                        d_kv_config_formate_json = d_kv_config_json.replace('"', '\\"')
                        node_cmd = f"{ids_env} && {vllm_command} --port {http_port} --kv-transfer-config '\\''{d_kv_config_formate_json}'\\''"
                        if docker_name:
                            ssh_cmd = f"ssh -f -n -p {ssh_port} {ip} \"docker exec {docker_name} /bin/bash -c '{node_cmd} > {d_instance_log_path} 2>&1 &'\""
                        else:
                            ssh_cmd = f'ssh -f -n -p {ssh_port} {d_address} "{node_cmd} > {d_instance_log_path} 2>&1 &"'
                        f.write(f"{ssh_cmd}\n\n")
                    else:
                        d_cmd = f"{ids_env} && {vllm_command} --port {http_port} --kv-transfer-config '\\''{d_kv_config_json}'\\''"
                        f.write(f"d_{j}_cmd='{d_cmd}'\n")
                        f.write(f"\n")
                        f.write(
                            f'nohup bash -c "$d_{i}_cmd; sync" >> {d_instance_log_path} 2>&1 &\n\n'
                        )

            else:
                f.write(f"ray_path=$(realpath $(which ray))\n")
                master_ip = nodes[0][0]
                target_port = nodes[0][1].get("port")

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
                        node_cmd = f"${{ray_path}} stop"

                        if before_start_cmd:
                            node_cmd = f"{before_start_cmd} && " + node_cmd

                        ssh_cmd = f'ssh -n -p {ssh_port} {ip} "{node_cmd}"'

                        if docker_name:
                            ssh_cmd = f"ssh -n -p {ssh_port} {ip} \"docker exec {docker_name} /bin/bash -c '{node_cmd}'\""
                        f.write(f"{ssh_cmd}\n")
                if before_start_cmd:
                    f.write(f"{before_start_cmd} && ${{ray_path}} stop\n")
                else:
                    f.write(f"${{ray_path}} stop\n")
                f.write("pkill -f 'run_inference_engine'\n")
                f.write("pkill -f 'run_fs_serve_vllm'\n")
                f.write("pkill -f 'vllm serve'\n")
                f.write(f"\n")

                master_port = target_port if target_port else get_free_port()

                address = f"{master_ip}:{master_port}"
                for index, (ip, node) in enumerate(nodes):
                    if not node.get("type", None):
                        raise ValueError(
                            f"Node type must be specified for node {node}. Available types are 'cpu', 'gpu', or a custom resource name."
                        )
                    if not node.get("slots", None):
                        raise ValueError(
                            f"Number of slots must be specified for node {node}. This can be done by setting the 'slots' attribute."
                        )
                    if index == 0:
                        # master node
                        f.write(f"# start cluster\n")
                        f.write(f"# master node\n")
                        if node.type == "gpu":
                            node_cmd = f"${{ray_path}} start --head --port={master_port} --num-gpus={node.slots}"
                        elif node.type == "cpu":
                            node_cmd = f"${{ray_path}} start --head --port={master_port} --num-cpus={node.slots}"
                        else:
                            resource = json.dumps({node.type: node.slots}).replace('"', '\\"')
                            node_cmd = f"${{ray_path}} start --head --port={master_port} --resources='{resource}'"
                        if before_start_cmd:
                            node_cmd = f"{before_start_cmd} && " + node_cmd
                        f.write(f"{node_cmd}\n")

                    else:
                        resource = json.dumps({node.type: node.slots}).replace('"', '\\"')
                        node_cmd = f"${{ray_path}} start --head --port={master_port} --resources='{resource}'"
                    if before_start_cmd:
                        node_cmd = f"{before_start_cmd} && " + node_cmd
                    f.write(f"{node_cmd}\n")

                else:
                    # worker nodes
                    if index == 1:
                        f.write(f"\n")
                        f.write(f"# worker nodes\n")
                    if node.type == "gpu":
                        node_cmd = (
                            f"${{ray_path}} start --address={address} --num-gpus={node.slots}"
                        )

                    elif node.type == "cpu":
                        node_cmd = (
                            f"${{ray_path}} start --address={address} --num-cpus={node.slots}"
                        )
                    else:
                        resource = json.dumps({node.type: node.slots}).replace('"', '\\"')
                        node_cmd = (
                            f"${{ray_path}} start --address={address} --resources='{resource}'"
                        )
                    if before_start_cmd:
                        node_cmd = f"{before_start_cmd} && " + node_cmd
                    ssh_cmd = f'ssh -n -p {ssh_port} {ip} "{node_cmd}"'
                    if docker_name:
                        ssh_cmd = f"ssh -n -p {ssh_port} {ip} \"docker exec {docker_name} /bin/bash -c '{node_cmd}'\""
                    f.write(f"{ssh_cmd}\n")
        else:
            # Note: config key device_type is specified for single node serving in neither gpu or cpu.
            device_type = None
            nproc_per_node = None
            if config.experiment.get("runner", None) and config.experiment.runner.get(
                "device_type", None
            ):
                device_type = config.experiment.runner.get("device_type", None)
                nproc_per_node = config.experiment.runner.get("nproc_per_node", None)
                if nproc_per_node is None:
                    raise ValueError(
                        f"nproc_per_node must be specified when device_type {device_type} is specified."
                    )
            node_cmd = None

            if deploy_config.get("use_fs_serve", True) and config.serve[0].get("engine", None):
                f.write(f"ray_path=$(realpath $(which ray))\n")
                if not device_type:
                    node_cmd = f"${{ray_path}} start --head"
                elif device_type == "gpu":
                    node_cmd = f"${{ray_path}} start --head --num-gpus={nproc_per_node}"
                elif device_type == "cpu":
                    node_cmd = f"${{ray_path}} start --head --num-cpus={nproc_per_node}"
                else:
                    resource = json.dumps({device_type: nproc_per_node}).replace('"', '\\"')
                    node_cmd = f"${{ray_path}} start --head --resources='{resource}'"
            if before_start_cmd:
                node_cmd = f"{before_start_cmd} && {node_cmd}" if node_cmd else before_start_cmd
            if node_cmd:
                f.write(f"{node_cmd}\n")

        f.write(f"mkdir -p {logging_config.log_dir}\n")
        f.write(f"mkdir -p {logging_config.pids_dir}\n")
        f.write(f"\n")
        f.write(f"cd {root_dir}\n")
        f.write(f"\n")
        f.write(f'cmd="{cmd}"\n')
        f.write(f"\n")
        # TODO: need a option to control whether to append or overwrite the output file
        # Now, it always appends to the output file
        f.write("echo '=========== launch task ==========='\n")
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
    logging_config = config.logging

    host_stop_script_file = os.path.join(
        logging_config.scripts_dir, f"host_{node_rank}_{host}_stop.sh"
    )

    host_pid_file = os.path.join(logging_config.pids_dir, f"host_{node_rank}_{host}.pid")

    os.makedirs(logging_config.scripts_dir, exist_ok=True)

    cmds_config = config.experiment.get("cmds", None)
    if cmds_config:
        after_stop = cmds_config.get("after_stop", "")
    else:
        after_stop = ""
    with open(host_stop_script_file, "w") as f:
        f.write("#!/bin/bash\n\n")
        f.write(f"ray_path=$(realpath $(which ray))\n")
        f.write(f"${{ray_path}} stop\n")
        f.write("pkill -f 'run_inference_engine'\n")
        f.write("pkill -f 'run_fs_serve_vllm'\n")
        f.write("pkill -f 'vllm serve'\n")
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
        self.deploy_config = self.config.experiment.get("deploy", {})
        self.inference_engine = _get_inference_engine(self.config)
        self.use_fs_serve = self.deploy_config.get("use_fs_serve", True)
        self.port = _reset_serve_port(config)
        self._prepare()
        self.host = None

    def _prepare(self):
        _update_config_serve(self.config)
        self.user_args = _get_args_vllm(self.config)
        self.user_envs = self.config.experiment.get("envs", {})
        entrypoint = self.config.experiment.task.get("entrypoint", None)
        if self.inference_engine:
            if self.config.experiment.get("deploy", {}).get("prefill_decode_disaggregation", False):
                self.user_script = "flagscale/serve/run_disagg_xpyd_router.py"
            elif not self.use_fs_serve:
                self.user_script = "flagscale/serve/run_inference_engine.py"
            else:
                self.user_script = "flagscale/serve/run_fs_serve_vllm.py"
        elif isinstance(entrypoint, str) and entrypoint.endswith(".py"):
            self.user_script = entrypoint
        elif entrypoint is None:
            self.user_script = "flagscale/serve/run_serve.py"
        else:
            raise ValueError(
                f"Invalid config entrypoint: {entrypoint}, must be a python file path or null."
            )
        hostfile_path = self.config.experiment.runner.get("hostfile", None)
        if hostfile_path:
            if os.path.isabs(hostfile_path):
                hostfile_path = hostfile_path
            else:
                hostfile_path = os.path.join(os.getcwd(), hostfile_path)
            if not os.path.exists(hostfile_path):
                raise ValueError(f"The hostfile {hostfile_path} does not exist")

        self.resources = None

        if hostfile_path:
            self.resources = parse_hostfile(hostfile_path)
            for key, value in self.resources.items():
                if not value.get("type", None):
                    logger.warning(
                        f"The hostfile key type is not set for host {key}, using gpu by default"
                    )
                    self.resources[key]["type"] = "gpu"
            if self.resources:
                OmegaConf.set_struct(self.config, False)
                self.config["nodes"] = list(self.resources.items())
                OmegaConf.set_struct(self.config, True)
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
        logging_config = self.config.logging
        host_pid_file = os.path.join(logging_config.pids_dir, f"host_{node_rank}_{host}.pid")
        with open(host_pid_file, "r") as f:
            pid = f.readlines()[0]
            pid = int(pid.strip())
        kill_process_tree(pid)

        ray_executable = shutil.which("ray")
        if ray_executable:
            ray_path = os.path.realpath(ray_executable)
            os.system(f"{ray_path} stop")
        else:
            logger.info("Failed to find ray path")

        os.system("pkill -f run_inference_engine")
        os.system("pkill -f run_fs_serve_vllm")
        os.system("pkill -f 'vllm serve'")

    def stop(self):
        self._stop_each("localhost", 0)
        return

    def _generate_query_script(self, host, node_rank):
        """Genetrate the query script for each host."""
        logging_config = self.config.logging

        host_query_script_file = os.path.join(
            logging_config.scripts_dir, f"host_{node_rank}_{host}_query.sh"
        )

        host_pid_file = os.path.join(logging_config.pids_dir, f"host_{node_rank}_{host}.pid")
        os.makedirs(logging_config.scripts_dir, exist_ok=True)

        with open(host_query_script_file, "w") as f:
            f.write("#!/bin/bash\n\n")
            f.write("if [ -f " + host_pid_file + " ]; then\n")
            f.write("    pid=$(cat " + host_pid_file + ")\n")
            f.write("    ps -p $pid -o state --no-headers\n")
            f.write("else\n")
            # TODO: This is a temporary fix. We need to find a better way to query the job.
            f.write(
                "    pid=$(ps aux | grep -E 'run_fs_serve_vllm|run_inference_engine' | grep -v grep | head -n 1 | awk '{print $2}')\n"
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
        logging_config = self.config.logging
        result = ""
        try:
            result = run_local_command(f"bash {host_query_script_file}", query=True)
        except Exception as e:
            logger.error(f"Failed to query job status on {host}: {e}")
        result = result.stdout.rstrip() if result else ""
        return result

    def _query_status(self):
        "Query Job status."
        results = []
        result = self._query_each("localhost", 0)
        results.append(result)
        if all((status != "" and status != "Z") for status in results):
            job_status = JobStatus.RUNNING
        elif all((status == "" or status == "Z") for status in results):
            job_status = JobStatus.COMPLETED_OR_IDLE
        else:
            job_status = JobStatus.TRANSITIONAL
        return job_status

    def _serve_alive(self):
        engine_args = _get_engine_args(self.config)
        model_name = engine_args.get("served_model_name", None) or engine_args.get("model", None)

        if not model_name:
            raise ValueError("No model specified in config file.")

        from openai import OpenAI

        # Modify OpenAI's API key and API base to use vLLM's API server.
        api_key = "EMPTY"
        api_url = f"http://{self.host}:{self.port}/v1"
        logger.info(f"Testing API {api_url}")

        try:
            client = OpenAI(api_key=api_key, base_url=api_url)
            messages = [{"role": "user", "content": "who are you?"}]
            response = client.chat.completions.create(model=model_name, messages=messages)
        except Exception as e:
            # logger.info(f"API {api_url} is not ready, please wait a moment")
            return False

        return True

    def _profile_serve(self):
        from vllm.transformers_utils.tokenizer import get_tokenizer

        tokenizer_mode = "auto"
        engine_args = _get_engine_args(self.config)

        trust_remote_code = engine_args.get("trust_remote_code", False)

        model_name = engine_args.get("served_model_name", None) or engine_args.get("model", None)

        if not model_name:
            raise ValueError("No model specified in config file.")

        tokenizer = get_tokenizer(
            model_name, tokenizer_mode=tokenizer_mode, trust_remote_code=trust_remote_code
        )

        dummy_input_requests = dummy_random_input(tokenizer=tokenizer, num_prompts=200)
        api_url = f"http://{self.host}:{self.port}/v1/chat/completions"
        logger.info(f"Profiling API {api_url}")

        ### allow metric = [\"ttft\", \"tpot\", \"itl\", \"e2el\"]
        ### allow percentiles = [\"25,50,75\"]
        result = asyncio.run(
            benchmark(
                api_url,
                model=model_name,
                tokenizer=tokenizer,
                input_requests=dummy_input_requests,
                selected_percentile_metrics="ttft,tpot,itl,e2el".split(","),
                selected_percentiles=[float(99)],
            )
        )
        return result
