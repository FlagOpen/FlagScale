import os
import re
import sys
import collections
import socket
import shlex
import subprocess
import json
import uuid
from datetime import datetime
from abc import ABC, abstractmethod
from omegaconf import DictConfig, OmegaConf
from ..logger import logger


def log_and_raise_error(message):
    logger.error(message)
    raise ValueError(message)


def parse_hostfile(hostfile_path):
    if hostfile_path is None or \
        not os.path.isfile(hostfile_path):
        logger.warning("Hostfile not found. The training will proceed using only local resources.")
        return None

    # e.g., worker0 slots=16 type=A100
    pattern = re.compile(r'^(\S+)\s+slots=(\d+)(?:\s+type=(\S+))?')

    resources = collections.OrderedDict()

    with open(hostfile_path, 'r') as fd:
        hostfile_lines = fd.readlines()

    for line in hostfile_lines:
        line = line.strip()
        match = pattern.search(line)
        if line.startswith("#") or line == "":
            # hostfile comment or empty line, ignore
            continue
        elif match:
            host = match.group(1)
            num_slots = int(match.group(2))
            machine_type = match.group(3) if match.group(3) else None
            if host in resources:
                log_and_raise_error(f"Hostfile contains multiple entries for host: {host}.")
            resources[host] = {'slots': num_slots, 'type': machine_type}
        else:
            log_and_raise_error(f"Invalid entry in hostfile: {line}.")

    if len(resources) == 0:
        log_and_raise_error("Hostfile is empty or not formatted correctly. Please check the hostfile.")

    return resources


def _flatten_dict_to_args(config_dict, ignore_keys=[]):
    args = []
    for key, value in config_dict.items():
        if key in ignore_keys:
            continue
        key = key.replace('_', '-')
        if isinstance(value, dict):
            args.extend(_flatten_dict_to_args(value, ignore_keys))
        elif isinstance(value, list):
            args.append(f'--{key}')
            for v in value:
                args.append(f'{v}')
        elif isinstance(value, bool):
            if value:
                args.append(f'--{key}')
        else:
            args.append(f'--{key}')
            args.append(f'{value}')
    return args


def get_megatron_args(config: DictConfig):
    assert config.experiment.backend == "megatron", \
        "This function only supports megatron backend."

    # Convert the DictConfig to a regular dictionary
    config_dict = OmegaConf.to_container(config, resolve=True)
    config_dict = config_dict['train']

    new_config_dict = {} 
    new_config_dict.update(config_dict['system'])
    new_config_dict.update(config_dict['model'])
    new_config_dict.update(config_dict['data'])

    ignore_keys = ["log_dir", "details_dir", "scripts_dir", "pids_dir"]
    # Flatten the dictionary to a list of arguments
    args = _flatten_dict_to_args(new_config_dict, ignore_keys)

    return args


def get_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]


def _update_config(config: DictConfig):
    exp_dir = os.path.abspath(config.experiment.exp_dir)
    assert os.path.isdir(exp_dir), f"Directory {exp_dir} does not exist."

    OmegaConf.set_struct(config, False)

    config = config.train.system

    ckpt_save_dir = (
        config.checkpoint.save
        if getattr(config.checkpoint, 'save', None)
        else os.path.join(exp_dir, "checkpoints")
    )
    ckpt_load_dir = (
        config.checkpoint.load
        if getattr(config.checkpoint, 'load', None)
        else os.path.join(exp_dir, "checkpoints")
    )
    wandb_dir = (
        config.logging.wandb_save_dir
        if getattr(config.logging, 'wandb_save_dir', None)
        else os.path.join(exp_dir, "wandb")
    )
    tensorboard_dir = (
        config.logging.tensorboard_dir
        if getattr(config.logging, 'tensorboard_dir', None)
        else os.path.join(exp_dir, "tensorboard")
    )
    log_dir = (
        config.logging.log_dir
        if getattr(config.logging, 'log_dir', None)
        else os.path.join(exp_dir, "logs")
    )
    scripts_dir = os.path.join(log_dir, "scripts")
    pids_dir = os.path.join(log_dir, "pids")
    details_dir = os.path.join(log_dir, "details")

    config.checkpoint.save = ckpt_save_dir
    config.checkpoint.load = ckpt_load_dir
    config.logging.log_dir = log_dir
    config.logging.scripts_dir = scripts_dir 
    config.logging.pids_dir = pids_dir 
    config.logging.details_dir = details_dir 
    config.logging.tensorboard_dir = tensorboard_dir
    config.logging.wandb_save_dir = wandb_dir

    OmegaConf.set_struct(config, True)


class MultiNodeRunner(ABC):

    @abstractmethod
    def run(self):
        """Run the command"""

    @abstractmethod
    def stop(self):
        """stop the command"""


class SSHRunner(MultiNodeRunner):

    def __init__(self, config: DictConfig):
        self.config = config
        _update_config(self.config)
        self.resources = parse_hostfile(self.config.experiment.get("hostfile", None))

    def _prepare(self):
        self.rdzv_id = datetime.now().strftime("%Y%m%d_%H%M%S.%f")
        self.user_envs = self.config.experiment.envs
        self.user_script = self.config.experiment.entrypoint
        if self.config.experiment.task == "train":
            self.user_args = get_megatron_args(self.config)
        else:
            raise ValueError(f"Unsupported task: {self.config.experiment.task}")

    def _generate_run_script(self, host, node_rank, cmd):
        system_config = self.config.train.system
        logging_config = self.config.train.system.logging

        host_output_file = os.path.join(
            logging_config.log_dir,
            f"host_{node_rank}_{host}.output"
        )
        host_run_script_file = os.path.join(
            logging_config.scripts_dir,
            f"host_{node_rank}_{host}_run.sh"
        )
        host_pid_file = os.path.join(
            logging_config.pids_dir,
            f"host_{node_rank}_{host}.pid"
        )

        # TODO: Implement the creation of the scripts_dir locally. 
        # If there's no shared file system, we'll need to securely copy (scp) 
        # the scripts_dir to the remote host.
        os.makedirs(logging_config.scripts_dir, exist_ok=True)
        root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        megatron_dir = os.path.join(root_dir, "megatron")
        with open(host_run_script_file, 'w') as f:
            f.write("#!/bin/bash\n\n")
            f.write(f"mkdir -p {system_config.checkpoint.load}\n")
            f.write(f"mkdir -p {system_config.checkpoint.save}\n")
            f.write(f"mkdir -p {system_config.logging.log_dir}\n")
            f.write(f"mkdir -p {system_config.logging.pids_dir}\n")
            f.write(f"mkdir -p {system_config.logging.details_dir}\n")
            f.write(f"mkdir -p {system_config.logging.tensorboard_dir}\n")
            f.write(f"mkdir -p {system_config.logging.wandb_save_dir}\n")
            f.write(f"\n")
            f.write(f"cd {root_dir}\n")
            f.write(f"\n")
            f.write(f"export PYTHONPATH={megatron_dir}:{root_dir}\n")
            f.write(f"\n")
            f.write(f"cmd=\"{cmd}\"\n")
            f.write(f"\n")
            # TODO: need a option to control whether to append or overwrite the output file
            # Now, it always appends to the output file
            f.write(f"nohup bash -c \"$cmd\" >> {host_output_file} 2>&1 & echo $! > {host_pid_file}\n")
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.chmod(host_run_script_file, 0o755)

        return host_run_script_file

    def _run_each(
        self, host, master_addr, master_port, nnodes, node_rank, nproc_per_node
    ):
        export_cmd = []
        for k, v in self.user_envs.items():
            export_cmd += [f"{k}={v}"]

        logging_config = self.config.train.system.logging
        torchrun_cmd = ["torchrun",
                        "--rdzv-id", self.rdzv_id, 
                        "--master_addr", str(master_addr),
                        "--master_port", str(master_port),
                        "--nnodes", str(nnodes),
                        "--node_rank", str(node_rank), 
                        "--nproc_per_node", str(nproc_per_node),
                        "--log_dir", str(logging_config.details_dir),
                        "--redirects", str(3),
                        "--tee", str(3),
                        ]

        cmd = shlex.join(export_cmd + torchrun_cmd + [self.user_script] + self.user_args)

        host_run_script_file = self._generate_run_script(host, node_rank, cmd)

        cmd = f"bash {host_run_script_file}"

        if host != "localhost":
            ssh_port = self.config.experiment.ssh_port
            if ssh_port:
                ssh_cmd = f"ssh -f -n -p {ssh_port} {host} '{cmd}'"
            else:
                ssh_cmd = f"ssh -f -n {host} '{cmd}'"
            logger.info(f"SSHRunner is running the command: {ssh_cmd}")
            subprocess.run(ssh_cmd, shell=True, check=True)
        else:
            logger.info(f"SSHRunner is running the command: {cmd}")
            subprocess.run(cmd, shell=True, check=True)

    def run(self):
        self._prepare()
        logger.info("\n************** configuration ***********")
        logger.info(f'\n{OmegaConf.to_yaml(self.config)}')

        visible_devices = self.user_envs.get("CUDA_VISIBLE_DEVICES", None) 
        if visible_devices:
            visible_devices = visible_devices.split(",")
            num_visible_devices = len(visible_devices)
        else:
            num_visible_devices = 1

        master_port = get_free_port()
        # If hostfile is not provided, run the command on localhost
        if self.resources is None:
            self._run_each(
                host="localhost",
                master_addr="localhost",
                master_port=master_port,
                nnodes=1,
                node_rank=0,
                nproc_per_node=num_visible_devices,
            )
            return

        host_list = list(self.resources.keys())
        for host, resource_info in self.resources.items():
            slots = resource_info['slots']
            if visible_devices:
                assert slots == num_visible_devices, \
                    f"Number of slots ({slots}) does not match the number of visible devices ({num_visible_devices})."
            type = resource_info['type']
            master_addr = host_list[0]
            nproc_per_node = slots
            nnodes = len(self.resources)
            node_rank = host_list.index(host)
            self._run_each(
                host, master_addr, master_port, nnodes, node_rank, nproc_per_node
            )

    def _generate_stop_script(self, host, node_rank):
        logging_config = self.config.train.system.logging

        host_stop_script_file = os.path.join(
            logging_config.scripts_dir,
            f"host_{node_rank}_{host}_stop.sh"
        )

        host_pid_file = os.path.join(
            logging_config.pids_dir,
            f"host_{node_rank}_{host}.pid"
        )

        with open(host_stop_script_file, 'w') as f:
            f.write("#!/bin/bash\n\n")
            f.write("if [ -f " + host_pid_file + " ]; then\n")
            f.write("    pid=$(cat " + host_pid_file + ")\n")
            f.write("    pkill -P $pid\n")
            f.write("else\n")
            # TODO: This is a temporary fix. We need to find a better way to stop the job.
            f.write("    pkill -f 'torchrun'\n")
            f.write("fi\n")
            f.flush()
            os.fsync(f.fileno())
        os.chmod(host_stop_script_file, 0o755)

        return host_stop_script_file

    def _stop_each(self, host, node_rank):
        host_stop_script_file = self._generate_stop_script(host, node_rank)

        cmd = f"bash {host_stop_script_file}"

        if host != "localhost":
            ssh_port = self.config.experiment.ssh_port
            if ssh_port:
                ssh_cmd = f"ssh -f -n -p {ssh_port} {host} '{cmd}'"
            else:
                ssh_cmd = f"ssh {host} '{cmd}'"
            logger.info(f"SSHRunner is stopping the job: {ssh_cmd}")
            subprocess.run(ssh_cmd, shell=True, check=True)
        else:
            logger.info(f"SSHRunner is stopping the job: {cmd}")
            subprocess.run(cmd, shell=True, check=True)

    def stop(self):
        if self.resources is None:
            self._stop_each("localhost", 0)
            return

        host_list = list(self.resources.keys())
        for host, _ in self.resources.items():
            node_rank = host_list.index(host)
            self._stop_each(host, node_rank)
