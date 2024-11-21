import os
import shlex
import time
from datetime import datetime

from omegaconf import DictConfig, OmegaConf

from flagscale.runner.runner_base import JobStatus, RunnerBase
from flagscale.runner.runner_utils import (
    add_decive_extra_config,
    flatten_dict_to_args,
    get_free_port,
    get_host_name_or_ip,
    get_nnodes,
    get_nproc_per_node,
    logger,
    parse_hostfile,
    run_local_command,
    run_scp_command,
    run_ssh_command,
)


def _get_args_megatron(config: DictConfig):
    assert (
        config.experiment.task.backend == "megatron"
    ), "This function only supports megatron backend."

    # Convert the DictConfig to a regular dictionary
    config_dict = OmegaConf.to_container(config, resolve=True)
    config_dict = config_dict["train"]

    new_config_dict = {}
    new_config_dict.update(config_dict["system"])
    new_config_dict.update(config_dict["model"])
    new_config_dict.update(config_dict["data"])

    ignore_keys = ["log_dir", "details_dir", "scripts_dir", "pids_dir"]
    # Flatten the dictionary to a list of arguments
    args = flatten_dict_to_args(new_config_dict, ignore_keys)

    return args


def _update_config_train(config: DictConfig):
    exp_dir = os.path.abspath(config.experiment.exp_dir)
    if not os.path.isdir(exp_dir):
        os.makedirs(exp_dir)
    assert os.path.isdir(exp_dir), f"Directory {exp_dir} does not exist."

    OmegaConf.set_struct(config, False)

    if config.experiment.runner.get("no_shared_fs", False):
        config.train.system.no_shared_fs = True

    config = config.train.system

    if config.get("checkpoint", None) is None:
        config.checkpoint = DictConfig({})

    if config.get("logging", None) is None:
        config.logging = DictConfig({})

    ckpt_save_dir = (
        os.path.abspath(config.checkpoint.save)
        if config.checkpoint.get("save", None)
        else os.path.join(exp_dir, "checkpoints")
    )
    ckpt_load_dir = (
        os.path.abspath(config.checkpoint.load)
        if config.checkpoint.get("load", None)
        else os.path.join(exp_dir, "checkpoints")
    )
    wandb_dir = (
        os.path.abspath(config.logging.wandb_save_dir)
        if config.logging.get("wandb_save_dir", None)
        else os.path.join(exp_dir, "wandb")
    )
    tensorboard_dir = (
        os.path.abspath(config.logging.tensorboard_dir)
        if config.logging.get("tensorboard_dir", None)
        else os.path.join(exp_dir, "tensorboard")
    )
    log_dir = (
        os.path.abspath(config.logging.log_dir)
        if config.logging.get("log_dir", None)
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

    OmegaConf.set_struct(config, False)


def _get_runner_cmd_train(
    host,
    master_addr,
    master_port,
    nnodes,
    node_rank,
    nproc_per_node,
    config: DictConfig,
):
    runner_config = config.experiment.runner
    logging_config = config.train.system.logging

    if runner_config.get("per_node_task", False):
        nnodes = 1
        node_rank = 0
        master_addr = "localhost"

    rdzv_id = runner_config.get("rdzv_id", "default")
    log_dir = runner_config.get("log_dir", logging_config.details_dir)
    log_dir = os.path.abspath(log_dir)
    no_shared_fs = config.experiment.get("no_shared_fs", False)
    if no_shared_fs:
        log_dir = os.path.join(log_dir, f"host")
    else:
        log_dir = os.path.join(log_dir, f"host_{node_rank}_{host}")
    log_dir = os.path.join(log_dir, datetime.now().strftime("%Y%m%d_%H%M%S.%f"))
    rdzv_backend = runner_config.get("rdzv_backend", "c10d")
    rdzv_endpoint = runner_config.get("rdzv_endpoint", f"{master_addr}:{master_port}")
    redirect = runner_config.get("redirects", "3")
    tee = runner_config.get("tee", "3")
    backend = runner_config.get("backend", "torchrun")

    runner_args = OmegaConf.to_container(runner_config, resolve=True)
    if "type" in runner_args:
        del runner_args["type"]
    if "backend" in runner_args:
        del runner_args["backend"]
    if "per_node_task" in runner_args:
        del runner_args["per_node_task"]
    if "hostfile" in runner_args:
        del runner_args["hostfile"]
    if "master_addr" in runner_args:
        del runner_args["master_addr"]
    if "master_port" in runner_args:
        del runner_args["master_port"]
    if "ssh_port" in runner_args:
        del runner_args["ssh_port"]
    runner_args["rdzv_id"] = rdzv_id
    # runner_args["master_addr"] = master_addr
    # runner_args["master_port"] = master_port
    runner_args["nnodes"] = nnodes
    runner_args["node_rank"] = node_rank
    runner_args["nproc_per_node"] = nproc_per_node
    runner_args["rdzv_backend"] = rdzv_backend
    runner_args["rdzv_endpoint"] = rdzv_endpoint
    runner_args["log_dir"] = (
        log_dir if backend == "torchrun" else os.path.join(log_dir, rdzv_id)
    )
    runner_args["redirects"] = redirect
    runner_args["tee"] = tee

    runner_cmd = [backend]
    for key, value in runner_args.items():
        if isinstance(value, bool):
            if value:
                runner_cmd.append(f"--{key}")
        else:
            runner_cmd.append(f"--{key}")
            runner_cmd.append(f"{value}")
    return runner_cmd


def _generate_run_script_train(
    config, host, node_rank, cmd, background=True, with_test=False
):
    system_config = config.train.system
    logging_config = config.train.system.logging

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
    megatron_dir = os.path.join(root_dir, "megatron")
    cmds_config = config.experiment.get("cmds", None)
    if cmds_config:
        before_start = cmds_config.get("before_start", "")
    else:
        before_start = ""
    with open(host_run_script_file, "w") as f:
        f.write("#!/bin/bash\n\n")
        f.write(f"{before_start}\n")
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
        f.write(f'cmd="{cmd}"\n')
        f.write(f"\n")
        if with_test:
            f.write(f'bash -c "$cmd; sync" \n')
        else:
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


def _generate_stop_script_train(config, host, node_rank):
    if getattr(config, "train", None):
        logging_config = config.train.system.logging
    else:
        logging_config = config.inference.system.logging

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
        f.write("if [ -f " + host_pid_file + " ]; then\n")
        f.write("    pid=$(cat " + host_pid_file + ")\n")
        f.write("    pkill -P $pid\n")
        f.write("else\n")
        # TODO: This is a temporary fix. We need to find a better way to stop the job.
        f.write("    pkill -f 'torchrun'\n")
        f.write("fi\n")
        f.write(f"{after_stop}\n")
        f.flush()
        os.fsync(f.fileno())
    os.chmod(host_stop_script_file, 0o755)

    return host_stop_script_file


class SSHTrainRunner(RunnerBase):
    def __init__(self, config: DictConfig):
        super().__init__(config)
        self.task_type = getattr(self.config.experiment.task, "type", None)
        assert self.task_type == "train", f"Unsupported task type: {self.task_type}"
        self._prepare()

    def _prepare(self):
        _update_config_train(self.config)
        self.user_args = _get_args_megatron(self.config)
        self.rdzv_id = datetime.now().strftime("%Y%m%d_%H%M%S.%f")
        self.user_envs = self.config.experiment.get("envs", {})
        self.cur_envs = None  # current node envs
        self.user_script = self.config.experiment.task.entrypoint
        self.resources = parse_hostfile(
            self.config.experiment.runner.get("hostfile", None)
        )
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
        device_type=None,
        with_test=False,
        dryrun=False,
    ):
        export_cmd = []

        for k, v in self.cur_envs.items():
            export_cmd += [f"{k}={v}"]

        runner_cmd = _get_runner_cmd_train(
            host,
            master_addr,
            master_port,
            nnodes,
            node_rank,
            nproc_per_node,
            self.config,
        )
        # update hetero-current-device-type according to the device_type in hostfile
        if device_type is not None:
            if "--hetero-current-device-type" in self.user_args:
                idx = self.user_args.index("--hetero-current-device-type")
                self.user_args[idx + 1] = device_type
            else:
                self.user_args += ["--hetero-current-device-type", device_type]

        cmd = shlex.join(export_cmd + runner_cmd + [self.user_script] + self.user_args)

        logging_config = self.config.train.system.logging
        host_run_script_file = _generate_run_script_train(
            self.config, host, node_rank, cmd, background=True, with_test=with_test
        )

        if host != "localhost":
            ssh_port = self.config.experiment.runner.get("ssh_port", 22)
            # Step 1: make sure the scripts_dir exists on the remote host
            run_ssh_command(
                host, f"mkdir -p {logging_config.scripts_dir}", ssh_port, dryrun
            )

            # Step 2: copy the host_run_script_file to the remote host
            no_shared_fs = self.config.experiment.runner.get("no_shared_fs", False)
            if no_shared_fs:
                run_scp_command(
                    host,
                    host_run_script_file,
                    logging_config.scripts_dir,
                    ssh_port,
                    dryrun,
                )

            # Step 3: run the host_run_script_file on the remote host
            run_ssh_command(host, f"bash {host_run_script_file}", ssh_port, dryrun)
        else:
            run_local_command(f"bash {host_run_script_file}", dryrun)

    def run(self, with_test=False, dryrun=False, monitor=False, interval=10):

        num_visible_devices = None
        runner_config = self.config.experiment.runner

        # If hostfile is provided, use the resources from the hostfile
        if self.resources is not None:
            nnodes_from_hostfile = len(self.resources.keys())
            nnodes_from_args = runner_config.get("nnodes", None)
            nnodes = get_nnodes(nnodes_from_hostfile, nnodes_from_args)
            available_ip = list(self.resources.keys())[0]
            available_port = get_free_port()
            for node_rank, (host, resource_info) in enumerate(self.resources.items()):
                if node_rank >= nnodes:
                    break
                self.cur_envs = add_decive_extra_config(
                    self.user_envs, resource_info["type"]
                )
                visible_devices = self.cur_envs.get("CUDA_VISIBLE_DEVICES", None)
                if visible_devices is not None and isinstance(visible_devices, str):
                    visible_devices = visible_devices.split(",")
                    num_visible_devices = len(visible_devices)
                nproc_from_hostfile = resource_info["slots"]
                nproc_from_args = runner_config.get("nproc_per_node", None)
                nproc_per_node = get_nproc_per_node(
                    nproc_from_hostfile, nproc_from_args, num_visible_devices
                )
                master_addr = runner_config.get("master_addr", available_ip)
                master_port = runner_config.get("master_port", available_port)
                self._run_each(
                    host,
                    master_addr,
                    master_port,
                    nnodes,
                    node_rank,
                    nproc_per_node,
                    device_type=resource_info["type"],
                    with_test=with_test,
                    dryrun=dryrun,
                )
        else:
            # If hostfile is not provided, run the job on localhost
            self.cur_envs = self.user_envs
            visible_devices = self.cur_envs.get("CUDA_VISIBLE_DEVICES", None)
            if visible_devices is not None and isinstance(visible_devices, str):
                visible_devices = visible_devices.split(",")
                num_visible_devices = len(visible_devices)
            nproc_from_args = runner_config.get("nproc_per_node", None)
            nproc_per_node = get_nproc_per_node(
                None, nproc_from_args, num_visible_devices
            )
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
        # If need monitor, query status continually
        if monitor:
            # sleep to wait task already started
            time.sleep(interval)
            try:
                while True:
                    status = self._query_status()
                    logger.info(f"Job Status: {status.name}")
                    if status == JobStatus.COMPLETED_OR_IDLE:
                        break
                    time.sleep(interval)
                logger.info("Job Ended.")
            except Exception as e:
                logger.info(e)

    def _stop_each(self, host, node_rank):
        host_stop_script_file = _generate_stop_script_train(
            self.config, host, node_rank
        )
        logging_config = self.config.train.system.logging

        if host != "localhost":
            ssh_port = self.config.experiment.runner.get("ssh_port", 22)
            # Step 1: make sure the scripts_dir exists on the remote host
            run_ssh_command(host, f"mkdir -p {logging_config.scripts_dir}", ssh_port)
            # Step 2: copy the host_run_script_file to the remote host
            no_shared_fs = self.config.experiment.runner.get("no_shared_fs", False)
            if no_shared_fs:
                run_scp_command(
                    host, host_stop_script_file, logging_config.scripts_dir, ssh_port
                )
            # Step 3: run the host_run_script_file on the remote host
            run_ssh_command(host, f"bash {host_stop_script_file}", ssh_port)
        else:
            run_local_command(f"bash {host_stop_script_file}")

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
        logging_config = self.config.train.system.logging

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
                "    pid=$(ps aux | grep 'torchrun' | grep -v grep | head -n 1 | awk '{print $2}')\n"
            )
            f.write("    ps -p $pid -o state --no-headers\n")
            f.write("fi\n")
            f.flush()
            os.fsync(f.fileno())
        os.chmod(host_query_script_file, 0o755)

        return host_query_script_file

    def _generate_query_sub_process_script(self, host, node_rank):
        """Genetrate the query script for each host."""
        logging_config = self.config.train.system.logging

        host_query_sub_process_script_file = os.path.join(
            logging_config.scripts_dir, f"host_{node_rank}_{host}_query_sub_process.sh"
        )

        host_pid_file = os.path.join(
            logging_config.pids_dir, f"host_{node_rank}_{host}.pid"
        )

        os.makedirs(logging_config.scripts_dir, exist_ok=True)

        with open(host_query_sub_process_script_file, "w") as f:
            f.write("#!/bin/bash\n\n")
            f.write("if [ -f " + host_pid_file + " ]; then\n")
            f.write("    pid=$(cat " + host_pid_file + ")\n")
            f.write("    ps -eo pid,ppid | awk -v ppid=$pid '$2 == ppid {print $1}'\n")
            f.write("else\n")
            # TODO: This is a temporary fix. We need to find a better way to query the job.
            f.write(
                "    pid=$(ps aux | grep 'torchrun' | grep -v grep | head -n 1 | awk '{print $2}')\n"
            )
            f.write("    ps -eo pid,ppid | awk -v ppid=$pid '$2 == ppid {print $1}'\n")
            f.write("fi\n")
            f.flush()
            os.fsync(f.fileno())
        os.chmod(host_query_sub_process_script_file, 0o755)

        return host_query_sub_process_script_file

    def _query_each(self, host, node_rank):
        "Query each node status."
        host_query_script_file = self._generate_query_script(host, node_rank)
        logging_config = self.config.train.system.logging
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

    def _query_each_sub_process(self, host, node_rank):
        "Query each node sub process status."
        host_query_script_file = self._generate_query_sub_process_script(
            host, node_rank
        )
        logging_config = self.config.train.system.logging
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
                logger.error(f"Failed to query sub process status on {host}: {e}")
        else:
            try:
                result = run_local_command(f"bash {host_query_script_file}", query=True)
            except Exception as e:
                logger.error(f"Failed to query sub process status on {host}: {e}")
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

    def _query_sub_process_status(self):
        "Query sub process status."
        results = []
        if self.resources is None:
            result = self._query_each_sub_process("localhost", 0)
            results.append(result)

        else:
            host_list = list(self.resources.keys())
            for host, _ in self.resources.items():
                node_rank = host_list.index(host)
                result = self._query_each_sub_process(host, node_rank)
                results.append(result)
        if all(status for status in results):
            status = True
        else:
            status = False
        return status

    def query(self, interval=10, timeout=None):
        """
        Query job status and log.
        There are three kinds of status for a Job:
            RUNNING: The job is running.
            COMPLETED_OR_IDLE: The job is completed or idle.
            TRANSITIONAL: The job is starting or stopping.

        Args:
            interval (int, optional): The interval of querying job status. Default: 10.
            timeout (float, optional): The timeout of query job status, if None, the query will keep indefinitely. Default: None.

        Returns:
            None

        """
        if timeout is None:
            while True:
                job_status = self._query_status()
                logger.info(f"Job status: {job_status.name}")
                time.sleep(interval)
        else:
            start_time = time.time()
            cur_time = time.time()
            while cur_time - start_time < timeout:
                job_status = self._query_status()
                logger.info(f"Job status: {job_status.name}")
                time.sleep(interval)
                cur_time = time.time()


class CloudTrainRunner(RunnerBase):
    def __init__(self, config: DictConfig):
        super().__init__(config)
        self.task_type = getattr(self.config.experiment.task, "type", None)
        assert self.task_type == "train", f"Unsupported task type: {self.task_type}"
        self._prepare()

    def _prepare(self):
        self.user_envs = self.config.experiment.get("envs", {})
        self.user_script = self.config.experiment.task.entrypoint
        _update_config_train(self.config)
        self.user_args = _get_args_megatron(self.config)

        logger.info("\n************** configuration ***********")
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

        runner_cmd = _get_runner_cmd_train(
            host,
            master_addr,
            master_port,
            nnodes,
            node_rank,
            nproc_per_node,
            self.config,
        )

        cmd = shlex.join(export_cmd + runner_cmd + [self.user_script] + self.user_args)

        host_run_script_file = _generate_run_script_train(
            self.config, host, node_rank, cmd, background=False, with_test=with_test
        )

        run_local_command(f"bash {host_run_script_file}", dryrun)

    def run(self, with_test=False, dryrun=False):
        if dryrun:
            logger.info("Dryrun mode is not supported in CloudRunner.")
            return

        num_visible_devices = None
        visible_devices = self.user_envs.get("CUDA_VISIBLE_DEVICES", None)
        if visible_devices:
            visible_devices = visible_devices.split(",")
            num_visible_devices = len(visible_devices)

        runner_config = self.config.experiment.runner
        nnodes_from_args = runner_config.get("nnodes", None)
        nnodes = get_nnodes(None, nnodes_from_args)
        node_rank = runner_config.node_rank
        nproc_from_args = runner_config.get("nproc_per_node", None)
        nproc_per_node = get_nproc_per_node(None, nproc_from_args, num_visible_devices)
        master_addr = runner_config.master_addr
        master_port = runner_config.master_port
        host = get_host_name_or_ip()
        self._run_each(
            host,
            master_addr,
            master_port,
            nnodes,
            node_rank,
            nproc_per_node,
            with_test=with_test,
            dryrun=dryrun,
        )
