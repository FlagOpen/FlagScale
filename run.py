import os
import re
import json
import copy
import argparse
import subprocess


def parse_args():
    parser = argparse.ArgumentParser(description="FlagScale runner")

    parser.add_argument(
        "--config", required=True, type=str, help="Path to the configuration file"
    )
    parser.add_argument(
        "--extra-config",
        default=None,
        type=str,
        help="Path to the extra configuration file",
    )
    parser.add_argument(
        "--action",
        choices=["generate", "run", "stop"],
        default="run",
        help="Action to perform: generate the bash script, run the experiment, or stop the experiment",
    )
    parser.add_argument(
        "--stop-key",
        default="torchrun",
        type=str,
        help="Key to match the process name to stop the experiment",
    )

    return parser.parse_args()


def get_config(config, key, default=None):
    for k, v in config.items():
        if k == key:
            return v
        elif isinstance(v, dict):
            item = get_config(v, key)
            if item is not None:
                return item
    return default


def set_config(config, key, value, override=True):
    def _set_config_recursive(config, key, value, override):
        if key in config:
            if override:
                config[key] = value
                return True
            else:
                return False
        else:
            for v in config.values():
                if isinstance(v, dict):
                    if _set_config_recursive(v, key, value, override):
                        return True
            return False
    
    has_set = _set_config_recursive(config, key, value, override)

    if not has_set:
        config[key] = value


def merge_config(base_config, extra_config):
    for key, value in extra_config.items():
        if isinstance(value, dict):
            # Get node or create one
            node = base_config.setdefault(key, {})
            merge_config(node, value)
        else:
            base_config[key] = value
    return base_config


def print_config(config, added=None, deleted=None, modified=None):
    print(f"\n{' Final Config ':-^40}")
    print(json.dumps(config, indent=4))
    print('-' * 40)

    print(f"\n{' Added Config ':-^40}")
    if added:
        for key, value in added.items():
            print(f"{key} (+): {value}")
    else:
        print("No added config.")
    print('-' * 40)

    print(f"\n{' Deleted Config ':-^40}")
    if deleted:
        for key, value in deleted.items():
            print(f"{key} (-): {value}")
    else:
        print("No deleted config.")
    print('-' * 40)

    print(f"\n{' Modified Config ':-^40}")
    if modified:
        for key, values in modified.items():
            old_value, new_value = values
            print(f"{key}: {old_value} => {new_value}")
    else:
        print("No modified config.")
    print('-' * 40)


def remove_config_comment(config):
    if isinstance(config, dict):
        return {
            k: remove_config_comment(v)
            for k, v in config.items()
            if not k.startswith("__comment__")
        }
    elif isinstance(config, list):
        return [remove_config_comment(v) for v in config]
    else:
        return config


def diff_dict(d1, d2):
    added = {}
    deleted = {}
    modified = {}

    for k in d2.keys() - d1.keys():
        added[k] = d2[k]
    for k in d1.keys() - d2.keys():
        deleted[k] = d1[k]
    for k in d2.keys() & d1.keys():
        if isinstance(d2[k], dict) and isinstance(d1[k], dict):
            a, d, m = diff_dict(d1[k], d2[k])
            added.update(a)
            deleted.update(d)
            modified.update(m)
        else:
            if d2[k] != d1[k]:
                modified[k] = (d1[k], d2[k])

    return added, deleted, modified


def generate_config(
    predefined_config_path,
    user_base_config_path,
    user_extra_config_path=None
):
    """
    Generates a configuration based on predefined and user-provided configurations.

    This function takes paths to predefined and user-provided configuration files, merges them, and
    returns a final configuration. If a user-provided extra configuration file path is given, it is
    also merged into the final configuration.

    Args:
        predefined_config_path (str): A string representing the path to the predefined configuration file.
        user_base_config_path (str): A string representing the path to the user-provided base configuration file.
        user_extra_config_path (str, optional): A string representing the path to the user-provided extra
                                                 configuration file. Defaults to None.

    Returns:
        dict: A dictionary representing the final merged configuration.
    """
    # Load default config
    with open(predefined_config_path, "r") as f:
        config = json.load(f)
        config = remove_config_comment(config)

    # Load and merge user base config
    with open(user_base_config_path, "r") as f:
        user_base_config = json.load(f)
        user_base_config = remove_config_comment(user_base_config)
        merge_config(config, user_base_config)

    # Backup for comparison later
    base_config = copy.deepcopy(config)

    # Load and merge user extra config if provided
    if user_extra_config_path:
        with open(user_extra_config_path, "r") as f:
            user_extra_config = json.load(f)
            user_extra_config = remove_config_comment(user_extra_config)
            merge_config(config, user_extra_config)

    return config, base_config


def config_to_args(config, is_env=False):
    def recurse_config(config):
        args = []
        for i, (key, value) in enumerate(config.items()):
            if key == "_comment" or value is None or value is False:
                continue
            if not is_env:
                key = key.replace("_", "-")
            if isinstance(value, dict):
                args.append(f'{key.upper()}-ARGS="')
                args.extend(recurse_config(value))
                args.append('"')
            elif isinstance(value, list):
                if is_env:
                    args.append(
                        f'    {key}={",".join(map(str, value))}'
                        + (" \\" if i < len(config) - 1 else "")
                    )
                else:
                    args.append(
                        f'    --{key} {" ".join(map(str, value))}'
                        + (" \\" if i < len(config) - 1 else "")
                    )
            else:
                if value is True:
                    args.append(f"    --{key}" + (" \\" if i < len(config) - 1 else ""))
                else:
                    if is_env:
                        args.append(
                            f"    {key}={value}"
                            + (" \\" if i < len(config) - 1 else "")
                        )
                    else:
                        args.append(
                            f"    --{key} {value}"
                            + (" \\" if i < len(config) - 1 else "")
                        )
        return args

    return "\n".join(recurse_config(config))


def print_cmd(host, cmd):
    print(f"\n{ 'run on ' + host + ' ':-^40}\n{cmd}\n{'-'*40}\n")


def generate_mkdir_cmds(config):
    auto_mkdir = get_config(config, 'auto_mkdir')
    log_dir = get_config(config, 'log_dir')
    assert os.path.exists(log_dir), f"Log directory {log_dir} does not exist."

    exp_name = get_config(config, 'exp_name')
    base_dir = os.path.join(log_dir, exp_name)

    load_dir = get_config(config, 'load')
    if not load_dir and auto_mkdir:
        load_dir = os.path.join(base_dir, 'ckpt')
        set_config(config, 'load', load_dir)

    save_dir = get_config(config, 'save')
    if not save_dir and auto_mkdir:
        # Use the same directory as load_dir if not specified
        save_dir = os.path.join(base_dir, 'ckpt')
        set_config(config, 'save', save_dir)

    tensorboard_dir = get_config(config, 'tensorboard_dir')
    if not tensorboard_dir and auto_mkdir:
        tensorboard_dir = os.path.join(base_dir, 'tensorboard')
        set_config(config, 'tensorboard_dir', tensorboard_dir)

    wandb_dir = get_config(config, 'wandb_dir')
    if not wandb_dir and auto_mkdir:
        wandb_dir = os.path.join(base_dir, 'wandb')
        set_config(config, 'wandb_dir', wandb_dir)

    mkdir_cmds = f"mkdir -p {load_dir}\n" \
                 f"mkdir -p {save_dir}\n" \
                 f"mkdir -p {tensorboard_dir}\n" \
                 f"mkdir -p {wandb_dir}\n"
    return mkdir_cmds


def generate_command(config):
    """
    Generates a command based on the provided configuration.

    This function takes a configuration object and generates a command that can be executed in a shell.
    The configuration object should contain all the necessary information to build the command.

    Args:
        config (dict): A dictionary containing the configuration parameters.

    Returns:
        str: A string representing the command to be executed.
    """
    mkdir_cmds = generate_mkdir_cmds(config)
    shell_cmds = get_config(config, "shell_cmds")
    env_args = 'ENV_ARGS="\n' + config_to_args(get_config(config, "env_vars"), is_env=True) + '\n"'
    launch_args = 'LAUNCH_ARGS="\n' + config_to_args(get_config(config, "launch")) + '\n"'
    entry_script = "pretrain_gpt.py"

    args_groups = []
    other_conifg = {} 
    for key, value in config.items():
        if key not in ["experiment", "env_vars", "launch", "shell_cmds"]:
            if isinstance(value, dict):
                args_groups.append(
                    f'{key.upper()}_ARGS="\n{config_to_args(value)}\n"'
                )
            else:
                other_conifg[key] = value
    other_group = f'OTHER_ARGS="\n{config_to_args(other_conifg)}\n"'
    args_groups.append(other_group)

    cmd = f'cmd="\n    $ENV_ARGS \\\n    torchrun $LAUNCH_ARGS \\\n    {entry_script}'
    for args in args_groups:
        cmd += f" \\\n    ${args.split('=')[0]}"
    cmd += '\n"'

    bash_script = f"#!/bin/bash\n\n{shell_cmds}\n\n{mkdir_cmds}\n\n{env_args}\n\n{launch_args}\n\n"
    for args in args_groups:
        bash_script += f"{args}\n\n"
    bash_script += f"{cmd}\n\necho $cmd\neval $cmd"

    return bash_script


def create_ssh_cmd(host, ssh_port, cmd, remote=False):
    wrapped_cmd = f"'bash -c \"{cmd}\"'"
    ssh_cmd_parts = (
        ["ssh", "-f", "-n", "-p", str(ssh_port), host, wrapped_cmd]
        if remote
        else [cmd]
    )
    ssh_cmd = " ".join(ssh_cmd_parts)
    print_cmd(host, ssh_cmd)
    return ssh_cmd


def create_scp_cmd(ssh_port, source_file, host, destination_file):
    scp_cmd_parts = ["scp", "-P", str(ssh_port), source_file, f"{host}:{destination_file}"]
    scp_cmd = " ".join(scp_cmd_parts)
    print_cmd(host, scp_cmd)
    return scp_cmd


def get_valid_hostfile_lines(hostfile):
    if not os.path.exists(hostfile):
        return ['localhost slots=1'] 

    valid_lines = []
    with open(hostfile, 'r') as file:
        for line in file:
            line = line.strip()
            # Skip empty lines and comment lines
            if line == "" or line.startswith("#"):
                continue
            # Check if the line matches the format "a slots=b c"
            if re.match(r'^\S+(\s+slots=\d+)?(\s+\S+)?$', line):
                valid_lines.append(line)
            else:
                raise ValueError(f"Invalid line in {hostfile}: {line}")
    return valid_lines


def run_experiment(config, generate_only=False):
    """
    Runs or generates an experiment based on the provided configuration.

    This function takes a configuration object and either runs an experiment or generates the commands
    for an experiment based on the `generate_only` flag. The configuration object should contain all
    the necessary information to execute the experiment.

    This function can run experiments locally or on remote hosts. The remote hosts are specified in
    the hostfile from the config args.

    Args:
        config (dict): A dictionary containing the configuration parameters for the experiment.
        generate_only (bool, optional): A flag indicating whether to only generate the commands for
                                         the experiment without executing them. Defaults to False.

    Returns:
        None
    """
    exp_config = get_config(config, "experiment")
    exp_name = get_config(exp_config, "exp_name")
    if exp_name is None:
        exp_name = "default"
        set_config(exp_config, "exp_name", "default")
    hostfile = get_config(exp_config, "hostfile", None)
    no_shared_fs = get_config(exp_config, "no_shared_fs")
    ssh_port = get_config(exp_config, "ssh_port", 22)
    log_dir = get_config(exp_config, "log_dir")
    if not log_dir:
        log_dir = 'logs'
        set_config(exp_config, 'log_dir', log_dir)
    exp_dir = os.path.join(log_dir, exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    lines = get_valid_hostfile_lines(hostfile)

    node_rank = 0
    for line in lines:
        launch_config = get_config(config, "launch")

        host = line.split()[0]
        master_addr = get_config(launch_config, "master_addr")
        if master_addr is None:
            if node_rank == 0:
                master_addr = host
            set_config(launch_config, "master_addr", master_addr)

        set_config(launch_config, "nnodes", len(lines))
        set_config(launch_config, "node_rank", node_rank)

        slots = None
        if 'slots=' in line:
            slots = int(line.split()[1].split("=")[1])
        nproc_per_node = get_config(launch_config, "nproc_per_node")
        if slots is None and nproc_per_node is None:
            slots = 1
        elif slots is None:
            slots = nproc_per_node
        set_config(launch_config, "nproc_per_node", slots)


        bash_script = generate_command(config)
        bash_file = f"{exp_dir}/{node_rank}_{host}.sh"
        bash_file = os.path.abspath(bash_file)
        with open(bash_file, "w") as f:
            f.write(bash_script)

        if no_shared_fs:
            scp_cmd = create_scp_cmd(ssh_port, bash_file, host, bash_file)
        
        # Execute the bash script background 
        home_dir = os.path.dirname(os.path.realpath(__file__))
        log_file = f"{exp_dir}/{node_rank}_{host}.log.txt"
        cmd = f"cd {os.path.join(home_dir, 'megatron')}; nohup bash {bash_file} > {log_file} 2>&1 &"
        ssh_cmd = create_ssh_cmd(host, ssh_port, cmd, remote=hostfile is not None)

        if not generate_only:
            if no_shared_fs:
                subprocess.run(scp_cmd, shell=True, check=True)
            subprocess.run(ssh_cmd, shell=True, check=True)

        node_rank += 1
        set_config(exp_config, 'node_rank', node_rank)


def stop_experiment(config, stop_key):
    """
    Stops an ongoing experiment based on the provided configuration and stop key.

    This function takes a configuration object and a stop key, and stops the experiment accordingly.
    The configuration object should contain all the necessary information to stop the experiment,
    and the stop key is used to match the process name of the experiment to be stopped.

    Args:
        config (dict): A dictionary containing the configuration parameters for the experiment.
        stop_key (str): A string used to matching the process name of the experiment to be stopped.

    Returns:
        None
    """
    hostfile = get_config(config, "hostfile", None)
    ssh_port = get_config(config, "ssh_port", 22)
    if hostfile is not None and os.path.exists(hostfile):
        with open(hostfile, "r") as file:
            lines = file.read().splitlines()
    else:
        lines = ["localhost"]

    for line in lines:
        host = line.split()[0]
        cmd = f"pkill -f {stop_key}"
        if hostfile is None:
            ssh_cmd = cmd
        else:
            ssh_cmd = f"ssh -p {str(ssh_port)} {host} '{cmd}'"
        print_cmd(host, ssh_cmd)
        subprocess.run(ssh_cmd, shell=True)


def main():
    args = parse_args()

    is_print_config = True if args.action != "stop" else False

    # Step1: generate config
    predefined_config_path = os.path.join(
        os.path.dirname(__file__), "predefined_args_megatron.json"
    )
    config, base_config = generate_config(
        predefined_config_path, args.config, args.extra_config
    )

    # Step2: perform action based on args.action
    if args.action == "generate":
        run_experiment(config, generate_only=True)
    elif args.action == "run":
        run_experiment(config, generate_only=False)
    elif args.action == "stop":
        stop_experiment(config, stop_key=args.stop_key)
    
    # Step3: print final config
    if is_print_config:
        added, deleted, modified = diff_dict(base_config, config)
        print_config(config, added, deleted, modified)


if __name__ == "__main__":
    main()
