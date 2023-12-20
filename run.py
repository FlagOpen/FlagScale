import os
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


def print_config_changes(config, added, deleted, modified):
    print(f"\n{' Config ':=^40}")
    print(f"\n{' Final Config ':-^40}")
    print(json.dumps(config, indent=4))

    print(f"\n{' Added Config ':-^40}")
    if added:
        for key, value in added.items():
            print(f"{key} (+): {value}")
    else:
        print("No added config.")

    print(f"\n{' Deleted Config ':-^40}")
    if deleted:
        for key, value in deleted.items():
            print(f"{key} (-): {value}")
    else:
        print("No deleted config.")

    print(f"\n{' Modified Config ':-^40}")
    if modified:
        for key, values in modified.items():
            old_value, new_value = values
            print(f"{key}: {old_value} => {new_value}")
    else:
        print("No modified config.")

    print(f"\n{'='*40}")


def deep_update(source, overrides):
    for key, value in overrides.items():
        if isinstance(value, dict):
            # Get node or create one
            node = source.setdefault(key, {})
            deep_update(node, value)
        else:
            source[key] = value
    return source


def remove_comments(config):
    if isinstance(config, dict):
        return {
            k: remove_comments(v)
            for k, v in config.items()
            if not k.startswith("__comment__")
        }
    elif isinstance(config, list):
        return [remove_comments(v) for v in config]
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
                modified[k] = (d2[k], d1[k])

    return added, deleted, modified


def generate_config(
    predefined_config_path,
    user_base_config_path,
    user_extra_config_path=None,
    print_config=False,
):
    """
    Generates a configuration based on predefined and user-provided configurations.

    This function takes paths to predefined and user-provided configuration files, merges them, and
    returns a final configuration. If a user-provided extra configuration file path is given, it is
    also merged into the final configuration. If the `print_config` flag is set to True, the final
    configuration is printed to the console.

    Args:
        predefined_config_path (str): A string representing the path to the predefined configuration file.
        user_base_config_path (str): A string representing the path to the user-provided base configuration file.
        user_extra_config_path (str, optional): A string representing the path to the user-provided extra
                                                 configuration file. Defaults to None.
        print_config (bool, optional): A flag indicating whether to print the final configuration to the console.
                                        Defaults to False.

    Returns:
        dict: A dictionary representing the final merged configuration.
    """
    # Load default config
    with open(predefined_config_path, "r") as f:
        config = json.load(f)
        config = remove_comments(config)

    # Load and merge user base config
    with open(user_base_config_path, "r") as f:
        user_base_config = json.load(f)
        user_base_config = remove_comments(user_base_config)
        deep_update(config, user_base_config)

    # Backup for comparison later
    base_config = copy.deepcopy(config)

    # Load and merge user extra config if provided
    if user_extra_config_path:
        with open(user_extra_config_path, "r") as f:
            user_extra_config = json.load(f)
            user_extra_config = remove_comments(user_extra_config)
            deep_update(config, user_extra_config)

    # print(json.dumps(base_config, indent=4))
    # print(json.dumps(config, indent=4))
    # Call the function
    added, deleted, modified = diff_dict(base_config, config)

    if print_config:
        print_config_changes(config, added, deleted, modified)

    return config


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
    shell_cmds = config["experiment"]["shell_cmds"]
    env_args = 'ENV_ARGS="\n' + config_to_args(config["env_vars"], is_env=True) + '\n"'
    launch_args = 'LAUNCH_ARGS="\n' + config_to_args(config["launch"]) + '\n"'
    entry_script = "pretrain_gpt.py"

    def _config_to_arg(key, value):
        key = key.replace("_", "-")
        return f"    --{key} {value}"

    other_args_groups = [
        f'{key.upper()}_ARGS="\n{config_to_args(value) if isinstance(value, dict) else _config_to_arg(key, value)}\n"'
        for key, value in config.items()
        if key not in ["experiment", "env_vars", "launch", "shell_cmds"]
    ]

    cmd = f'cmd="\n    $ENV_ARGS \\\n    torchrun $LAUNCH_ARGS \\\n    {entry_script}'
    for other_args in other_args_groups:
        cmd += f" \\\n    ${other_args.split('=')[0]}"
    cmd += '\n"'

    bash_script = f"#!/bin/bash\n\n{shell_cmds}\n\n{env_args}\n\n{launch_args}\n\n"
    for other_args in other_args_groups:
        bash_script += f"{other_args}\n\n"
    bash_script += f"{cmd}\n\necho $cmd\neval $cmd"

    return bash_script


def print_cmd(host, cmd):
    print(f"\n{ host + ' ':=^40}")
    print(f"{cmd}")
    print(f"{'='*40}\n")


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
    expr_config = config["experiment"]
    hostfile = expr_config.get("hostfile", None)
    ssh_port = expr_config.get("ssh_port", 22)
    log_dir = expr_config["log_dir"]
    if log_dir is None:
        log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)

    if hostfile is not None:
        with open(hostfile, "r") as file:
            lines = file.read().splitlines()
    else:
        lines = ["localhost"]

    procs = []
    node_rank = 0
    for line in lines:
        host = line.split()[0]
        if node_rank == 0:
            master_addr = host
        launch_config = config["launch"]
        launch_config["nnodes"] = len(lines)
        launch_config["node_rank"] = node_rank
        if hostfile is not None:
            nproc_per_node = int(line.split()[1].split("=")[1])
        else:
            nproc_per_node = (
                1
                if launch_config["nproc_per_node"] is None
                else launch_config["nproc_per_node"]
            )
        launch_config["nproc_per_node"] = nproc_per_node
        launch_config["master_addr"] = master_addr

        bash_script = generate_command(config)
        bash_file = f"{log_dir}/{node_rank}_{host}.sh"
        bash_file = os.path.abspath(bash_file)
        with open(bash_file, "w") as f:
            f.write(bash_script)

        cmd = f"cd {os.path.join(expr_config['home_dir'], 'megatron')}; nohup bash {bash_file} 2>&1 &"
        wrapped_cmd = f"'bash -c \"{cmd}\"'"
        ssh_cmd_parts = (
            ["ssh", "-f", "-n", "-p", str(ssh_port), host, wrapped_cmd]
            if hostfile is not None
            else [cmd]
        )
        ssh_cmd = " ".join(ssh_cmd_parts)

        log_file = f"{log_dir}/{node_rank}_{host}.log.txt"
        with open(log_file, "w") as f:
            print_cmd(host, ssh_cmd)
            if not generate_only:
                p = subprocess.Popen(
                    ssh_cmd, shell=True, stdout=f, stderr=subprocess.STDOUT
                )
                procs.append(p)
        node_rank += 1

    for p in procs:
        p.wait()


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
    expr_config = config["experiment"]
    hostfile = expr_config.get("hostfile", None)
    ssh_port = expr_config.get("ssh_port", 22)
    if hostfile is not None:
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

    print_config = True if args.action != "stop" else False

    # Step1: generate config
    predefined_config_path = os.path.join(
        os.path.dirname(__file__), "predefined_args_megatron.json"
    )
    config = generate_config(
        predefined_config_path, args.config, args.extra_config, print_config
    )

    # Step2: perform action based on args.action
    if args.action == "generate":
        run_experiment(config, generate_only=True)
    elif args.action == "run":
        run_experiment(config, generate_only=False)
    elif args.action == "stop":
        stop_experiment(config, stop_key=args.stop_key)


if __name__ == "__main__":
    main()
