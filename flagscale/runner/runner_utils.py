import collections
import os
import re
import socket
import subprocess
import collections

from flagscale.logger import logger


def log_and_raise_error(message):
    logger.error(message)
    raise ValueError(message)


def parse_hostfile(hostfile_path):
    if hostfile_path is None or not os.path.isfile(hostfile_path):
        logger.warning(
            "Hostfile not found. The training will proceed using only local resources."
        )
        return None

    # e.g., worker0 slots=8 type=A100
    pattern = re.compile(r"^(\S+)\s+slots=(\d+)(?:\s+type=(\S+))?")

    resources = collections.OrderedDict()

    with open(hostfile_path, "r") as fd:
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
                log_and_raise_error(
                    f"Hostfile contains multiple entries for host: {host}."
                )
            resources[host] = {"slots": num_slots, "type": machine_type}
        else:
            log_and_raise_error(f"Invalid entry in hostfile: {line}.")

    if len(resources) == 0:
        log_and_raise_error(
            "Hostfile is empty or not formatted correctly. Please check the hostfile."
        )

    return resources


def get_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def get_host_name_or_ip():
    host_name = socket.gethostname()
    if host_name:
        return host_name
    try:
        # doesn't even have to be reachable
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.connect(("10.255.255.255", 1))
        IP = sock.getsockname()[0]
    except Exception:
        IP = "127.0.0.1"
    finally:
        if (
            "sock" in locals()
        ):  # Ensure 'sock' was successfully created before attempting to close it
            sock.close()
    return IP


def run_local_command(cmd, dryrun=False, query=False):
    logger.info(f"Run the local command: {cmd}")
    if dryrun:
        return
    if query:
        result = subprocess.run(
            cmd, shell=True, check=True, capture_output=True, text=True
        )
        return result
    else:
        result = subprocess.run(
            cmd, shell=True, check=True, capture_output=True, text=True
        )
        if result.returncode != 0:
            print(f"Command {cmd} failed with return code {result.returncode}.")
            print(f"Output: {result.stdout}")
            print(f"Error: {result.stderr}")
            sys.exit(result.returncode)


def run_ssh_command(host, cmd, port=None, dryrun=False, query=False):
    if port:
        ssh_cmd = f"ssh -f -n -p {port} {host} '{cmd}'"
    else:
        ssh_cmd = f"ssh -f -n {host} '{cmd}'"
    if not query:
        logger.info(f"Running the ssh command: {ssh_cmd}")
    if dryrun:
        return
    if query:
        result = subprocess.run(
            ssh_cmd, shell=True, check=True, text=True, stdout=subprocess.PIPE
        )
        return result
    else:
        subprocess.run(ssh_cmd, shell=True, check=True)


def run_scp_command(host, src, dst, port=None, dryrun=False):
    if port:
        scp_cmd = f"scp -P {port} -r {src} {host}:{dst} "
    else:
        scp_cmd = f"scp -r {src} {host}:{dst} "
    logger.info(f"Run the scp command: {scp_cmd}")
    if dryrun:
        return
    subprocess.run(scp_cmd, shell=True, check=True)


def flatten_dict_to_args(config_dict, ignore_keys=[]):
    args = []
    for key, value in config_dict.items():
        if key in ignore_keys:
            continue
        key = key.replace("_", "-")
        if isinstance(value, dict):
            args.extend(flatten_dict_to_args(value, ignore_keys))
        elif isinstance(value, list):
            args.append(f"--{key}")
            for v in value:
                args.append(f"{v}")
        elif isinstance(value, bool):
            if value:
                args.append(f"--{key}")
        else:
            args.append(f"--{key}")
            args.append(f"{value}")
    return args


def get_nnodes(nnodes_from_hostfile=None, nnodes_from_args=None):
    assert nnodes_from_hostfile is not None or nnodes_from_args is not None
    if nnodes_from_hostfile is not None and nnodes_from_args is not None:
        if isinstance(nnodes_from_args, str) and ":" in nnodes_from_args:
            # Ignore the max nnodes from the args, no elastic support
            nnodes_from_args, _ = nnodes_from_args.split(":")
        return min(nnodes_from_hostfile, int(nnodes_from_args))
    elif nnodes_from_hostfile is not None:
        return nnodes_from_hostfile
    elif nnodes_from_args is not None:
        if isinstance(nnodes_from_args, str) and ":" in nnodes_from_args:
            # Ignore the max nnodes from the args, no elastic support
            nnodes_from_args, _ = nnodes_from_args.split(":")
        return int(nnodes_from_args)


def get_nproc_per_node(
    nproc_from_hostfile=None, nproc_from_args=None, num_visible_devices=None
):
    if nproc_from_hostfile is not None and nproc_from_args is not None:
        nproc = min(nproc_from_hostfile, int(nproc_from_args))
        if num_visible_devices:
            return min(nproc, num_visible_devices)
        else:
            return nproc
    elif nproc_from_hostfile is not None:
        if num_visible_devices:
            return min(nproc_from_hostfile, num_visible_devices)
        else:
            return nproc_from_hostfile
    elif nproc_from_args is not None:
        if num_visible_devices:
            return min(int(nproc_from_args), num_visible_devices)
        else:
            return nproc_from_args
    else:
        if num_visible_devices:
            return num_visible_devices
        else:
            return 1
