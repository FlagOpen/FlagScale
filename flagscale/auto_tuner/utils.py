import os
import re
import socket
import subprocess

from flagscale.launcher.runner import parse_hostfile


def divisible(x, y):
    if x % y == 0:
        return True
    return False


def beside(keys, strategy, history):
    """Compare strategy with history strategies Whether same besides given keys"""
    from .search.searcher import __BUILT_IN_STRATEGY_DIMS__

    retrieval = []
    for task in history:
        is_same = True
        for dim in task:
            if dim not in __BUILT_IN_STRATEGY_DIMS__:
                continue
            if dim in keys:
                continue
            if strategy[dim] != task[dim]:
                is_same = False
                break
        if is_same:
            retrieval.append(task)
    return retrieval


def sort_by_memory(strategy):
    """Sort strategy by memory."""
    return (
        -strategy["use_recompute"],
        -strategy["tensor_model_parallel_size"],
        (
            -strategy["sequence_parallel"]
            if strategy["sequence_parallel"] is not None
            else -float("inf")
        ),
        strategy["micro_batch_size"],
        -strategy["pipeline_model_parallel_size"],
        strategy["data_parallel_size"],
        (
            -strategy["use_distributed_optimizer"]
            if strategy["use_distributed_optimizer"] is not None
            else -float("inf")
        ),
    )


def sort_by_performance(strategy):
    """Sort strategy by performance potentially."""
    return (
        strategy["use_recompute"],
        -strategy["tensor_model_parallel_size"],
        (
            -strategy["sequence_parallel"]
            if strategy["sequence_parallel"] is not None
            else -float("inf")
        ),
        strategy["micro_batch_size"],
        strategy["pipeline_model_parallel_size"],
        -strategy["data_parallel_size"],
        (
            strategy["recompute_num_layers"]
            if strategy["recompute_num_layers"] is not None
            else float("inf")
        ),
    )


def is_ip_addr(master):
    """Check if master is ip address."""

    if not isinstance(master, str):
        return False
    pattern = (
        r"^((25[0-5]|2[0-4]\d|1\d{2}|[1-9]?\d)\.){3}(25[0-5]|2[0-4]\d|1\d{2}|[1-9]?\d)$"
    )
    result = re.match(pattern, master)
    if result:
        return True
    else:
        return False


def get_ip_addr():
    """Get ip address."""
    try:
        hostname = socket.gethostname()
        ip = socket.gethostbyname(socket.getfqdn(hostname))
    except:
        ip = "127.0.0.1"
    return ip


def is_master(config):
    """Check if current node is master."""
    nnodes = config.experiment.runner.get("nnodes", 1)

    hostfile = None
    if config.experiment.runner.get("hostfile", None):
        hostfile = config.experiment.runner["hostfile"]
    if os.environ.get("AIRS_SWITCH", None):
        if os.environ.get("AIRS_HOSTFILE_PATH", None):
            hostfile = os.environ["AIRS_HOSTFILE_PATH"]

    resources = parse_hostfile(hostfile)
    if not resources and nnodes > 1:
        raise ValueError("In the multi-node mode, please set the hostfile")

    if resources:
        master = list(resources.keys())[0]
        if is_ip_addr(master):
            return get_ip_addr() == master
        else:
            output = subprocess.run(
                "hostname", check=True, shell=True, text=True, capture_output=True
            )
            hostname = output.stdout.strip()
            return hostname == master
    # Local host Scene
    return True
