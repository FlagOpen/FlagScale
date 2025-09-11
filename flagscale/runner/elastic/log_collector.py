import os
import shlex

from datetime import datetime

from flagscale.runner.utils import logger, run_local, run_scp

_log_offsets = {}


def collect_logs(config, host, node_rank, destination_dir, dryrun=False):
    """
    Collect logs incrementally from a specified host and node rank, saving to destination_dir.
    Args:
        config (DictConfig): Configuration object containing experiment and logging details.
        host (str): Hostname or IP of the node.
        node_rank (int): Rank of the node.
        destination_dir (str): Directory to store collected logs.
        dryrun (bool): If True, simulate the collection without executing commands.
    Returns:
        str: Path to the collected log file.
    """
    logging_config = config.train.system.logging
    no_shared_fs = config.experiment.runner.get("no_shared_fs", False)
    log_dir = logging_config.log_dir
    src_log_file = os.path.join(
        log_dir, f"host{'_' + str(node_rank) + '_' + host if not no_shared_fs else ''}.output"
    )
    dest_log_file = os.path.join(
        destination_dir,
        f"host_{node_rank}_{host}_temp_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
    )

    os.makedirs(destination_dir, exist_ok=True)

    log_key = f"{host}_{node_rank}"
    offset = _log_offsets.get(log_key, 0)

    try:
        if host != "localhost":
            ssh_port = config.experiment.runner.get("ssh_port", 22)
            command = f"tail -c +{offset + 1} {src_log_file} > {dest_log_file}"
            run_scp(host, src_log_file, dest_log_file, ssh_port, dryrun, incremental=True)
            logger.debug(
                f"Collected incremental log from {host} (node {node_rank}) to {dest_log_file}"
            )
        else:
            command = f"tail -c +{offset + 1} {src_log_file} > {dest_log_file}"
            run_local(command, dryrun)
            logger.debug(f"Collected incremental local log to {dest_log_file}")

        if os.path.exists(dest_log_file) and os.path.getsize(dest_log_file) > 0:
            new_offset = os.path.getsize(src_log_file)
            _log_offsets[log_key] = new_offset
            return dest_log_file
        else:
            logger.warning(f"Log file {src_log_file} not found or empty")
            if os.path.exists(dest_log_file):
                os.remove(dest_log_file)
            return None

    except Exception as e:
        logger.error(f"Failed to collect logs from {host} (node {node_rank}): {e}")
        if os.path.exists(dest_log_file):
            os.remove(dest_log_file)
        return None
