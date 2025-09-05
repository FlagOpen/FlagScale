import os
import shlex
import threading
import time

from datetime import datetime

from flagscale.runner.utils import logger, run_local, run_scp


class LogCollector:
    def __init__(self):
        self.offsets = {}
        self.lock = threading.Lock()
        
    def reset_offset(self, host, node_rank):
        """Resets the log offset for the specified node"""
        log_key = f"{host}_{node_rank}"
        with self.lock:
            if log_key in self.offsets:
                del self.offsets[log_key]
                logger.debug(f"Reset log offset for {host} (node {node_rank})")
    
    def get_offset(self, host, node_rank):
        """Gets the log offset for the specified node"""
        log_key = f"{host}_{node_rank}"
        with self.lock:
            return self.offsets.get(log_key, 0)
    
    def update_offset(self, host, node_rank, new_offset):
        """Updates the log offset for the specified node"""
        log_key = f"{host}_{node_rank}"
        with self.lock:
            self.offsets[log_key] = new_offset

# Global log collector
_global_collector = LogCollector()


def collect_logs(config, host, node_rank, destination_dir, dryrun=False, process_running=True):
    """
    Collect logs incrementally from a specified host and node rank, saving to destination_dir.
    Args:
        config (DictConfig): Configuration object containing experiment and logging details.
        host (str): Hostname or IP of the node.
        node_rank (int): Rank of the node.
        destination_dir (str): Directory to store collected logs.
        dryrun (bool): If True, simulate the collection without executing commands.
        process_running (bool): If False, collect all remaining logs ignoring offset.
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

    # Gets the offset
    offset = _global_collector.get_offset(host, node_rank) if process_running else 0

    try:
        if not os.path.exists(src_log_file):
            logger.warning(f"Source log file {src_log_file} not found")
            return None
            
        if host != "localhost":
            ssh_port = config.experiment.runner.get("ssh_port", 22)
            if process_running:
                command = f"tail -c +{offset + 1} {src_log_file}"
            else:
                command = f"cat {src_log_file}"
            
            run_scp(host, src_log_file, dest_log_file, ssh_port, dryrun, incremental=process_running)
            logger.debug(
                f"Collected {'incremental' if process_running else 'final'} log from {host} (node {node_rank}) to {dest_log_file}"
            )
        else:
            if process_running:
                command = f"tail -c +{offset + 1} {src_log_file} > {dest_log_file}"
            else:
                command = f"cat {src_log_file} > {dest_log_file}"
            run_local(command, dryrun)
            logger.debug(f"Collected {'incremental' if process_running else 'final'} local log to {dest_log_file}")

        if os.path.exists(dest_log_file) and os.path.getsize(dest_log_file) > 0:
            if process_running:
                # Update the offset
                new_offset = os.path.getsize(src_log_file) if os.path.exists(src_log_file) else 0
                _global_collector.update_offset(host, node_rank, new_offset)
            return dest_log_file
        else:
            logger.debug(f"No new log content collected from {src_log_file}")
            if os.path.exists(dest_log_file):
                os.remove(dest_log_file)
            return None

    except Exception as e:
        logger.error(f"Failed to collect logs from {host} (node {node_rank}): {e}")
        if os.path.exists(dest_log_file):
            try:
                os.remove(dest_log_file)
            except:
                pass
        return None


def reset_log_collection(host, node_rank):
    """Resets the log collection state of the specified node"""
    _global_collector.reset_offset(host, node_rank)


def cleanup_log_collection():
    """Clean up any log collection state"""
    with _global_collector.lock:
        _global_collector.offsets.clear()
    logger.debug("Cleaned up all log collection states")
