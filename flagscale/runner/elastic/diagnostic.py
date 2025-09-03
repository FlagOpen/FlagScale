import os

from datetime import datetime

from flagscale.runner.utils import logger

error_types = {
    "completed": "Completed: The task finished successfully with no errors.",
    "codeerror": "CodeError: An error was raised from the user training script.",
    "OutOfMemoryError": "WorkerOOM: The training process ran out of GPU memory.",
    "evaluatoroom": "EvaluatorOOM: The evaluator process ran out of memory.",
    "workererror": "WorkerError: The worker process exited abnormally.",
    "evaluatorerror": "EvaluatorError: The evaluator process exited abnormally.",
    "nodecheckfailed": "NodeCheckFailed: Node health check failed.",
    "hangerror": "HangError: The training task made no progress for a long time.",
    "rdzvtimeout": "RdzvTimeout: Rendezvous timeout, nodes in multi-node training failed to synchronize initialization within the allowed time.",
    "pendingtimeout": "PendingTimeout: The task waited too long in the scheduling queue without being started.",
    "uncompletedtimeout": "UncompletedTimeout: The task did not finish within the specified time.",
    "storageerror": "StorageError: Dataset or checkpoint storage system error.",
    "signalException": "KilledByUser: The task was manually stopped by the user.",
}


def generate_diagnostic_report(config, host, node_rank, log_file, return_content=False):
    """
    Generate a diagnostic report from a log file.
    Args:
        config (DictConfig): Configuration object.
        host (str): Hostname or IP.
        node_rank (int): Node rank.
        log_file (str): Path to the log file.
        return_content (bool): If True, return report as string instead of writing to file.
    Returns:
        str: Diagnostic report content if return_content=True, else None.
    """
    report_content = f"Diagnostic Report for {host} (node {node_rank})\n"
    report_content += f"Generated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    # report_content += f"Log file analyzed: {log_file}\n"
    report_content += "Analysis:\n"

    try:
        if not os.path.exists(log_file) or os.path.getsize(log_file) == 0:
            report_content += "- Log file is empty or does not exist, no analysis possible.\n"
            return report_content if return_content else None
        else:
            with open(log_file, 'r') as f:
                log_content = f.read()
            if not log_content.strip():
                report_content += "- Log file is empty, no analysis possible.\n"
            else:
                matched_errors = []
                for key, desc in error_types.items():
                    if key in log_content.lower():
                        matched_errors.append(desc)

                if matched_errors:
                    for err in matched_errors:
                        report_content += f"- {err}\n"
                else:
                    report_content += "- No errors or unknown error detected in logs.\n"
    except Exception as e:
        logger.error(f"Failed to read log file {log_file} for {host} (node {node_rank}): {e}")
        report_content += f"- Error reading log file: {e}\n"

    if return_content:
        return report_content
    else:
        try:
            diagnostic_file = log_file.replace("temp", "diagnostic").replace(".log", ".txt")
            os.makedirs(os.path.dirname(diagnostic_file), exist_ok=True)
            with open(diagnostic_file, 'w') as f:
                f.write(report_content)
            logger.debug(
                f"Generated diagnostic report for {host} (node {node_rank}) at {diagnostic_file}"
            )
            return diagnostic_file
        except Exception as e:
            logger.error(f"Failed to write diagnostic report for {host} (node {node_rank}): {e}")
            return None
