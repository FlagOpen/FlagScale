import os

from datetime import datetime

from flagscale.runner.utils import logger


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
    report_content += f"Log file analyzed: {log_file}\n"
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
            elif "error" in log_content.lower():
                report_content += "- Found error in logs, please check for issues.\n"
            else:
                report_content += "- No errors detected in logs.\n"
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
            logger.info(
                f"Generated diagnostic report for {host} (node {node_rank}) at {diagnostic_file}"
            )
            return diagnostic_file
        except Exception as e:
            logger.error(f"Failed to write diagnostic report for {host} (node {node_rank}): {e}")
            return None
