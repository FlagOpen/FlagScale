import os
import re

from datetime import datetime

from flagscale.runner.utils import logger

error_patterns = {
    "completed": {
        "pattern": r'\b(successfully\s+completed|training\s+completed|job\s+completed)\b',
        "description": "Completed: The task finished successfully with no errors."
    },
    "codeerror": {
        "pattern": r'\b(RuntimeError|ValueError|TypeError|AttributeError|ImportError|SyntaxError|NameError)\b',
        "description": "CodeError: An error was raised from the user training script."
    },
    "OutOfMemoryError": {
        "pattern": r'\b(OutOfMemoryError|CUDA out of memory|GPU memory|out of memory)\b',
        "description": "WorkerOOM: The training process ran out of GPU memory."
    },
    "evaluatoroom": {
        "pattern": r'\b(evaluator.*out of memory|evaluator.*OOM)\b',
        "description": "EvaluatorOOM: The evaluator process ran out of memory."
    },
    "workererror": {
        "pattern": r'\b(worker.*exited|worker.*failed|worker.*error|process.*died)\b',
        "description": "WorkerError: The worker process exited abnormally."
    },
    "evaluatorerror": {
        "pattern": r'\b(evaluator.*exited|evaluator.*failed|evaluator.*error)\b',
        "description": "EvaluatorError: The evaluator process exited abnormally."
    },
    "nodecheckfailed": {
        "pattern": r'\b(node.*check.*failed|health.*check.*failed|node.*unreachable)\b',
        "description": "NodeCheckFailed: Node health check failed."
    },
    "hangerror": {
        "pattern": r'\b(hang|hung|no.*progress|stuck|timeout.*progress)\b',
        "description": "HangError: The training task made no progress for a long time."
    },
    "rdzvtimeout": {
        "pattern": r'\b(rendezvous.*timeout|rdzv.*timeout|synchronization.*timeout)\b',
        "description": "RdzvTimeout: Rendezvous timeout, nodes in multi-node training failed to synchronize initialization within the allowed time."
    },
    "pendingtimeout": {
        "pattern": r'\b(pending.*timeout|queue.*timeout|waiting.*timeout)\b',
        "description": "PendingTimeout: The task waited too long in the scheduling queue without being started."
    },
    "uncompletedtimeout": {
        "pattern": r'\b(uncompleted.*timeout|execution.*timeout|task.*timeout)\b',
        "description": "UncompletedTimeout: The task did not finish within the specified time."
    },
    "storageerror": {
        "pattern": r'\b(storage.*error|checkpoint.*error|dataset.*error|I/O.*error|disk.*error)\b',
        "description": "StorageError: Dataset or checkpoint storage system error."
    },
    "signalException": {
        "pattern": r'\b(killed.*by.*user|SIGTERM|SIGINT|manually.*stopped|user.*interrupted)\b',
        "description": "KilledByUser: The task was manually stopped by the user."
    },
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
                confidence_scores = []
                
                for error_type, error_info in error_patterns.items():
                    pattern = error_info["pattern"]
                    description = error_info["description"]
                    
                    # Use regular expressions for case-insensitive matching
                    matches = re.findall(pattern, log_content, re.IGNORECASE)
                    if matches:
                        # Calculate confidence score (based on number of matches)
                        confidence = min(len(matches) * 0.1, 1.0)
                        matched_errors.append((description, confidence, len(matches)))
                        confidence_scores.append(confidence)

                if matched_errors:
                    # Sort by confidence
                    matched_errors.sort(key=lambda x: x[1], reverse=True)
                    for desc, confidence, count in matched_errors:
                        report_content += f"- {desc} (confidence: {confidence:.1f}, match_count: {count})\n"
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
