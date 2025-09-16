import os
import time

from flagscale.runner.utils import logger

# Record the number of rows last diagnosed and analyzed by each host
_diagnostic_offsets = {}

error_types = {
    # Success indicators
    "completed": "Completed: The task finished successfully with no errors.",
    # Memory errors
    "out of memory": "OutOfMemoryError: The training process ran out of GPU memory.",
    "outofmemoryerror": "OutOfMemoryError: The training process ran out of GPU memory.",
    "cuda out of memory": "OutOfMemoryError: CUDA out of memory error occurred.",
    # Connection and network errors
    "rendezvousconnectionerror": "RendezvousConnectionError: Connection to rendezvous backend failed.",
    "rendezvous": "RendezvousError: Rendezvous coordination failed between nodes.",
    "connection refused": "ConnectionError: Network connection refused.",
    "connection timeout": "ConnectionTimeout: Network connection timeout.",
    # Import and code errors
    "importerror": "ImportError: Failed to import required modules.",
    "modulenotfounderror": "ModuleNotFoundError: Required Python module not found.",
    "traceback": "CodeError: Python exception occurred during execution.",
    "error": "GeneralError: An error occurred during training.",
    # Process errors
    "killed": "ProcessKilled: Training process was killed.",
    "kill": "ProcessKilled: Process was manually killed.",
    "sigterm": "ProcessKilled: Process received SIGTERM signal.",
    "sigkill": "ProcessKilled: Process received SIGKILL signal.",
    "signal": "ProcessKilled: Process received termination signal.",
    "terminated": "ProcessKilled: Process was terminated.",
    "sigint": "ProcessKilled: Process received SIGINT signal (Ctrl+C).",
    "segmentation fault": "SegmentationFault: Process crashed due to memory access error.",
    "core dumped": "CoreDump: Process crashed and dumped core.",
    # CUDA errors
    "cuda": "CUDAError: CUDA-related error occurred.",
    "cudnn": "CUDNNError: CuDNN library error occurred.",
    "gpu": "GPUError: GPU-related error occurred.",
    # File and storage errors
    "no such file": "FileNotFound: Required file or directory not found.",
    "permission denied": "PermissionError: File permission denied.",
    "disk space": "StorageError: Insufficient disk space.",
    # Timeout errors
    "timeout": "TimeoutError: Operation timed out.",
    "hanging": "HangError: Process appears to be hanging.",
}


def find_error_lines(lines, error_key, start_line=0):
    """
    Search for the error keyword in the log line and return a list of line numbers
    Args:
        lines (list): All lines of the log file
        error_key (str): errors keyword
        start_line (int): The line number where the search began
    Returns:
        list: A list of errors line numbers
    """
    matches = []
    for i in range(start_line, len(lines)):
        if error_key.lower() in lines[i].lower():
            matches.append(i + 1)  # Line numbers start from 1
    return matches


def format_line_range(line_numbers):
    """
    Format the line number range display
    Args:
        line_numbers (list): List of line numbers
    Returns:
        str: Formatted line number range, such as "111-112" or "111"
    """
    if not line_numbers:
        return "unknown"
    elif len(line_numbers) == 1:
        return str(line_numbers[0])
    else:
        return f"{min(line_numbers)}-{max(line_numbers)}"


def generate_diagnostic_report(config, host, node_rank, log_file, return_content=False):
    """
    Generate an incremental diagnostic report from a log file.
    Args:
        config (DictConfig): Configuration object.
        host (str): Hostname or IP.
        node_rank (int): Node rank.
        log_file (str): Path to the log file.
        return_content (bool): If True, return report as string instead of writing to file.
    Returns:
        str: Diagnostic report content if return_content=True, else diagnostic file path.
    """
    global _diagnostic_offsets

    # Generate the path of the diagnostic file
    log_dir = os.path.dirname(log_file)
    diagnostic_file = os.path.join(log_dir, f"host_{node_rank}_{host}_diagnostic.txt")
    host_key = f"{host}_{node_rank}"

    try:
        if not os.path.exists(log_file) or os.path.getsize(log_file) == 0:
            logger.debug(f"Log file {log_file} is empty or does not exist")
            return diagnostic_file if not return_content else ""
        # Read all lines of the log file
        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
            all_lines = f.readlines()

        # Get the number of rows analyzed last time
        last_analyzed_line = _diagnostic_offsets.get(host_key, 0)

        # If it is the first analysis, create the header of the diagnostic file
        if last_analyzed_line == 0 and not os.path.exists(diagnostic_file):
            os.makedirs(os.path.dirname(diagnostic_file), exist_ok=True)
            header_content = f"Diagnostic Report for {host} (node {node_rank})\n"
            header_content += (
                f"Generated at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}\n"
            )
            header_content += "Analysis:\n"

            with open(diagnostic_file, 'w', encoding='utf-8') as f:
                f.write(header_content)

        # Only analyze the newly added rows
        new_lines = all_lines[last_analyzed_line:]
        if not new_lines:
            logger.debug(f"No new lines to analyze in {log_file}")
            return diagnostic_file if not return_content else ""

        # Analyze the errors in the new line
        current_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        new_errors = []
        for key, desc in error_types.items():
            # Look for errors in the newly added lines
            error_lines = find_error_lines(all_lines, key, last_analyzed_line)
            if error_lines:
                line_range = format_line_range(error_lines)
                log_filename = os.path.basename(log_file)
                error_entry = f"[{current_time}] {desc} Check {log_filename} line:{line_range}"
                new_errors.append(error_entry)

        # Update Analysis Location
        _diagnostic_offsets[host_key] = len(all_lines)

        # If there are new errors, append them to the diagnostic file
        if new_errors:
            os.makedirs(os.path.dirname(diagnostic_file), exist_ok=True)
            with open(diagnostic_file, 'a', encoding='utf-8') as f:
                for error in new_errors:
                    f.write(f"{error}\n")

            logger.debug(
                f"Added {len(new_errors)} new errors to diagnostic report for {host} (node {node_rank})"
            )

        if return_content:
            return '\n'.join(new_errors) if new_errors else ""
        else:
            return diagnostic_file
    except Exception as e:
        logger.error(f"Failed to generate diagnostic report for {host} (node {node_rank}): {e}")
        if return_content:
            return f"Error analyzing log file: {e}"
        else:
            return None
