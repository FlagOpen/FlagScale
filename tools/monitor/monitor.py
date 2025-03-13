import datetime
import json
import re
import smtplib
import sys
import time
from email.mime.text import MIMEText
from getpass import getpass

import paramiko
import yaml


# Function to read configuration from a YAML file
def read_config(config_file):
    try:
        with open(config_file, "r") as file:
            # Parse YAML file
            config = yaml.safe_load(file)
        print("Loaded configuration:")
        print(yaml.dump(config))  # Output the loaded configuration
        return config
    except FileNotFoundError:
        print(f"{datetime.datetime.now()} - Configuration file not found.")
        print("Error: Configuration file not found.")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"{datetime.datetime.now()} - Error parsing configuration file: {e}")
        print(f"Error: Error parsing configuration file: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"{datetime.datetime.now()} - Error reading configuration file: {e}")
        print(f"Error: Error reading configuration file: {e}")
        sys.exit(1)


# Function to connect to remote host and read log file
def read_logs(remote_host, remote_user, remote_port, remote_log_path):
    print(f"{datetime.datetime.now()} - Connecting to remote host...")
    try:
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(hostname=remote_host, username=remote_user, port=remote_port)

        sftp = ssh.open_sftp()
        with sftp.open(remote_log_path, "r") as remote_file:
            print("Loading log file content...")
            contents = remote_file.readlines()
            print("Loading finish")

        sftp.close()
        ssh.close()

        return contents
    except Exception as e:
        print(f"{datetime.datetime.now()} - Error reading remote log file: {e}")
        print(f"Error: Error reading remote log file: {e}")
        sys.exit(1)


# Function to parse log file contents
def parse_logs(log_lines):
    print(f"{datetime.datetime.now()} - Parsing log content...")

    pattern = (
        r"\[+(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]+\s*"
        r"iteration\s+(?P<iteration>\d+\s*/\s*\d+)\s+\|\s*"
        r"consumed samples:\s+(?P<consumed_samples>\d+)\s+\|\s*"
        r"elapsed time per iteration \(ms\):\s+(?P<elapsed_time_per_iteration_ms>\d+\.\d+)\s+\|\s*"
        r"throughput per GPU \(TFLOP/s/GPU\):\s+(?P<throughput_per_GPU_TFLOPs_per_GPU>\d+\.\d+)\s+\|\s*"
        r"learning rate:\s+(?P<learning_rate>\d+\.?\d*E?-?\d*)\s+\|\s*"
        r"global batch size:\s+(?P<global_batch_size>\d+)\s+\|\s*"
        r"lm loss:\s+(?P<lm_loss>\d+\.\d+E?[-\+]?\d*)\s+\|\s*"
        r"load_balancing_loss:\s+(?P<load_balancing_loss>\d+\.\d+E?[-\+]?\d*)\s+\|\s*"
        r"loss scale:\s+(?P<loss_scale>\d+\.\d+)\s+\|\s*"
        r"grad norm:\s+(?P<grad_norm>\d+\.\d+)\s+\|\s*"
        r"num zeros:\s+(?P<num_zeros>[\d.]+)\s+\|\s*"
        r"params norm:\s+(?P<params_norm>\d+\.\d+)\s+\|\s*"
        r"number of skipped iterations:\s+(?P<number_of_skipped_iterations>\d+)\s+\|\s*"
        r"number of nan iterations:\s+(?P<number_of_nan_iterations>\d+)\s*"
    )

    lines_to_keep = []
    for line in log_lines:
        match = re.search(pattern, line)
        if match:
            parsed_data = match.groupdict()
            if parsed_data["iteration"].startswith("1/"):
                lines_to_keep = []
            lines_to_keep.append(parsed_data)
    return lines_to_keep


# Function to check training status based on parsed logs
def check_logs(logs):
    print(f"{datetime.datetime.now()} - Checking log content...")
    recent_logs = logs[-10:]
    last_log = logs[-1]
    last_timestamp = datetime.datetime.strptime(
        last_log["timestamp"], "%Y-%m-%d %H:%M:%S"
    )
    last_elapsed = float(last_log["elapsed_time_per_iteration_ms"])
    now = datetime.datetime.now()
    time_delta_ms = (now - last_timestamp).total_seconds() * 1000

    if time_delta_ms > 3 * last_elapsed:
        status_code = 1
        message = "Training appears to be stuck."
    elif historical_elapsed := [
        float(entry["elapsed_time_per_iteration_ms"]) for entry in logs[:-1]
    ]:
        average_elapsed = sum(historical_elapsed) / len(historical_elapsed)
        if last_elapsed > 2 * average_elapsed:
            status_code = 2
            message = "Training appears to be slowing down."
        else:
            status_code = 0
            message = "Training is normal."
    else:
        status_code = 0
        message = "Training is normal."

    return status_code, message, recent_logs


# Function to send email alerts
def send_email(
    smtp_server, source_email, source_email_password, target_email, subject, body
):
    print(f"{datetime.datetime.now()} - Sending email...")
    try:
        msg = MIMEText(body)
        msg["Subject"] = subject
        msg["From"] = source_email
        msg["To"] = target_email

        server = smtplib.SMTP(smtp_server, 25)
        server.login(source_email, source_email_password)
        server.sendmail(source_email, [target_email], msg.as_string())
        server.quit()
        print(f"{datetime.datetime.now()} - Sending email finish.")
    except Exception as e:
        print(f"{datetime.datetime.now()} - Error sending email: {e}")
        print(f"Error: Error sending email: {e}")


# Main function
def main():
    config_file = "config.yaml"
    config = read_config(config_file)

    remote_host = config["remote_host"]
    remote_user = config["remote_user"]
    remote_port = int(config["remote_port"])
    remote_log_path = config["remote_log_path"]
    target_email = config["target_email"]
    smtp_server = config["smtp_server"]
    source_email = config["source_email"]
    check_interval = int(config["check_interval"])

    # Securely get the password without displaying it
    source_email_password = getpass(f"Enter the password for {source_email}: ")

    while True:
        logs = read_logs(remote_host, remote_user, remote_port, remote_log_path)
        useful_logs = parse_logs(logs)
        status_code, message, recent_logs = check_logs(useful_logs)
        recent_logs = json.dumps(recent_logs, indent=4)

        if status_code == 1:
            send_email(
                smtp_server,
                source_email,
                source_email_password,
                target_email,
                message,
                recent_logs,
            )
        elif status_code == 2:
            send_email(
                smtp_server,
                source_email,
                source_email_password,
                target_email,
                message,
                recent_logs,
            )

        print("Waiting for next checking...")
        time.sleep(check_interval)


if __name__ == "__main__":
    main()
