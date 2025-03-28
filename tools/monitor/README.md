# Training Log Monitor

This monitoring tool is used to monitor training logs on remote servers, check for anomalies in the logs, send reminders via email or Feishu robot. This tool aims to ensure that problems during the training process can be identified in a timely manner, including timely monitoring of training jamming or slowing down.

# NOTE

For email reminders:
   1. This program requires a password to be entered during runtime, so please ensure that it is used in a secure environment to avoid the risk of password leakage.
   2. Training anomaly monitoring relies on historical training data analysis using various statistical methods. Please manually observe the logs for a period of time to ensure that at least the first 10 iterations are normal.

For Feishu robot reminders:
   1. This program requires a URL of Feishu Robot to be entered during runtime, so please ensure that it is used in a secure environment to avoid the risk of URL leakage.
   2. Training anomaly monitoring relies on historical training data analysis using various statistical methods. Please manually observe the logs for a period of time to ensure that at least the first 10 iterations are normal.

## Features

- Monitors a remote log file for training status.
- Sends corresponding abnormal information prompt emails based on log analysis results, including sample content for clarity.
- Configurable check interval.

## Prerequisites

Before running the script, ensure you have a password-free SSH login to the remote host.

## Installation

   ```bash
   git clone https://github.com/FlagOpen/FlagScale.git
   cd FlagScale/tools/monitor
   pip install -r requirements.txt
   ```

## Configuration

1. For Email:

   Modify the provided configuration file 'config-email.yaml' example to set actual values:

   ```yaml
   # Target email address for receiving alerts
   target_email: example_alert@domain.com  # The email address that will receive alerts

   # SMTP server setup for sending emails
   smtp_server: smtp.example.com  # The SMTP server used for sending emails

   # Email address used to send alerts
   source_email: example_sender@domain.com  # The email address to send alerts from

   # Remote host IP address for accessing log files
   remote_host: 192.0.2.1  # The IP address of the remote host where logs are stored

   # Username for SSH login to the remote host
   remote_user: example_user  # The username for SSH login

   # Port number for SSH access
   remote_port: 22  # Standard SSH port

   # Path to the log file on the remote host
   remote_log_path: /path/example_log_file.log  # Path to the log file

   # Interval in seconds for log checking
   check_interval: 1200  # Check logs every 1200 seconds
   ```

2. For Feishu Root

   Modify the provided configuration file 'config-feishu.yaml' example to set actual values:

   ```yaml
   # Remote host IP address for accessing log files
   remote_host: 192.0.2.1  # The IP address of the remote host where logs are stored

   # Username for SSH login to the remote host
   remote_user: example_user  # The username for SSH login

   # Port number for SSH access
   remote_port: 22  # Standard SSH port

   # Path to the log file on the remote host
   remote_log_path: /path/example_log_file.log  # Path to the log file

   # Interval in seconds for log checking
   check_interval: 1200  # Check logs every 1200 seconds
   ```


## Usage

1. For Email:

   ```bash
   python monitor.py --notice email
   ```

   You will then be prompted to enter your source email's password.

2. For Feishu robot:

   ```bash
   python monitor.py --notice feishu
   ```

   You will then be prompted to enter Feishu robot URL.

## Next steps

We will add monitoring perspectives, including:
- Prompt when training ends.
- Perform communication group-based monitoring.
- Monitor hardware utilization anomalies.
- More user needs...
