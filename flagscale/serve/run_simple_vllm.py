import os, sys
import logging
import yaml
import ray
import subprocess
import argparse


# os.environ["RAY_LOG_TO_STDERR"] = "1"
# os.environ["RAY_BACKEND_LOG_LEVEL"] = "info"  # 或 debug
os.environ["PYTHONUNBUFFERED"] = "1"
sys.stdout.reconfigure(line_buffering=True)
# root_logger = logging.getLogger()
# root_logger.setLevel(logging.DEBUG)
# handler = logging.StreamHandler(sys.stdout)
# handler.setLevel(logging.DEBUG)
# formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
# handler.setFormatter(formatter)
# root_logger.addHandler(handler)


# ray.init(log_to_driver=True, logging_level=logging.DEBUG, configure_logging=True)
ray.init(log_to_driver=True, logging_level=logging.INFO)


# logging.getLogger("vllm").setLevel(logging.DEBUG)
@ray.remote(num_gpus=1)
def start_vllm_serve(args):

    vllm_args = args["serve"]["vllm"]

    command = ["vllm", "serve"]
    command.append(vllm_args["model-tag"])
    for item in vllm_args:
        if item not in {"model-tag", "action-args"}:
            command.append(f"--{item}={vllm_args[item]}")
    for arg in vllm_args["action-args"]:
        command.append(f"--{arg}")

    # Start the subprocess

    print(f"Starting vllm serve with command: {' '.join(command)}")
    process = subprocess.Popen(command, stdout=sys.stdout, stderr=sys.stderr)
    # process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)

    stdout, stderr = process.communicate()

    print("Standard Output:")
    print(stdout)
    # print(stdout.decode())
    print("Standard Error:")
    print(stderr)
    # print(stderr.decode())

    return process.returncode


def main():
    parser = argparse.ArgumentParser(description="Start vllm serve with Ray")

    parser.add_argument(
        "--config-path", type=str, required=True, help="Path to the model"
    )

    args = parser.parse_args()
    with open(args.config_path, "r") as file:
        config = yaml.safe_load(file)
    result = start_vllm_serve.remote(config)

    return_code = ray.get(result)

    print(f"vllm serve exited with return code: {return_code}")


if __name__ == "__main__":
    main()
