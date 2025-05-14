import os
import subprocess
import sys

from flagscale.inference.arguments import parse_config
from flagscale.runner.utils import flatten_dict_to_args


def inference(cfg):
    merged_args = cfg.llm
    merged_args.update(cfg.generate)
    command = ["./third_party/llama.cpp/build/bin/llama-cli"]
    command.extend(flatten_dict_to_args(merged_args))

    print(f"[Inference]: Starting vllm serve with command: {' '.join(command)}")

    process = subprocess.Popen(command, stdout=sys.stdout, stderr=sys.stderr)
    pid = os.getpid()
    print(f"[Inference]: Current vLLM PID: {pid} ")

    stdout, stderr = process.communicate()
    print(f"[Inference]: Standard Output: {stdout}")
    print(f"[Inference]: Standard Error: {stderr}")


if __name__ == "__main__":
    cfg = parse_config()
    inference(cfg)
