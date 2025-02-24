import asyncio
import collections
import json
import os
import re
import socket
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass, field
from typing import List, Optional

import aiohttp
import numpy as np
from omegaconf import DictConfig, OmegaConf
from tqdm.asyncio import tqdm

from flagscale.logger import logger
from flagscale.metric import calculate_metrics

AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=6 * 60 * 60)


def log_and_raise_error(message):
    logger.error(message)
    raise ValueError(message)


def parse_hostfile(hostfile_path):
    if hostfile_path is None or not os.path.isfile(hostfile_path):
        logger.warning(
            f"Hostfile {hostfile_path} not found. The training will proceed using only local resources."
        )
        return None

    # e.g., worker0 slots=8 type=A100
    pattern = re.compile(r"^(\S+)\s+slots=(\d+)(?:\s+type=(\S+))?")

    resources = collections.OrderedDict()

    with open(hostfile_path, "r") as fd:
        hostfile_lines = fd.readlines()

    for line in hostfile_lines:
        line = line.strip()
        match = pattern.search(line)
        if line.startswith("#") or line == "":
            # hostfile comment or empty line, ignore
            continue
        elif match:
            host = match.group(1)
            num_slots = int(match.group(2))
            machine_type = match.group(3) if match.group(3) else None
            if host in resources:
                log_and_raise_error(
                    f"Hostfile contains multiple entries for host: {host}."
                )
            resources[host] = {"slots": num_slots, "type": machine_type}
        else:
            log_and_raise_error(f"Invalid entry in hostfile: {line}.")

    assert all(info["type"] == None for _, info in resources.items()) or all(
        info["type"] != None for _, info in resources.items()
    ), "All hosts must have the a machine type or no machine type specified."

    if len(resources) == 0:
        log_and_raise_error(
            "Hostfile is empty or not formatted correctly. Please check the hostfile."
        )

    return resources


def get_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def get_host_name_or_ip():
    host_name = socket.gethostname()
    if host_name:
        return host_name
    try:
        # doesn't even have to be reachable
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.connect(("10.255.255.255", 1))
        IP = sock.getsockname()[0]
    except Exception:
        IP = "127.0.0.1"
    finally:
        if (
            "sock" in locals()
        ):  # Ensure 'sock' was successfully created before attempting to close it
            sock.close()
    return IP


def run_local_command(cmd, dryrun=False, query=False):
    logger.info(f"Run the local command: {cmd}")
    if dryrun:
        return
    if query:
        result = subprocess.run(
            cmd,
            shell=True,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        return result
    else:
        result = subprocess.run(
            cmd,
            shell=True,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if result.returncode != 0:
            print(f"Command {cmd} failed with return code {result.returncode}.")
            print(f"Output: {result.stdout}")
            print(f"Error: {result.stderr}")
            sys.exit(result.returncode)


def run_ssh_command(host, cmd, port=None, dryrun=False, query=False):
    if port:
        ssh_cmd = f"ssh -f -n -p {port} {host} '{cmd}'"
    else:
        ssh_cmd = f"ssh -f -n {host} '{cmd}'"
    if not query:
        logger.info(f"Running the ssh command: {ssh_cmd}")
    if dryrun:
        return
    result = subprocess.run(
        ssh_cmd,
        shell=True,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode != 0:
        print(f"SSH command {ssh_cmd} failed with return code {result.returncode}.")
        print(f"Output: {result.stdout}")
        print(f"Error: {result.stderr}")
        sys.exit(result.returncode)
    if query:
        return result


def run_scp_command(host, src, dst, port=None, dryrun=False):
    if port:
        scp_cmd = f"scp -P {port} -r {src} {host}:{dst} "
    else:
        scp_cmd = f"scp -r {src} {host}:{dst} "
    logger.info(f"Run the scp command: {scp_cmd}")
    if dryrun:
        return
    result = subprocess.run(
        scp_cmd,
        shell=True,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode != 0:
        print(f"SCP command {scp_cmd} failed with return code {result.returncode}.")
        print(f"Output: {result.stdout}")
        print(f"Error: {result.stderr}")
        sys.exit(result.returncode)


def flatten_dict_to_args(config_dict, ignore_keys=[]):
    args = []
    for key, value in config_dict.items():
        if key in ignore_keys:
            continue
        key = key.replace("_", "-")
        if isinstance(value, dict):
            args.extend(flatten_dict_to_args(value, ignore_keys))
        elif isinstance(value, list):
            args.append(f"--{key}")
            for v in value:
                args.append(f"{v}")
        elif isinstance(value, bool):
            if value:
                args.append(f"--{key}")
        else:
            args.append(f"--{key}")
            args.append(f"{value}")
    return args


def get_nnodes(nnodes_from_hostfile=None, nnodes_from_args=None):
    assert nnodes_from_hostfile is not None or nnodes_from_args is not None
    if nnodes_from_hostfile is not None and nnodes_from_args is not None:
        if isinstance(nnodes_from_args, str) and ":" in nnodes_from_args:
            # Ignore the max nnodes from the args, no elastic support
            nnodes_from_args, _ = nnodes_from_args.split(":")
        return min(nnodes_from_hostfile, int(nnodes_from_args))
    elif nnodes_from_hostfile is not None:
        return nnodes_from_hostfile
    elif nnodes_from_args is not None:
        if isinstance(nnodes_from_args, str) and ":" in nnodes_from_args:
            # Ignore the max nnodes from the args, no elastic support
            nnodes_from_args, _ = nnodes_from_args.split(":")
        return int(nnodes_from_args)


def get_nproc_per_node(
    nproc_from_hostfile=None, nproc_from_args=None, num_visible_devices=None
):
    if nproc_from_hostfile is not None and nproc_from_args is not None:
        nproc = min(nproc_from_hostfile, int(nproc_from_args))
        if num_visible_devices:
            return min(nproc, num_visible_devices)
        else:
            return nproc
    elif nproc_from_hostfile is not None:
        if num_visible_devices:
            return min(nproc_from_hostfile, num_visible_devices)
        else:
            return nproc_from_hostfile
    elif nproc_from_args is not None:
        if num_visible_devices:
            return min(int(nproc_from_args), num_visible_devices)
        else:
            return nproc_from_args
    else:
        if num_visible_devices:
            return num_visible_devices
        else:
            return 1


def add_decive_extra_config(config, device_type):
    if device_type is None:
        logger.warning(
            f"type in hostfile is not specified. All the nodes use the same arguments inlucding evnironment variables."
        )
        return OmegaConf.to_container(config, resolve=True)
    cur_node_config = {}
    temp_dict = {}
    if isinstance(config, DictConfig):
        temp_dict = OmegaConf.to_container(config, resolve=True)
    else:
        temp_dict = config
    for key, value in temp_dict.items():
        if isinstance(value, dict):
            if key == device_type:
                cur_node_config.update(value)
            else:
                continue
        else:
            cur_node_config[key] = value
    return cur_node_config


def is_ip_addr(master):
    """Check if master is ip address."""

    if not isinstance(master, str):
        return False
    pattern = (
        r"^((25[0-5]|2[0-4]\d|1\d{2}|[1-9]?\d)\.){3}(25[0-5]|2[0-4]\d|1\d{2}|[1-9]?\d)$"
    )
    result = re.match(pattern, master)
    if result:
        return True
    else:
        return False


def get_ip_addr():
    """Get ip address."""
    try:
        hostname = socket.gethostname()
        ip = socket.gethostbyname(socket.getfqdn(hostname))
    except:
        ip = "127.0.0.1"
    return ip


def is_master(config):
    """Check if current node is master."""
    nnodes = config.experiment.runner.get("nnodes", 1)

    hostfile = None
    if config.experiment.runner.get("hostfile", None):
        hostfile = config.experiment.runner["hostfile"]
    if os.environ.get("AIRS_SWITCH", None):
        if os.environ.get("AIRS_HOSTFILE_PATH", None):
            hostfile = os.environ["AIRS_HOSTFILE_PATH"]

    resources = parse_hostfile(hostfile)
    if not resources and nnodes > 1:
        raise ValueError("In the multi-node mode, please set the hostfile")

    if resources:
        master = list(resources.keys())[0]
        if is_ip_addr(master):
            return get_ip_addr() == master
        else:
            output = subprocess.run(
                "hostname", check=True, shell=True, text=True, capture_output=True
            )
            hostname = output.stdout.strip()
            return hostname == master
    # Local host Scene
    return True


@dataclass
class RequestFuncInput:
    prompt: str
    api_url: str
    prompt_len: int
    output_len: int
    model: str
    model_name: Optional[str] = None
    best_of: int = 1
    logprobs: Optional[int] = None
    extra_body: Optional[dict] = None
    multi_modal_content: Optional[dict] = None
    ignore_eos: bool = False


@dataclass
class RequestFuncOutput:
    generated_text: str = ""
    success: bool = False
    latency: float = 0.0
    output_tokens: int = 0
    ttft: float = 0.0  # Time to first token
    itl: List[float] = field(default_factory=list)  # List of inter-token latencies
    tpot: float = 0.0  # avg next-token latencies
    prompt_len: int = 0
    error: str = ""


def dummy_random_input(
    tokenizer,
    prefix_len=0,
    input_len=1024,
    output_len=128,
    num_prompts=1000,
    range_ratio=1.0,
):
    prefix_token_ids = np.random.randint(
        0, tokenizer.vocab_size, size=prefix_len
    ).tolist()

    input_lens = np.random.randint(
        int(input_len * range_ratio),
        input_len + 1,
        size=num_prompts,
    )
    output_lens = np.random.randint(
        int(output_len * range_ratio),
        output_len + 1,
        size=num_prompts,
    )
    offsets = np.random.randint(0, tokenizer.vocab_size, size=num_prompts)
    input_requests = []
    for i in range(num_prompts):
        prompt = tokenizer.decode(
            prefix_token_ids
            + [
                (offsets[i] + i + j) % tokenizer.vocab_size
                for j in range(input_lens[i])
            ]
        )

        input_requests.append(
            (prompt, int(prefix_len + input_lens[i]), int(output_lens[i]), None)
        )

    return input_requests


async def async_request_openai_completions(
    request_func_input,
    pbar=None,
):
    api_url = request_func_input.api_url
    assert api_url.endswith(
        ("completions", "profile")
    ), "OpenAI Completions API URL must end with 'completions' or 'profile'."

    async with aiohttp.ClientSession(
        trust_env=True, timeout=AIOHTTP_TIMEOUT
    ) as session:
        payload = {
            "model": (
                request_func_input.model_name
                if request_func_input.model_name
                else request_func_input.model
            ),
            "prompt": request_func_input.prompt,
            "temperature": 0.0,
            "best_of": request_func_input.best_of,
            "max_tokens": request_func_input.output_len,
            "logprobs": request_func_input.logprobs,
            "stream": True,
            "stream_options": {
                "include_usage": True,
            },
        }
        if request_func_input.ignore_eos:
            payload["ignore_eos"] = request_func_input.ignore_eos
        if request_func_input.extra_body:
            payload.update(request_func_input.extra_body)
        headers = {"Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}"}

        output = RequestFuncOutput()
        output.prompt_len = request_func_input.prompt_len

        generated_text = ""
        st = time.perf_counter()
        most_recent_timestamp = st
        try:
            async with session.post(
                url=api_url, json=payload, headers=headers
            ) as response:
                if response.status == 200:
                    first_chunk_received = False
                    async for chunk_bytes in response.content:
                        chunk_bytes = chunk_bytes.strip()
                        if not chunk_bytes:
                            continue

                        chunk = chunk_bytes.decode("utf-8").removeprefix("data: ")
                        if chunk != "[DONE]":
                            data = json.loads(chunk)

                            # NOTE: Some completion API might have a last
                            # usage summary response without a token so we
                            # want to check a token was generated
                            if choices := data.get("choices"):
                                # Note that text could be empty here
                                # e.g. for special tokens
                                text = choices[0].get("text")
                                timestamp = time.perf_counter()
                                # First token
                                if not first_chunk_received:
                                    first_chunk_received = True
                                    ttft = time.perf_counter() - st
                                    output.ttft = ttft

                                # Decoding phase
                                else:
                                    output.itl.append(timestamp - most_recent_timestamp)

                                most_recent_timestamp = timestamp
                                generated_text += text or ""
                            elif usage := data.get("usage"):
                                output.output_tokens = usage.get("completion_tokens")
                    if first_chunk_received:
                        output.success = True
                    else:
                        output.success = False
                        output.error = (
                            "Never received a valid chunk to calculate TTFT."
                            "This response will be marked as failed!"
                        )
                    output.generated_text = generated_text
                    output.latency = most_recent_timestamp - st
                else:
                    output.error = response.reason or ""
                    output.success = False
        except Exception:
            output.success = False
            exc_info = sys.exc_info()
            output.error = "".join(traceback.format_exception(*exc_info))

    if pbar:
        pbar.update(1)
    return output


async def get_request(
    input_requests,
):
    input_requests = iter(input_requests)
    # Calculate scale parameter theta to maintain the desired request_rate.
    for request in input_requests:
        yield request


async def benchmark(
    api_url,
    model,
    tokenizer,
    input_requests,
    selected_percentile_metrics,
    selected_percentiles,
):

    async def limited_request_func(request_func_input, pbar):
        return await request_func(request_func_input=request_func_input, pbar=pbar)

    request_func = async_request_openai_completions
    req_model_id = req_model_name = model
    pbar = tqdm(total=len(input_requests))

    benchmark_start_time = time.perf_counter()
    tasks = []
    async for request in get_request(input_requests):
        prompt, prompt_len, output_len, mm_content = request

        request_func_input = RequestFuncInput(
            model=req_model_id,
            model_name=req_model_name,
            prompt=prompt,
            api_url=api_url,
            prompt_len=prompt_len,
            output_len=output_len,
            multi_modal_content=mm_content,
        )
        tasks.append(
            asyncio.create_task(
                limited_request_func(request_func_input=request_func_input, pbar=pbar)
            )
        )
    outputs = await asyncio.gather(*tasks)
    pbar.close()

    benchmark_duration = time.perf_counter() - benchmark_start_time

    metrics, actual_output_lens = calculate_metrics(
        input_requests=input_requests,
        outputs=outputs,
        dur_s=benchmark_duration,
        tokenizer=tokenizer,
        selected_percentile_metrics=selected_percentile_metrics,
        selected_percentiles=selected_percentiles,
    )

    print("{s:{c}^{n}}".format(s=" Serving Benchmark Result ", n=50, c="="))
    print("{:<40} {:<10}".format("Successful requests:", metrics.completed))
    print("{:<40} {:<10.2f}".format("Benchmark duration (s):", benchmark_duration))
    print("{:<40} {:<10}".format("Total input tokens:", metrics.total_input))
    print("{:<40} {:<10}".format("Total generated tokens:", metrics.total_output))
    print(
        "{:<40} {:<10.2f}".format(
            "Request throughput (req/s):", metrics.request_throughput
        )
    )
    print(
        "{:<40} {:<10.2f}".format(
            "Output token throughput (tok/s):", metrics.output_throughput
        )
    )
    print(
        "{:<40} {:<10.2f}".format(
            "Total Token throughput (tok/s):", metrics.total_token_throughput
        )
    )

    result = {
        "duration": benchmark_duration,
        "completed": metrics.completed,
        "total_input_tokens": metrics.total_input,
        "total_output_tokens": metrics.total_output,
        "request_throughput": metrics.request_throughput,
        "output_throughput": metrics.output_throughput,
        "total_token_throughput": metrics.total_token_throughput,
    }

    def process_one_metric(
        # E.g., "ttft"
        metric_attribute_name: str,
        # E.g., "TTFT"
        metric_name: str,
        # E.g., "Time to First Token"
        metric_header: str,
    ):
        # This function prints and adds statistics of the specified
        # metric.
        if metric_attribute_name not in selected_percentile_metrics:
            return
        print("{s:{c}^{n}}".format(s=metric_header, n=50, c="-"))
        print(
            "{:<40} {:<10.2f}".format(
                f"Mean {metric_name} (ms):",
                getattr(metrics, f"mean_{metric_attribute_name}_ms"),
            )
        )
        print(
            "{:<40} {:<10.2f}".format(
                f"Median {metric_name} (ms):",
                getattr(metrics, f"median_{metric_attribute_name}_ms"),
            )
        )
        result[f"mean_{metric_attribute_name}_ms"] = getattr(
            metrics, f"mean_{metric_attribute_name}_ms"
        )
        result[f"median_{metric_attribute_name}_ms"] = getattr(
            metrics, f"median_{metric_attribute_name}_ms"
        )
        result[f"std_{metric_attribute_name}_ms"] = getattr(
            metrics, f"std_{metric_attribute_name}_ms"
        )
        for p, value in getattr(metrics, f"percentiles_{metric_attribute_name}_ms"):
            p_word = str(int(p)) if int(p) == p else str(p)
            print("{:<40} {:<10.2f}".format(f"P{p_word} {metric_name} (ms):", value))
            result[f"p{p_word}_{metric_attribute_name}_ms"] = value

    process_one_metric("ttft", "TTFT", "Time to First Token")
    process_one_metric("tpot", "TPOT", "Time per Output Token (excl. 1st token)")
    process_one_metric("itl", "ITL", "Inter-token Latency")
    process_one_metric("e2el", "E2EL", "End-to-end Latency")

    print("=" * 50)

    return result
