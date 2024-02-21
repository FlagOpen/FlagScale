import os
import sys
import time
import json
import random
import argparse
import torch
from tqdm import tqdm

pardir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(os.path.join(pardir, "vllm"))

from vllm import LLM, SamplingParams


def sampling_requests(data_path, tokenizer, num_requests, generate_len):
    requests = []
    with open(data_path) as f:
        for line in f:
            data = json.loads(line)

            prompt = data["question"]
            completion = data["answer"]

            input_len = len(tokenizer(prompt).input_ids)
            output_len = len(tokenizer(completion).input_ids)
            if generate_len is not None:
                output_len = generate_len
            requests.append((prompt, input_len, output_len))

    return random.sample(requests, num_requests)


def add_text_generate_args():
    parser = argparse.ArgumentParser(description="Benchmark vllm text generation")
    parser.add_argument("--num-iters", type=int, default=1,
                       help='Number of iters to run generation for a single batch request.')
    parser.add_argument("--temperature", type=float, default=1.0,
                       help='Sampling temperature.')
    parser.add_argument("--top-p", type=float, default=0.0,
                       help='Top p sampling.')
    parser.add_argument("--top-k", type=int, default=0,
                       help='Top k sampling.')
    parser.add_argument("--prompt-len", type=int, default=None,
                       help='Length of each prompt')
    parser.add_argument("--generate-len", type=int, default=None,
                       help='The maximum numbers of tokens to generate.')
    parser.add_argument("--dataset-path", type=str, default=None,
                       help='Path to the requests data.')

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model", type=str, default="BAAI/AquilaChat2-7B")
    # parser.add_argument("--tokenizer", type=str, default=None)
    parser.add_argument("--tensor-parallel-size", "-tp", type=int, default=1)
    parser.add_argument('--pipeline-parallel-size', '-pp', type=int, default=1)

    return parser.parse_args()


if __name__ == "__main__":

    args = add_text_generate_args()
    print(args)
    if args.dataset_path is None:
        assert args.prompt_len is not None
        assert args.generate_len is not None
    else:
        assert args.prompt_len is None

    random.seed(args.seed)
    llm = LLM(
        model=args.model,
        # tokenizer=args.tokenizer,
        tensor_parallel_size=args.tensor_parallel_size,
        pipeline_parallel_size=args.pipeline_parallel_size,
        seed=args.seed,
        trust_remote_code=True,
        disable_log_stats=False,
    )
    tokenizer = llm.get_tokenizer()

    if args.dataset_path is not None and os.path.exists(args.dataset_path):
        print(f"loading data from {args.dataset_path} ...")
        print("'prompt_len' and 'generate_len' will be rewritten by real data")
        requests = sampling_requests(args.dataset_path, tokenizer, 1, args.generate_len)
    else:
        print("making fake data ...")
        prompt = "我" * args.prompt_len
        input_len = len(tokenizer.tokenize(prompt))
        assert input_len == args.prompt_len
        requests = [(prompt, args.prompt_len, args.generate_len)]

    prompt, input_len, output_len = requests[0]
    # Add the requests to the engine.
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        ignore_eos=True,
        max_tokens=output_len,
    )
    llm._add_request(
        prompt=prompt,
        prompt_token_ids=None,
        sampling_params=sampling_params,
    )

    print("Warming up...")
    llm.generate(prompt, sampling_params)

    print("Benchmark....")
    start = time.perf_counter()
    pbar = tqdm(total=args.num_iters, desc="Processed iterations")
    for i in range(args.num_iters):
        outputs = llm._run_engine(use_tqdm=False)
        pbar.update(i)
    pbar.close()
    end = time.perf_counter()

    # Print the outputs.
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

    num_totol_tokens = sum([il + ol for _, il, ol in requests])
    memory_used = torch.cuda.max_memory_allocated() / (1024 ** 3)
    memory_pct = memory_used / (torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory / (1024 ** 3)) * 100

    print("------------ SUMMARY ------------")
    print(f"Num Request: {len(requests)}")
    print(f"Num totol tokens: {num_totol_tokens}")
    print(f"Batch Size: {1}")
    print(f"Elapsed time: {end - start:.8f} seconds")
    print(f"Avg Latency: {(end - start) / args.num_iters:.8f} seconds")
    print(f"Max Memory: {memory_used:.2f} GB ({memory_pct:.2f}%)")
    print("---------------------------------")
