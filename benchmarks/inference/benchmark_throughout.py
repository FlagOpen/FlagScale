import os
import sys
import torch
import json
import random
import time
from tqdm import tqdm

pardir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(os.path.join(pardir, "megatron"))

from megatron import get_tokenizer
from megatron import get_args
from megatron import get_timers
from megatron import print_rank_0
from megatron.checkpointing import load_checkpoint
from megatron.initialize import initialize_megatron
from megatron.model import GPTModel
from megatron.training import get_model
from megatron.arguments import core_transformer_config_from_args
from megatron.text_generation import generate_and_post_process
from megatron.theoretical_memory_usage import compute_weight_and_optimizer_memory


def sampling_requests(data_path, tokenizer, num_requests):
    requests = []
    with open(data_path) as f:
        for line in f:
            data = json.loads(line)

            prompt = data["question"]
            completion = data["answer"]

            input_len = len(tokenizer.tokenize(prompt))
            output_len = len(tokenizer.tokenize(completion))
            requests.append((prompt, input_len, output_len))

    return random.sample(requests, num_requests)


def model_provider(pre_process=True, post_process=True):
    """Build the model."""

    config = core_transformer_config_from_args(get_args())

    print_rank_0('building GPT model ...')
    model = GPTModel(
        config,
        num_tokentypes=0,
        parallel_output=False,
        pre_process=pre_process,
        post_process=post_process
    )
    compute_weight_and_optimizer_memory(get_args(), verbose=True)
    return model

def add_text_generate_args(parser):
    group = parser.add_argument_group(title='Benchmark text generation')

    group.add_argument("--num-requests", type=int, default=1,
                       help='Number of requests to process.')
    group.add_argument("--temperature", type=float, default=1.0,
                       help='Sampling temperature.')
    group.add_argument("--top-p", type=float, default=0.0,
                       help='Top p sampling.')
    group.add_argument("--top-k", type=int, default=0,
                       help='Top k sampling.')
    group.add_argument("--prompt-len", type=int, default=32,
                       help='Length of each prompt')
    group.add_argument("--generate-len", type=int, default=1024,
                       help='The maximum numbers of tokens to generate.')
    group.add_argument("--dataset-path", type=str, default=None,
                       help='Path to the requests data.')

    return parser

if __name__=="__main__":
    initialize_megatron(extra_args_provider=add_text_generate_args,
                        args_defaults={'tokenizer_type': 'AquilaTokenizer',
                                       'no_load_rng': True,
                                       'no_load_optim': True})
    args = get_args()
    timers = get_timers()
    print(f"args is {args}")
    if args.num_layers_per_virtual_pipeline_stage is not None:
        print("Interleaved pipeline schedule is not yet supported for text generation.")
        exit()
    print_rank_0("WARNING: Forcing exit_on_missing_checkpoint to True for text "
                 "generation.")
    args.exit_on_missing_checkpoint = True

    # Set up model and load checkpoint
    model = get_model(model_provider, wrap_with_ddp=False)
    print(f"get model successfully....")

    if args.load is not None:
        _ = load_checkpoint(model, None, None)
    print("loading model successfully....")

    assert len(model) == 1, "Above condition should have caught this"
    model = model[0]

    tokenizer = get_tokenizer()

    # validate args
    assert args.num_requests % args.micro_batch_size == 0
    if args.dataset_path is None:
        assert args.prompt_len is not None
        assert args.generate_len is not None
    else:
        assert args.prompt_len is None

    # prepare requests
    if args.dataset_path is not None and os.path.exists(args.dataset_path):
        print_rank_0(f"loading data from {args.dataset_path} ...")
        print_rank_0("'prompt_len' and 'generate_len' will be rewritten by real data")
        requests = sampling_requests(args.dataset_path, tokenizer, args.num_requests)
    else:
        print_rank_0("making fake data ...")
        prompt = "我" * args.prompt_len
        input_len = len(tokenizer.tokenize(prompt))
        assert input_len == args.prompt_len
        requests = [(prompt, args.prompt_len, args.generate_len)] * args.num_requests

    print("warming up....")
    prompt, input_len, output_len = requests[0]
    generate_and_post_process(
            model,
            prompts=[prompt],
            tokens_to_generate=output_len,
            return_output_log_probs=True,
            top_k_sampling=args.top_k,
            top_p_sampling=args.top_p,
            top_p_decay=0.0,
            top_p_bound=0.0,
            temperature=args.temperature,
            add_BOS=False,
            use_eod_token_for_early_termination=True,
            stop_on_double_eol=False,
            stop_on_eol=False,
            prevent_newline_after_colon=False,
            random_seed=args.seed,
    )

    print("Benchmark....")
    start = time.perf_counter()
    pbar = tqdm(total=args.num_requests, desc="Processed requests")
    prompts = []
    results = []
    max_output_len = 0
    for prompt, _, output_len in requests:
        prompts.append(prompt)
        max_output_len = max(output_len, max_output_len)

        if len(prompts) < args.micro_batch_size:
            continue

        responses, _, _, _ = generate_and_post_process(
            model,
            prompts=prompts,
            tokens_to_generate=max_output_len,
            return_output_log_probs=True,
            top_k_sampling=args.top_k,
            top_p_sampling=args.top_p,
            top_p_decay=0.0,
            top_p_bound=0.0,
            temperature=args.temperature,
            add_BOS=False,
            use_eod_token_for_early_termination=True,
            stop_on_double_eol=False,
            stop_on_eol=False,
            prevent_newline_after_colon=False,
            random_seed=args.seed,
        )
        for i in range(len(prompts)):
            results.append([prompts[i], responses[i]])

        pbar.update(len(prompts))
        prompts = []
        max_output_len = 0

    pbar.close()
    end = time.perf_counter()

    for res in results:
        prompt = res[0]
        generated_text = res[1][len(prompt):]
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

    num_totol_tokens = sum([il + ol for _, il, ol in requests])
    memory_used = torch.cuda.max_memory_allocated() / (1024 ** 3)
    memory_pct = memory_used / (torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory / (1024 ** 3)) * 100

    print("------------ SUMMARY ------------")
    print(f"Num Request: {len(requests)}")
    print(f"Num totol tokens: {num_totol_tokens}")
    print(f"Batch Size: {args.micro_batch_size}")
    print(f"Elapsed time: {end - start}")
    print(f"Throughput: {args.num_requests / (end - start):.2f} requests/s, {num_totol_tokens / (end - start):.2f} tokens/s")
    print(f"Max Memory: {memory_used:.2f} GB ({memory_pct:.2f}%)")
    print("---------------------------------")
