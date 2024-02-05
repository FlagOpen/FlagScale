import os
import sys
import torch
from test_throughout import sampling_requests

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

    return model

def add_text_generate_args(parser):
    group = parser.add_argument_group(title='Benchmark latency for a single batch request')

    group.add_argument("--num-iters", type=int, default=1,
                       help='Number of iters to run generation for a single batch request.')
    group.add_argument("--temperature", type=float, default=1.0,
                       help='Sampling temperature.')
    group.add_argument("--top_p", type=float, default=0.0,
                       help='Top p sampling.')
    group.add_argument("--top_k", type=int, default=0,
                       help='Top k sampling.')
    group.add_argument("--prompt-len", type=int, default=32,
                       help='Length of prompt for each request.')
    group.add_argument("--generate-len", type=int, default=1024,
                       help='Length of generated text for each request.')
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

    # prepare requests
    if os.path.exists(args.dataset_path) and os.path.basename(args.dataset_path) == 'test.jsonl':
        print_rank_0(f"loading data from {args.dataset_path} ...")
        print_rank_0("'prompt_len' and 'generate_len' will be rewritten by real data")
        requests = sampling_requests(args.dataset_path, tokenizer, args.micro_batch_size)
    else:
        print_rank_0("making fake data ...")
        prompt = "我" * args.prompt_len
        input_len = len(tokenizer.tokenize(prompt))
        assert input_len == args.prompt_len
        requests = [(prompt, args.prompt_len, args.generate_len)] * args.micro_batch_size

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

    prompts = []
    max_output_len = 0
    for prompt, _, output_len in requests:
        prompts.append(prompt)
        max_output_len = max(output_len, max_output_len)

    print("Benchmark....")
    timers("all_iters", log_level=0).start(barrier=True)
    for i in range(args.num_iters):
        respose, _, _, _ = generate_and_post_process(
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
        print(f"iter-{i}/{args.num_iters}")

    time = timers("all_iters").elapsed()
    print(f"elapsed time: {time}")

    device = next(model.parameters()).device
    memory_used = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
    memory_pct = memory_used / (torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)) * 100

    num_totol_tokens = sum([il + ol for _, il, ol in requests])

    print("=" * 11 + "SUMMARY" + "="*12)
    print(f"Num Request: {len(requests)}")
    print(f"Num totol tokens: {num_totol_tokens}")
    print(f"Batch Size: {args.micro_batch_size}")
    print(f"Num iters: {args.num_iters}")
    print(f"Avg Latency: {time / args.num_iters:.2f} seconds")
    print(f"Max Memory: {memory_used:.2f} GB ({memory_pct:.2f}%)")
    print("="*30)
