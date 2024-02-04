import os
import sys
import torch
pardir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
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
    group = parser.add_argument_group(title='Benchmark text generation')

    group.add_argument("--num-requests", type=int, default=1,
                       help='Number of requests to process.')
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
    # group.add_argument("--dataset-path", type=str, default=None,
    #                    help='Path to the requests data.')

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

    # TODO(zhaoyingli): get requests from real dataset.
    prompt = "我" * (args.prompt_len - 1)
    requests = [(prompt, args.generate_len)] * args.num_requests

    print("warming up....")
    generate_and_post_process(
            model,
            prompts=prompt,
            tokens_to_generate=args.generate_len,
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
    timers("all_requests", log_level=0).start(barrier=True)
    for idx, (promot, gen_len) in enumerate(requests):
        timers(f"request_{idx}", log_level=0).start()
        generate_and_post_process(
            model,
            prompts=prompt,
            tokens_to_generate=gen_len,
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
        print(f"request_{idx} elapsed time:", timers(f"request_{idx}").elapsed())

    time = timers("all_requests").elapsed()
    total_num_tokens = (args.prompt_len + args.generate_len) * args.num_requests

    device = next(model.parameters()).device
    memory_used = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
    memory_pct = memory_used / (torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)) * 100

    print("="*30)
    print(f"Requests num: {args.num_requests}")
    print(f"Prompt len: {args.prompt_len}")
    print(f"Generate len: {args.generate_len}")
    print(f"Throughput: {args.num_requests / time:.2f} requests/s, {total_num_tokens / time:.2f} tokens/s")
    print(f"Max Memory: {memory_used:.2f} GB ({memory_pct:.2f}%)")
    print("="*30)
