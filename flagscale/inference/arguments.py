import argparse
from typing import List, Union

from vllm import EngineArgs


def parse_args(ignore_unknown_args=False):
    parser = argparse.ArgumentParser(description='vLLM Inference')
    _add_additional_args(parser)
    _add_vllm_engine_args(parser)
    _add_sampling_args(parser)

    if ignore_unknown_args:
        args, _ = parser.parse_known_args()
    else:
        args = parser.parse_args()

    return args


def _add_additional_args(parser):
    group = parser.add_argument_group(title='vLLM-additional-args')

    group.add_argument("--prompts-path",
                        type=str,
                        default=None,
                        help="the text file contain the prompts")
    group.add_argument("--prompts", 
                       nargs='*',
                       help="A list of prompts to generate completions for.")


def _add_vllm_engine_args(parser):
    group = parser.add_argument_group(title='vLLM-Engine')
    group = EngineArgs.add_cli_args(group)
    return parser


def _add_sampling_args(parser):
    group = parser.add_argument_group(title='vLLM-sampling-params')

    group.add_argument("--n",
                        type=int,
                        default=1,
                        help="Number of output sequences to return for the given prompt.")
    group.add_argument("--best_of",
                        type=int,
                        default=None,
                        help="Number of output sequences that are generated from the prompt.")
    group.add_argument("--presence-penalty",
                       type=float,
                       default=0.0,
                       help="Float that penalizes new tokens based on whether they appear in the generated text so far.")
    group.add_argument("--frequency-penalty",
                       type=float,
                       default=0.0,
                       help="Float that penalizes new tokens based on their frequency in the generated text so far.")
    group.add_argument("--repetition-penalty",
                        type=float,
                        default=1.0,
                        help="Float that penalizes new tokens based on whether they appear in the prompt and the generated text so far.")
    group.add_argument("--temperature",
                        type=float,
                        default=1.0,
                        help="Float that controls the randomness of the sampling.")
    group.add_argument("--top-p",
                        type=float,
                        default=1.0,
                        help="Float that controls the cumulative probability of the top tokens to consider.")
    group.add_argument("--top-k",
                        type=int,
                        default=-1,
                        help="Integer that controls the number of top tokens to consider.")
    group.add_argument("--min-p",
                        type=float,
                        default=0.0,
                        help="Float that represents the minimum probability for a token to be considered.")
    group.add_argument("--use-beam-search",
                        type=bool,
                        default=False,
                        help="Whether to use beam search instead of sampling.")
    group.add_argument("--length-penalty",
                        type=float,
                        default=1.0,
                        help="Float that penalizes sequences based on their length.")
    group.add_argument("--early-stopping",
                        type=Union[bool, str],
                        default=False,
                        help="Controls the stopping condition for beam search.")
    group.add_argument("--stop",
                        type=Union[str, List[str]],
                        default=None,
                        help="List of strings that stop the generation when they are generated.")
    group.add_argument("--stop-token-ids",
                        type=List[int],
                        default=None,
                        help="List of tokens that stop the generation when they are generated.")
    group.add_argument("--include-stop-str-in-output",
                        type=bool,
                        default=False,
                        help="Whether to include the stop strings in output text.")
    group.add_argument("--ignore-eos",
                        type=bool,
                        default=False,
                        help="Whether to ignore the EOS token and continue generating tokens after the EOS token is generated.")
    group.add_argument("--max-tokens",
                        type=int,
                        default=16,
                        help="Maximum number of tokens to generate per output sequence.")
    group.add_argument("--min-tokens",
                        type=int,
                        default=0,
                        help="Minimum number of tokens to generate per output sequence before EOS or stop_token_ids can be generated.")
    group.add_argument("--logprobs",
                        type=int,
                        default=None,
                        help="Number of log probabilities to return per output token.")
    group.add_argument("--prompt-logprobs",
                        type=int,
                        default=None,
                        help="Number of log probabilities to return per prompt token.")
    group.add_argument("--detokenize",
                        type=bool,
                        default=True,
                        help="Whether to detokenize the output.")
    group.add_argument("--skip-special-tokens",
                        type=bool,
                        default=True,
                        help="Whether to skip special tokens in the output.")
    group.add_argument("--spaces-between-special-tokens",
                        type=bool,
                        default=True,
                        help="Whether to add spaces between special tokens in the output.")
