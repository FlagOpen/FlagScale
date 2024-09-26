import os
import sys
from flagscale.utils import CustomModuleFinder
sys.path.append(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))
sys.meta_path.insert(0, CustomModuleFinder())

import argparse
from typing import List

from transformers import AutoTokenizer
from vllm import LLM
from vllm.sampling_params import SamplingParams

from flagscale.inference.arguments import parse_args, FSArgumentParser


def get_group_args(group, args_dict):
    group_args = {arg.dest for arg in group._group_actions}
    return {key: args_dict[key] for key in group_args if key in args_dict}


def inference(args: argparse.Namespace, parser: FSArgumentParser, prompts: List[str], negative_prompts: List[str]=None):
    """Initialize the LLMEngine"""
    # step 1: parse arguments for different group
    args_dict = vars(args)
    engine_arg_group = parser.get_group("vLLMEngine")
    engine_args = get_group_args(engine_arg_group, args_dict)
    sampling_arg_group = parser.get_group("SamplingParams")
    sampling_params_args = get_group_args(sampling_arg_group, args_dict)

    # step 2: initialize the LLM engine
    fs_llm = LLM(**engine_args)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    fs_llm.set_tokenizer(tokenizer)
    print(f"==> {engine_args=}")

    # step 3: initialize the sampling parameters
    # TODO(zhaoyinglia): add logits processor
    assert sampling_params_args["logits_processors"] is None, "logits_processors is not supported yet."
    sampling_params_args["seed"] = sampling_params_args.pop("sampling_seed")
    sampling_params = SamplingParams(**sampling_params_args)
    print(f"==> {sampling_params=}")

    # step 4: build inputs
    inputs = {}
    if negative_prompts is not None:
        assert args.guidance_scale is not None
        assert len(prompts) == len(negative_prompts)
        inputs = [
            {
                "prompt": prompt, 
                "negative_prompt": negative_prompt
            } for prompt, negative_prompt in zip(prompts, negative_prompts)
        ]
    else:
        inputs = [{"prompt": prompt} for prompt in prompts]
    print(f"=> {inputs=}")

    # step 5: generate outputs
    outputs = fs_llm.generate(inputs, sampling_params)
    for output in outputs:
        print("*"*50)
        print(f"=> {output.prompt}")
        print(f"=> {output.outputs[0].text}")


if __name__ == '__main__':
    args, parser = parse_args()

    prompts = []
    if args.prompts_path is not None:
        with open(args.prompts_path, "r") as f:
            while True:
                prompt = f.readline()
                if not prompt:
                    break
                prompts.append(prompt[:-1]) # remove the last '\n' of prompt
    elif len(args.prompts) > 1:
        prompts = args.prompts
    else:
        raise ValueError("Pleace set right prompts_path or prompts data.")

    negative_prompts = None
    if len(args.negative_prompts) == 1:
        negative_prompts = args.negative_prompts * len(prompts)

    inference(args, parser, prompts, negative_prompts)
    print("=> Done!")
