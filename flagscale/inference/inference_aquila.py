import argparse
from typing import List, Union

import torch

from transformers import AutoTokenizer, LlamaForCausalLM, GenerationConfig
from vllm import EngineArgs, LLMEngine, RequestOutput, SamplingParams

from arguments import parse_args


def process_requests(prompts: List[str],
                     engine: LLMEngine,
                     sampling_params: SamplingParams):
    """Continuously process a list of prompts and handle the outputs."""
    request_id = 0
    while prompts:
        prompt = prompts.pop(0)
        engine.add_request(str(request_id), prompt, sampling_params)
        request_id += 1

    outputs: List[Union[RequestOutput]] = []
    while engine.has_unfinished_requests():
        step_outputs = engine.step()
        for output in step_outputs:
            if output.finished:
                outputs.append(output)

    outputs = sorted(outputs, key=lambda x: int(x.request_id))
    return outputs


def inference(args: argparse.Namespace, prompts: List[str]):
    """Initialize the LLMEngine"""
    engine_args = EngineArgs.from_cli_args(args)
    llm_engine = LLMEngine.from_engine_args(engine_args)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    llm_engine.tokenizer.tokenizer = tokenizer

    """Initialize the SamplingParams"""
    sampling_params = SamplingParams(
        n=args.n,
        best_of=args.best_of,
        frequency_penalty=args.frequency_penalty,
        repetition_penalty=args.repetition_penalty,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        min_p=args.min_p,
        seed=args.seed,
        use_beam_search=args.use_beam_search,
        length_penalty=args.length_penalty,
        early_stopping=args.early_stopping,
        stop=args.stop,
        stop_token_ids=args.stop_token_ids,
        include_stop_str_in_output=args.include_stop_str_in_output,
        ignore_eos=args.ignore_eos,
        max_tokens=args.max_tokens,
        min_tokens=args.min_tokens,
        logprobs=args.logprobs,
        prompt_logprobs=args.prompt_logprobs,
        detokenize=args.detokenize,
        skip_special_tokens=args.skip_special_tokens,
        spaces_between_special_tokens=args.spaces_between_special_tokens,
        # logits_processors=,
        # truncate_prompt_tokens=,
    )

    outputs = process_requests(prompts, llm_engine, sampling_params)
    for output in outputs:
        print("\n")
        print("="*50)
        print("=> RequestOutput:", output)
        token_ids = output.outputs[0].token_ids
        print("=> generated text:", tokenizer.decode(token_ids))


def generate(args: argparse.Namespace, prompts: List[str]):

    model = LlamaForCausalLM.from_pretrained(
        args.model, 
        torch_dtype=torch.bfloat16, 
        attn_implementation="flash_attention_2", 
        trust_remote_code=True
    ).to('cuda')
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    for prompt in prompts:
        print("\n")
        print("="*50)
        print("=> prompt:", prompt)
        tokens = tokenizer.encode_plus(prompt)["input_ids"]
        tokens = torch.tensor(tokens)[None,].to(model.device)
        input_length = len(tokens[0])
        generation_config = GenerationConfig(
            do_sample=True,
            eos_token_id=tokenizer.convert_tokens_to_ids('<|extra_204|>'),
            pad_token_id=tokenizer.convert_tokens_to_ids('<|endoftext|>'),
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
        )
        out = model.generate(
            tokens,
            generation_config,
            return_dict_in_generate=True, 
            output_scores=True,
        )
        out_ids = out["sequences"][0][input_length:].cpu().numpy()
        out_text = tokenizer.decode(out_ids.tolist())
        print("=> generated text:", out_text)


if __name__ == '__main__':
    args = parse_args()

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

    """
    vllm inference
    """
    inference(args, prompts)

    """
    transformers inference
    """
    # generate(args, prompts)
