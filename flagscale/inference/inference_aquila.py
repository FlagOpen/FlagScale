import os
import sys
sys.path.append(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))
from flagscale.utils import CustomModuleFinder
sys.meta_path.insert(0, CustomModuleFinder())

from transformers import AutoTokenizer

from vllm import LLM
from vllm.sampling_params import SamplingParams

from flagscale.inference.arguments import parse_config


def inference():
    """Initialize the LLMEngine"""
    # step 1: parse inference config
    cfg = parse_config()

    prompts = []
    if cfg.generate.get("prompts_path", None):
        with open(cfg.generate.prompts_path, "r") as f:
            while True:
                prompt = f.readline()
                if not prompt:
                    break
                prompts.append(prompt[:-1]) # remove the last '\n' of prompt
    elif cfg.generate.get("prompts", None):
        prompts = cfg.generate.prompts
    else:
        raise ValueError("Pleace set right prompts_path or prompts.")

    negative_prompts = []
    if cfg.generate.get("negative_prompts_path", None):
        with open(cfg.generate.prompts_path, "r") as f:
            while True:
                negative_prompt = f.readline()
                if not negative_prompt:
                    break
                negative_prompts.append(negative_prompt[:-1]) # remove the last '\n' of prompt
    elif cfg.generate.get("prompts", None):
        negative_prompts = cfg.generate.negative_prompts
    else:
        raise ValueError("Pleace set right negative_prompts_path or negative_prompts.")

    # step 2: initialize the LLM engine
    llm_cfg = cfg.get("llm", {})
    llm = LLM(**llm_cfg)

    tokenizer_cfg = llm_cfg.get("tokenizer", None)
    if tokenizer_cfg:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_cfg, trust_remote_code=True)
        llm.set_tokenizer(tokenizer)

    # step 3: initialize the sampling parameters
    # TODO(zhaoyinglia): support config logits processor
    sampling_cfg = cfg.generate.get("sampling", {})
    assert not sampling_cfg.get("logits_processors", None), "logits_processors is not supported yet."
    sampling_params = SamplingParams(**sampling_cfg)
    print(f"=> {sampling_params=}")

    # step 4: build inputs
    inputs = [{"prompt": prompt, "negative_prompt": negative_prompt} for prompt, negative_prompt in zip(prompts, negative_prompts)]
    print(f"=> {inputs=}")

    # step 5: generate outputs
    outputs = llm.generate(inputs, sampling_params)
    for output in outputs:
        print("*"*50)
        print(f"{output.prompt=}")
        print(f"{output.outputs[0].text=}")
        print(f"{output.outputs[0].token_ids=}")


if __name__ == '__main__':
    inference()
