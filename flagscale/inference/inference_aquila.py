import os
import sys

from PIL import Image
from transformers import AutoTokenizer

from vllm import LLM
from vllm.sampling_params import SamplingParams

from flagscale.inference.arguments import parse_config


def build_multimodal_prompt(question: str, modality: str = "image"):
    """
    Build a multimodal prompt with placeholders for image/video.
    """
    if modality == "image":
        placeholder = "<|image_pad|>"
    elif modality == "video":
        placeholder = "<|video_pad|>"
    else:
        raise ValueError(f"Unsupported modality: {modality}")

    prompt = (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        f"<|im_start|>user\n<|vision_start|>{placeholder}<|vision_end|>{question}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    return prompt


def clean_user_prompt(prompt: str):
    """
    Extract only the user's original input text from the full prompt.
    """
    start_token = "<|vision_end|>"
    end_token = "<|im_end|>"
    start_idx = prompt.find(start_token)
    if start_idx != -1:
        start_idx += len(start_token)
        end_idx = prompt.find(end_token, start_idx)
        if end_idx == -1:
            end_idx = len(prompt)
        user_text = prompt[start_idx:end_idx].strip()
        return user_text
    return prompt.strip()


def clean_text(text: str):
    """
    Remove placeholders or template tags from model output.
    """
    tags = [
        "<|image_pad|>",
        "<|video_pad|>",
        "<|vision_start|>",
        "<|vision_end|>",
        "<|im_start|>",
        "<|im_end|>",
        "<think>",
        "</think>",
        "<answer>",
        "</answer>",
    ]
    for tag in tags:
        text = text.replace(tag, "")
    return text.strip()


def inference(cfg):
    """Initialize the LLMEngine"""
    # step 1: parse inference config
    prompts = cfg.generate.get("prompts", [])
    assert prompts, "Please set the prompts in the config yaml."

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
    assert not sampling_cfg.get(
        "logits_processors", None
    ), "logits_processors is not supported yet."
    sampling_params = SamplingParams(**sampling_cfg)
    print(f"=> {sampling_params=}")

    # step 4: build inputs
    inputs = [{"prompt": prompt} for prompt in prompts]
    print(f"=> {inputs=}")

    # step 5: generate outputs
    outputs = llm.generate(inputs, sampling_params)
    for output in outputs:
        print("*" * 50)
        print(f"{output.prompt=}")
        print(f"{output.outputs[0].text=}")
        print(f"{output.outputs[0].token_ids=}")
    print("#" * 50)


def inference_mul(cfg):
    """Initialize the LLMEngine"""
    # step 1: parse inference config
    images = cfg.generate.get("images", [])
    prompts = cfg.generate.get("prompts", [])
    assert len(prompts) == len(images), "Please set the same length of prompts and images."

    # step 2: initialize the LLM engine
    llm_cfg = cfg.get("llm", {})
    llm = LLM(**llm_cfg)

    tokenizer_cfg = llm_cfg.get("tokenizer", None)
    if tokenizer_cfg:
        print(f"&#128230: Loading tokenizer from: {tokenizer_cfg}")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_cfg, trust_remote_code=True)
        llm.set_tokenizer(tokenizer)
    else:
        print(
            "&#9888; Warning: 'tokenizer' not found in configuration, model default tokenizer will be used"
        )

    # step 3: initialize the sampling parameters
    sampling_cfg = cfg.generate.get("sampling", {})
    assert not sampling_cfg.get(
        "logits_processors", None
    ), "logits_processors is not supported yet."
    sampling_params = SamplingParams(**sampling_cfg)
    print(f"=> {sampling_params=}")

    # step 4: build inputs
    inputs = [
        {
            "prompt": (
                "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                f"<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{prompt}<|im_end|>\n"
                "<|im_start|>assistant\n"
            ),
            "multi_modal_data": {"image": [Image.open(image).convert("RGB")]},
        }
        for prompt, image in zip(prompts, images)
    ]

    print(f"=> {inputs=}")

    # step 5: generate outputs
    try:
        outputs = llm.generate(inputs, sampling_params)
        for output in outputs:
            print("*" * 50)
            clean_prompt = clean_user_prompt(output.prompt)
            clean_output = clean_text(output.outputs[0].text) if output.outputs else ""
            print(f"output.prompt='{clean_prompt}'")
            print(f"output.outputs[0].text='{clean_output}'")
            print(f"{output.outputs[0].token_ids=}")
        print("#" * 50)
    except Exception as e:
        print(f"Errors occurred during the reasoning process: {str(e)}")


if __name__ == "__main__":
    cfg = parse_config()
    if cfg.generate.get("images", []) is None:
        inference(cfg)
    else:
        inference_mul(cfg)
