import os
import sys
from flagscale.utils import CustomModuleFinder
sys.path.append(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))
sys.meta_path.insert(0, CustomModuleFinder())

from typing import List

from vllm import LLM, SamplingParams
from vllm.inputs import PromptInputs

llm = LLM(
    model="/share/project/zhaoyingli/checkpoints/opt-6.7b/",
    tensor_parallel_size=1,
    use_v2_block_manager=True,
)

prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
negative_prompts = ["diverse languages"] * 4

inputs: List[PromptInputs]=[{"prompt": prompt, "negative_prompt": prompt[-1]} for prompt in prompts]

sampling_params = SamplingParams(temperature=0.8, top_p=0.95, guidance_scale=5.0)
outputs = llm.generate(inputs, sampling_params)

for i, output in enumerate(outputs):
    prompt = prompts[i]
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
