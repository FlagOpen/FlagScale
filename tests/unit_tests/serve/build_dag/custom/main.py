import os
from pydantic import BaseModel
from vllm import LLM, SamplingParams

from custom.models import fn


#os.environ['CUDA_VISIBLE_DEVICES']="0,1,2,3,4,5"

class GenerateRequest(BaseModel):
    prompt: str


class LLMActor:
    def __init__(self):
        # Initialize the LLM inside the actor to avoid serialization
        self.llm = LLM(
            model="/models/Qwen2.5-0.5B-Instruct",
            tensor_parallel_size=1,
            gpu_memory_utilization=0.5
        )

    def generate(self, prompt: str) -> str:
        sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.95,
            max_tokens=1000
        )
        result = self.llm.generate([prompt], sampling_params=sampling_params)
        return result[0].outputs[0].text

#llm = LLM(model="/models/Qwen2.5-0.5B-Instruct", tensor_parallel_size=1, gpu_memory_utilization=0.5)

actor = LLMActor()

#prompt="introduce Bruce Lee"
def model_A(prompt):
    #prompt="introduce Bruce Lee"
    #llm = LLM(model="/models/Qwen2.5-0.5B-Instruct", tensor_parallel_size=1, gpu_memory_utilization=0.5)
    #sampling_params = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=1000)

    result = actor.generate(prompt)                                                                                                                                                   
    return result

def model_B(input_data):
    res = input_data + "__add_model_B"
    return res

if __name__ == "__main__":
    print(model_A())
