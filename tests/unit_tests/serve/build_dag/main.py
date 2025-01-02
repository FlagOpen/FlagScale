from vllm import LLM, SamplingParams
from custom.models import fn
import ray

class RemoteBase:
    def __init__(self, *args, **kwargs):
        self.remote_instance = ray.remote(self.__class__).remote(*args, **kwargs)

    def __getattr__(self, name):
        attr = getattr(self.remote_instance, name)
        
        if callable(attr):
            def wrapper(*args, **kwargs):
                return ray.get(attr.remote(*args, **kwargs))
            return wrapper
        else:
            return attr

class LLMActor(RemoteBase):
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


llm = LLMActor()

def model_A(prompt):
    result = llm.generate([prompt])
    return fn(result[0].outputs[0].text)


def model_B(input_data):
    res = input_data + "__add_model_B"
    return res


if __name__ == "__main__":
    prompt = "introduce Bruce Lee"
    print(model_A(prompt))
