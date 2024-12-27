from vllm import LLM, SamplingParams
from custom.models import fn


def model_A(prompt):
    llm = LLM(
        model="/models/Qwen2.5-0.5B-Instruct",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.5,
    )
    sampling_params = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=1000)

    result = llm.generate([prompt], sampling_params=sampling_params)
    return fn(result[0].outputs[0].text)


def model_B(input_data):
    res = input_data + "__add_model_B"
    return res


if __name__ == "__main__":
    prompt = "introduce Bruce Lee"
    print(model_A(prompt))
