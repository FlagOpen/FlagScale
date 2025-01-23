from vllm import LLM, SamplingParams
from custom.models import fn
from flagscale.serve.core import auto_remote


def model_A(prompt):
    result = prompt + "__add_model_A"
    return fn(result)


def model_B(input_data):
    res = input_data + "__add_model_B"
    return res


if __name__ == "__main__":
    prompt = "introduce Bruce Lee"
    print(model_A(prompt))
