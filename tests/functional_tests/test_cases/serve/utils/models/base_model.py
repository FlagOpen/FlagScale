from util_models.util_model import fn


def model_A(prompt):
    result = prompt + "__add_model_A"
    return fn(result)


def model_B(input_data):
    res = input_data + "__add_model_B"
    return res


if __name__ == "__main__":
    prompt = "introduce Bruce Lee"
    print(model_A(prompt))
