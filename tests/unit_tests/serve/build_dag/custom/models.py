def fn(input_data):
    res = input_data + "__add_process_fn"
    return res


def model_C(input_data):
    res = input_data + "__add_model_C"
    return res


def model_D(input_data_B, input_data_C):
    output_data = input_data_B + input_data_C
    return output_data
