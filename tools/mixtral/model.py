import time
from megatron.core.enums import ModelType

import sys
sys.path.append("..")
from utils import suspend_nn_inits

model_type = ModelType.encoder_or_decoder


def get_hf_model(dtype, model_path=None, config=None):
    from transformers import AutoModelForCausalLM
    s_time = time.time()
    # with suspend_nn_inits():
    if config is None:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, device_map="cpu", trust_remote_code=True, torch_dtype=dtype
        )
    elif model_path is None:
        model = AutoModelForCausalLM.from_config(
            config=config, trust_remote_code=True, torch_dtype=dtype
        )
        print("model:", model)
    else:
        raise ValueError("")
    print("> loading huggingface model elapsed time:", time.time() - s_time)
    return model


def get_mg_model(dtype, pre_process, post_process):
    from pretrain_gpt import model_provider
    s_time = time.time()
    # with suspend_nn_inits():
    model = model_provider(pre_process, post_process).to(dtype)
    print("> loading megatron model elapsed time:", time.time() - s_time)
    return model
