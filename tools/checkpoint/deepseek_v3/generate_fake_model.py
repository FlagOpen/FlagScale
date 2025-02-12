import time
import sys

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

# save faked hf model
path = "./generate_bf16_model"
print(f"start saving deepseek v3")
config = AutoConfig.from_pretrained(path, trust_remote_code=True)
model = AutoModelForCausalLM.from_config(config, torch_dtype=config.torch_dtype,trust_remote_code=True)
save_path = "./fake_bf16_model"
model.save_pretrained(save_path)
print(f"end saving deepseek v3")
print(f"config is {config}")
print(f"model is {model}")

