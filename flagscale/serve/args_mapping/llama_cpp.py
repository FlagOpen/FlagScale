from flagscale.serve.args_mapping.llama_cpp_mfuncs import (
    kv_mapping_kv_cache_dtype,
    kv_mapping_lora_modules,
    kv_mapping_max_model_len,
    kv_mapping_override_pooler_config,
    kv_mapping_reasoning_parser,
    kv_mapping_speculative_config,
    kv_mapping_uvicorn_log_level,
)

# Serve args(keys) mapping from vllm to llama_cpp
VLLM_LLAMA_K_MAPPING = {
    "disable_uvicorn_access_log": "log_disable",
    "ssl_keyfile": "ssl_key_file",
    "ssl_certfile": "ssl_cert_file",
    "rope_theta": "rope_freq_base",
    "served_model_name": "alias",
    "show_hidden_metrics_for_version": "metrics",
    "max_num_seqs": "parallel",
}


# Serve args(keys and values) mapping from vllm to llama_cpp
VLLM_LLAMA_KV_MAPPING = {
    "uvicorn_log_level": kv_mapping_uvicorn_log_level,
    "lora_modules": kv_mapping_lora_modules,
    "max_model_len": kv_mapping_max_model_len,
    "reasoning_parser": kv_mapping_reasoning_parser,
    "kv_cache_dtype": kv_mapping_kv_cache_dtype,
    "speculative_config": kv_mapping_speculative_config,
    "override_pooler_config": kv_mapping_override_pooler_config,
}


def llama_cpp_args_converter(src_map: dict) -> dict:
    dst_map = {}
    for key, value in src_map.items():
        if key in VLLM_LLAMA_K_MAPPING:
            dst_key = VLLM_LLAMA_K_MAPPING[key]
            dst_map[dst_key] = value
        elif key in VLLM_LLAMA_KV_MAPPING:
            dst_func = VLLM_LLAMA_KV_MAPPING[key]
            new_kvs = dst_func(value)
            dst_map.update(new_kvs)
        else:
            dst_map[key] = value
    return dst_map


if __name__ == "__main__":
    src_map = {
        "model": "llama-7b",
        "max_model_len": "1K",
        "max_num_seqs": 4,
        "kv_cache_dtype": "f16",
        "uvicorn_log_level": "warning",
    }
    print(llama_cpp_args_converter(src_map))
