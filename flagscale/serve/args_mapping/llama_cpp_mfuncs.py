import json


def kv_mapping_speculative_config(v) -> dict:
    if isinstance(v, str):
        try:
            v = json.loads(v)
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON string: {v}")
    if not isinstance(v, dict):
        raise ValueError(f"Expected a dictionary, got {type(v)}: {v}")

    draft_max = 16
    draft_min = 5
    if "num_speculative_tokens" in v.keys():
        if isinstance(v["num_speculative_tokens"], str):
            try:
                v["num_speculative_tokens"] = int(v["num_speculative_tokens"])
            except ValueError:
                raise ValueError(f"Invalid num_speculative_tokens: {v['num_speculative_tokens']}")
            draft_min = v["num_speculative_tokens"]
            draft_max = v["num_speculative_tokens"] * 2
    if "model" not in v.keys():
        raise ValueError(f"Missing 'model' key in the dictionary: {v}")
    model = v["model"]
    return {"draft_max": draft_max, "draft_min": draft_min, "model_draft": model}


def kv_mapping_override_pooler_config(v) -> dict:
    if isinstance(v, str):
        try:
            v = json.loads(v)
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON string: {v}")
    if not isinstance(v, dict):
        raise ValueError(f"Expected a dictionary, got {type(v)}: {v}")
    if "pooling_type" not in v:
        raise ValueError(f"Missing 'pooler_type' key in the dictionary: {v}")
    pooling_type = v["pooling_type"]
    if pooling_type not in ["none", "mean", "cls", "last", "rank"]:
        raise ValueError(f"Invalid pooling_type: {pooling_type}")
    return {"pooling": pooling_type}


def kv_mapping_rope_scaling(v) -> dict:
    if isinstance(v, str):
        try:
            v = json.loads(v)
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON string: {v}")
    if not isinstance(v, dict):
        raise ValueError(f"Expected a dictionary, got {type(v)}: {v}")
    rope_type = v.get("rope_type")
    factor = v.get("factor")
    if rope_type not in ["none", "linear", "yarn"]:
        raise ValueError(f"Invalid rope_type: {rope_type}")
    if factor is None:
        raise ValueError(f"Missing factor in the dictionary: {v}")
    return {"rope_scaling": rope_type, "rope_scale": factor}


def kv_mapping_kv_cache_dtype(v) -> dict:
    if v in ["f32", "f16", "bf16", "q8_0", "q4_0", "q4_1", "iq4_nl", "q5_0", "q5_1"]:
        return {"cache_type_k": v, "cache_type_v": v}
    raise ValueError(f"Invalid kv_cache_dtype for llama.cpp: {v}")


def kv_mapping_reasoning_parser(v) -> dict:
    if v in ["deepseek", "deepseek_r1", "deep-seek", "DeepSeek", "Deep-Seek"]:
        return {"reasoning_format": "deepseek"}
    raise ValueError(f"Invalid reasoning_parser for llama.cpp: {v}")


def kv_mapping_lora_modules(v) -> dict:
    if isinstance(v, str):
        try:
            v = json.loads(v)
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON string: {v}")
    if not isinstance(v, dict):
        raise ValueError(f"Expected a dictionary, got {type(v)}: {v}")
    if "path" not in v:
        raise ValueError(f"Missing 'lora' key in the dictionary: {v}")

    return {"lora": v["path"]}


def kv_mapping_max_model_len(v) -> int:
    if isinstance(v, int):
        return {"ctx_size": v}
    if not isinstance(v, str):
        raise ValueError(f"Invalid max_model_len: {v, type(v)}")

    if "k" in v:
        v = v.replace("k", "")
        try:
            v = int(v) * 1000
        except TypeError:
            raise TypeError(f"Invalid max_model_len: {v, type(v)}")
    elif "K" in v:
        v = v.replace("K", "")
        try:
            v = int(v) * 1024
        except TypeError:
            raise TypeError(f"Invalid max_model_len: {v, type(v)}")
    if not isinstance(v, int):
        raise ValueError(f"Expected an integer, got {type(v)}: {v}")
    return {"ctx_size": v}


def kv_mapping_uvicorn_log_level(v) -> dict:
    if v in ["warning", "error", "critical", "trace", "WARNING", "ERROR", "CRITICAL", "TRACE"]:
        return {"log_verbosity": 0}
    elif v in ["debug", "info", "DEBUG", "INFO"]:
        return {"log_verbosity": 1}
    raise ValueError(f"Invalid uvicorn_log_level(log_verbosity) for llama.cpp: {v}")
