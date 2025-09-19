from typing import Any, Optional, Union

import torch


def parse_torch_dtype(value: Any) -> Optional[Union[torch.dtype, str]]:
    """Parse a YAML-provided dtype value into a torch.dtype.

    Accepts values like:
    - torch.bfloat16, bfloat16, bf16
    - torch.float16, float16, fp16, half
    - torch.float32, float32, fp32
    - torch.float64, float64, fp64
    - auto, none

    Returns:
        - torch.dtype if recognized
        - None if value is None, "none", or unrecognized
        - Depending on policy, "auto" may be returned as a literal string. Current policy: return None
          to omit torch_dtype and let the backend decide.
    """

    if value is None:
        return None

    if isinstance(value, torch.dtype):
        return value

    if isinstance(value, str):
        s = value.strip()
        # Current policy: treat "auto" and "none" as unset
        if s.lower() in {"auto", "none"}:
            return None
        if s.startswith("torch."):
            s = s.split(".", 1)[1]
        alias_map = {
            "bf16": "bfloat16",
            "bfloat16": "bfloat16",
            "fp16": "float16",
            "half": "float16",
            "float16": "float16",
            "fp32": "float32",
            "float32": "float32",
            "fp64": "float64",
            "float64": "float64",
        }
        key = alias_map.get(s.lower(), s.lower())
        if hasattr(torch, key) and isinstance(getattr(torch, key), torch.dtype):
            return getattr(torch, key)

    return None
