# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

import torch

TORCH_MAJOR = int(torch.__version__.split(".")[0])
TORCH_MINOR = int(torch.__version__.split(".")[1])

def fake_torch_compile(func):
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper

jit_fuser = fake_torch_compile
# nvFuser is deprecated in PyTorch JIT starting from 2.2
if (TORCH_MAJOR > 2) or (TORCH_MAJOR == 2 and TORCH_MINOR >= 5):
    jit_fuser = torch.compile
