import os
from functools import wraps
import torch

try:
    import torch_mlu
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True
    from torch_mlu.utils.model_transfer import transfer
except:
    pass


if hasattr(torch.distributed, "all_gather_into_tensor") and \
   hasattr(torch.distributed, "reduce_scatter_tensor"):
    torch.distributed._all_gather_base = torch.distributed.all_gather_into_tensor
    torch.distributed._reduce_scatter_base = torch.distributed.reduce_scatter_tensor

def wrapper_type(fn):
    @wraps(fn)
    def decorated(*args, **kwargs):
        output = fn(*args, **kwargs)
        if isinstance(output, str):
            if output == 'torch.mlu.FloatTensor':
                output = 'torch.cuda.FloatTensor'
            elif output == 'torch.mlu.BFloat16Tensor':
                output = 'torch.cuda.BFloat16Tensor'
            elif output == 'torch.mlu.HalfTensor':
                output = 'torch.cuda.HalfTensor'
        return output

    return decorated

def wrapper_backend(fn):
    @wraps(fn)
    def decorated(*args, **kwargs):
        output = fn(*args, **kwargs)
        if isinstance(output, str):
            if output == 'cncl':
                output = 'nccl'
        return output

    return decorated


os.environ['CUDA_DEVICE_MAX_CONNECTIONS'] = '1'

torch.Tensor.type = wrapper_type(torch.Tensor.type)
torch.distributed.get_backend = wrapper_backend(torch.distributed.get_backend)
