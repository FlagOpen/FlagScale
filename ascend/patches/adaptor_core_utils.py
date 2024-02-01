import torch
import megatron


def _kernel_make_viewless_tensor(inp, requires_grad):
    out = torch.empty((1,), dtype=inp.dtype, device=inp.device, requires_grad=requires_grad, )
    with torch.no_grad():
        out.set_(inp.data)
    return out


megatron.core.utils._kernel_make_viewless_tensor = _kernel_make_viewless_tensor
