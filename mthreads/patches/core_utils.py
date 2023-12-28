import operator
from functools import reduce

import torch
import megatron

class GlobalMemoryBuffer:
    """Global buffer to avoid dynamic memory allocations.
    Caller should ensure that buffers of the same name
    are not used concurrently."""

    def __init__(self):
        self.buffer = {}

    def get_tensor(self, tensor_shape, dtype, name):
        required_len = reduce(operator.mul, tensor_shape, 1)
        if (
            self.buffer.get((name, dtype), None) is None
            or self.buffer[(name, dtype)].numel() < required_len
        ):
            self.buffer[(name, dtype)] = torch.empty(
                required_len, dtype=dtype, device='musa:{}'.format(torch.musa.current_device()), requires_grad=False
            )

        return self.buffer[(name, dtype)][0:required_len].view(*tensor_shape)
    
megatron.core.utils.GlobalMemoryBuffer = GlobalMemoryBuffer