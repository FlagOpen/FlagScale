import torch

import megatron
from megatron.core.parallel_state import (
    get_tensor_model_parallel_group,
    get_tensor_model_parallel_world_size,
)

def _gather_along_first_dim(input_):
    """Gather tensors and concatinate along the first dimension."""

    world_size = get_tensor_model_parallel_world_size()
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_

    dim_size = list(input_.size())
    dim_size[0] = dim_size[0] * world_size

    output = torch.empty(dim_size, dtype=input_.dtype, device='musa:{}'.format(torch.musa.current_device()))
    torch.distributed._all_gather_base(
        output, input_.contiguous(), group=get_tensor_model_parallel_group()
    )

    return output

def _reduce_scatter_along_first_dim(input_):
    """Reduce-scatter the input tensor across model parallel group."""
    world_size = get_tensor_model_parallel_world_size()
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_

    dim_size = list(input_.size())
    assert (
        dim_size[0] % world_size == 0
    ), "First dimension of the tensor should be divisible by tensor parallel size"

    dim_size[0] = dim_size[0] // world_size

    output = torch.empty(dim_size, dtype=input_.dtype, device='musa:{}'.format(torch.musa.current_device()))
    torch.distributed._reduce_scatter_base(
        output, input_.contiguous(), group=get_tensor_model_parallel_group()
    )
    return output

# megatron.core.tensor_parallel.mappings._gather_along_first_dim = _gather_along_first_dim
# megatron.core.tensor_parallel.mappings._reduce_scatter_along_first_dim = _reduce_scatter_along_first_dim
