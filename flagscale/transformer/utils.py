import torch
from typing import Any, Tuple

import torch.distributed as dist

from megatron.core import parallel_state 


def single_all_to_all(input, scatter_idx, gather_idx, group=None):

    seq_world_size = dist.get_world_size(group)
    # [seq_len/P, batch, nhead, dim] or [seq_len, batch, (nhead/P)*dim]
    inp_shape = list(input.shape)
    # [seq_len/P, batch, nhead, dim] -> [seq_len/P, batch, nhead/P, dim] (scatter_idx=2)
    # [seq_len, batch, (nhead/P)*dim] -> [seq_len/P, batch, (nhead/P)*dim] (scatter_idx=0)
    inp_shape[scatter_idx] = inp_shape[scatter_idx] // seq_world_size

    if scatter_idx < 2:
        # (scatter_idx=0)
        # [seq_len, batch, (nhead/P)*dim] -reshape-> [P, seq_len/P, batch, (nhead/P)*dim]
        input_t = input.reshape(
            [seq_world_size, inp_shape[scatter_idx]] + \
            inp_shape[scatter_idx + 1:]
        ).contiguous()
    else:
        # (scatter_idx=2)
        # [seq_len/P, batch, nhead, dim] -reshape-> [-1, P, nhead/P, dim] -transpose(0,1)-> [P, (seq_len/P)*batch, nhead/P, dim]
        input_t = input.reshape(
            [-1, seq_world_size, inp_shape[scatter_idx]] + \
            inp_shape[scatter_idx + 1:]
        ).transpose(0, 1).contiguous()

    output = torch.empty_like(input_t)
    # input_t: 
    # [P, (seq_len/P)*batch, nhead/P, dim] or [P, seq_len/P, batch, (nhead/P)*dim]
    dist.all_to_all_single(output, input_t, group=group)

    # if scattering the seq-dim, transpose the heads back to the original dimension
    if scatter_idx < 2:
        # [P, seq_len/P, batch, (nhead/P)*dim] -transpose-> [batch, seq_len/P, P, (nhead/P)*dim] -> [seq_len/P, batch, P, (nhead/P)*dim]
        output = output.transpose(0, 2).transpose(0, 1).contiguous()

    # [P, (seq_len/P)*batch, nhead/P, dim] -reshape-> [seq_len, batch, nhead/P, dim] (gather_idx=0)
    # [seq_len/P, batch, P, (nhead/P)*dim] -reshape-> [seq_len/P, batch, nhead*dim] (gather_idx=2)
    output = output.reshape(
        inp_shape[: gather_idx] + \
        [inp_shape[gather_idx] * seq_world_size,] + \
        inp_shape[gather_idx + 1:]).contiguous()

    return output


class _SeqAllToAll(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx: Any, 
        group,
        input: torch.Tensor, 
        scatter_idx: int,
        gather_idx: int,
    ) -> torch.Tensor:

        ctx.group = group
        ctx.scatter_idx = scatter_idx
        ctx.gather_idx = gather_idx
        return single_all_to_all(input, scatter_idx, gather_idx, group)

    @staticmethod
    def backward(
        ctx: Any, 
        *grad_output: torch.Tensor,
    ) -> Tuple[None, torch.Tensor, None, None]:
        return (
            None,
            _SeqAllToAll.apply(ctx.group, *grad_output, ctx.gather_idx, ctx.scatter_idx), 
            None, 
            None,
        )
