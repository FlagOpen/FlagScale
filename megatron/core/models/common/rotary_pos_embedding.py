# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import importlib.util

import torch
from torch import einsum, nn

import os

try:
    import torch_xmlir
except Exception:
    torch_xmlir = None

__all__ = ['RotaryEmbedding', 'apply_rotary_pos_emb']


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, seq_len_interpolation_factor=None):
        super().__init__()
        self.seq_len_interpolation_factor = seq_len_interpolation_factor
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq, persistent=False)
        self.dim = dim

    def forward(self, max_seq_len, offset=0, use_rotary_in_fp32=False):
        seq = torch.arange(max_seq_len, device=self.inv_freq.device, dtype=torch.float32) + offset
        if self.seq_len_interpolation_factor is not None:
            seq = seq.type_as(self.inv_freq)
            seq *= 1 / self.seq_len_interpolation_factor
        if use_rotary_in_fp32 and self.inv_freq.dtype is not torch.float32:
            self.inv_freq = 1.0 / (10000 ** (torch.arange(0, self.dim, 2, dtype=torch.float32, device=self.inv_freq.device).float() / self.dim))
            freqs = torch.outer(seq, self.inv_freq)
        else:
            freqs = einsum('i , j -> i j', seq.type_as(self.inv_freq), self.inv_freq)

        # first part even vector components, second part odd vector components,
        #  2 * dim in dimension size
        emb = torch.cat((freqs, freqs), dim=-1)
        # emb [seq_length, .., dim]
        return emb[:, None, None, :]

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        state_dict.pop(f'{prefix}inv_freq', None)
        return super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

class RotaryPositionalEmbeddingWithFreqFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, t, freqs):
        t_ = t.contiguous()
        freqs_ = freqs.contiguous()
        output = t_.new_empty(t_.shape)
        torch.ops.custom_ops.rotary_pos_emb(t_, freqs_, out = output)
        ctx.freqs = freqs_
        return output

    @staticmethod
    def backward(ctx, grad_output):
        freqs_ = ctx.freqs
        grad_t = grad_output.new_empty(grad_output.shape)
        torch.ops.custom_ops.rotary_pos_emb_backward(grad_output, freqs_, out = grad_t)
        return grad_t, None

def _rotate_half(x):
    """
    change sign so the last dimension becomes [-odd, +even]
    """
    x1, x2 = torch.chunk(x, 2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(t, freqs):
    """
    input tensor t is of shape [seq_length, ..., dim]
    rotary positional embeding tensor freqs is of shape [seq_length, ..., dim]
    check https://kexue.fm/archives/8265 for detailed formulas
    """
    rot_dim = freqs.shape[-1]
    # ideally t_pass is empty so rotary pos embedding is applied to all tensor t
    t, t_pass = t[..., :rot_dim], t[..., rot_dim:]

    # first part is cosine component
    # second part is sine component, need to change signs with _rotate_half method
    inference_flag = (os.getenv("AQUILA_INFERENCE_ROTARY_EMBEDDING", "false").lower() == "true")
    if torch_xmlir is not None and inference_flag:
        t = RotaryPositionalEmbeddingWithFreqFunction.apply(t, freqs)
    else:
        t = (t * freqs.cos()) + (_rotate_half(t) * freqs.sin())
    return torch.cat((t, t_pass), dim=-1)
