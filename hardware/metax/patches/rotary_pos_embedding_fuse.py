from __future__ import annotations

import torch
import megatron
import importlib

from torch import nn, einsum


class RotaryEmbedding(nn.Module):
   def __init__(self, dim):
       super().__init__()
       inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
       self.register_buffer('inv_freq', inv_freq)
       if importlib.util.find_spec('einops') is None:
           raise RuntimeError("einops is required for Rotary Embedding")

   def forward(self, max_seq_len, offset=0):
       seq = torch.arange(max_seq_len, device=self.inv_freq.device) + offset
       freqs = einsum('i , j -> i j', seq.type_as(self.inv_freq), self.inv_freq)
       # first part even vector components, second part odd vector components,
       #  2 * dim in dimension size
       emb = torch.cat((freqs, freqs), dim=-1)
       # emb [seq_length, .., dim]
       from einops import rearrange
       return rearrange(emb, 'n d -> n 1 1 d')   




import sys
for k in sys.modules:
    if k.startswith('megatron.core.models.common.embeddings.rotary_pos_embedding'):
        if getattr(sys.modules[k], 'RotaryEmbedding', None):
            setattr(sys.modules[k], 'RotaryEmbedding', RotaryEmbedding)

