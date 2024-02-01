import sys

import torch_npu
import megatron
from megatron.core.models.common.rotary_pos_embedding import apply_rotary_pos_emb

def apply_rotary_pos_emb(t, freqs):
    """
    input tensor t is of shape [seq_length, ..., dim]
    rotary positional embeding tensor freqs is of shape [seq_length, ..., dim]
    check https://kexue.fm/archives/8265 for detailed formulas
    """
    return torch_npu.npu_rotary_mul(t, freqs.cos(), freqs.sin())

megatron.core.models.common.rotary_pos_embedding.apply_rotary_pos_emb = apply_rotary_pos_emb

for k, v in sys.modules.items():
    if 'megatron' in k and hasattr(v, 'apply_rotary_pos_emb'):
        setattr(v, 'apply_rotary_pos_emb', apply_rotary_pos_emb)
