# [metax] start of change
from __future__ import annotations


from typing import Tuple

import rotary_emb
import torch
import torch.nn.functional as F
from einops import rearrange
from flash_attn.layers.rotary import ApplyRotaryEmb as LegacyApplyRotaryEmb
from flash_attn.layers.rotary import ApplyRotaryEmbQKV_ as LegacyApplyRotaryEmbQKV_
from torch import Tensor, nn, einsum
import numpy as np

from megatron.core import parallel_state
from flagscale.patches_utils import add_patches_module

__all__ = ['RotaryEmbedding', 'apply_rotary_pos_emb']




class ApplyRotaryEmbQKV_(torch.autograd.Function):
    """MENGXIA
    ApplyRotaryEmbQKV_
    """

    @staticmethod
    def forward(ctx, qkv, cos, sin, cos_k=None, sin_k=None):
        """
            qkv: (total, 3, nheads, headdim)
            cos, sin: (seqlen, rotary_dim / 2)
            cos_k, sin_k: (seqlen, rotary_dim / 2), optional
        rotary_dim must be <= headdim
        Apply rotary embedding *inplace* to the first rotary_dim of q and k.
        """
        _, three, _, headdim = qkv.shape
        assert three == 3
        print('cos_shape = ', cos.shape)

        rotary_seqlen, rotary_dim = cos.shape
        rotary_dim *= 2
        assert rotary_dim <= headdim
        cos_k = cos if cos_k is None else cos_k
        sin_k = sin if sin_k is None else sin_k
        assert sin.shape == cos_k.shape == sin_k.shape == (rotary_seqlen, rotary_dim // 2)
        q1, q2 = qkv[:, 0, :, :rotary_dim].chunk(2, dim=-1)
        rotary_emb.apply_rotary(q1, q2, rearrange(cos, "s d -> s 1 d"), rearrange(sin, "s d -> s 1 d"), q1, q2, False)
        k1, k2 = qkv[:, 1, :, :rotary_dim].chunk(2, dim=-1)
        rotary_emb.apply_rotary(
            k1, k2, rearrange(cos_k, "s d -> s 1 d"), rearrange(sin_k, "s d -> s 1 d"), k1, k2, False
        )
        ctx.save_for_backward(cos, sin, cos_k, sin_k)
        return qkv

    @staticmethod
    def backward(ctx, dqkv):
        cos, sin, cos_k, sin_k = ctx.saved_tensors
        rotary_dim = cos.shape[-1]
        rotary_dim *= 2
        dq1, dq2 = dqkv[:, 0, :, :rotary_dim].chunk(2, dim=-1)
        rotary_emb.apply_rotary(
            dq1, dq2, rearrange(cos, "s d -> s 1 d"), rearrange(sin, "s d -> s 1 d"), dq1, dq2, True
        )
        dk1, dk2 = dqkv[:, 1, :, :rotary_dim].chunk(2, dim=-1)
        rotary_emb.apply_rotary(
            dk1, dk2, rearrange(cos_k, "s d -> s 1 d"), rearrange(sin_k, "s d -> s 1 d"), dk1, dk2, True
        )
        return dqkv, None, None, None, None




class RotaryEmbedding(torch.nn.Module):
    """
    The rotary position embeddings from RoFormer_ (Su et. al).
    A crucial insight from the method is that the query and keys are
    transformed by rotation matrices which depend on the relative positions.

    Other implementations are available in the Rotary Transformer repo_ and in
    GPT-NeoX_, GPT-NeoX was an inspiration

    .. _RoFormer: https://arxiv.org/abs/2104.09864
    .. _repo: https://github.com/ZhuiyiTechnology/roformer
    .. _GPT-NeoX: https://github.com/EleutherAI/gpt-neox

    If scale_base > 0, this implements XPos (Sun et al., https://arxiv.org/abs/2212.10554).
    A recommended value for scale_base is 512: https://github.com/HazyResearch/flash-attention/issues/96
    Reference: https://github.com/sunyt32/torchscale/blob/main/torchscale/component/xpos_relative_position.py
    """

    def __init__(self, dim: int, base=10000, scale_base=0, device=None):
        """ """
        super().__init__()
        # Generate and save the inverse frequency buffer (non trainable)
        self.inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim))
        self.scale_base = scale_base
        self.scale = (
            (torch.arange(0, dim, 2, device=device, dtype=torch.float32) + 0.4 * dim) / (1.4 * dim)
            if scale_base > 0
            else None
        )

        self._seq_len_cached = 0
        self._cos_cached = None
        self._sin_cached = None
        self._cos_k_cached = None
        self._sin_k_cached = None

    def _update_cos_sin_cache(self, x, indexes):
        """x: (batch, seqlen, nheads, headdim) or (batch, seqlen, 3, nheads, headdim)"""
        if not isinstance(indexes, int):
            seqlen = np.array(indexes).max().item() + 1
        else:
            seqlen = indexes + 1  # eval_forward
       


        # Reset the tables if the sequence length has changed,
        # or if we're on a new device (possibly due to tracing for instance)
        if seqlen > self._seq_len_cached or self._cos_cached.device != x.device or self._cos_cached.dtype != x.dtype:
            self._seq_len_cached = seqlen
            t = torch.arange(seqlen, device=x.device, dtype=self.inv_freq.dtype)
            # Don't do einsum, it converts fp32 to fp16
            freqs = torch.outer(t, self.inv_freq.to(device=t.device))


            if self.scale is None:
                self._cos_cached = torch.cos(freqs).to(x.dtype)
                self._sin_cached = torch.sin(freqs).to(x.dtype)
            else:
                power = (
                    torch.arange(seqlen, dtype=self.scale.dtype, device=self.scale.device) - seqlen // 2
                ) / self.scale_base
                scale = self.scale.to(device=power.device) ** rearrange(power, "s -> s 1")
                # We want the multiplication by scale to happen in fp32
                self._cos_cached = (torch.cos(freqs) * scale).to(x.dtype)
                self._sin_cached = (torch.sin(freqs) * scale).to(x.dtype)
                self._cos_k_cached = (torch.cos(freqs) / scale).to(x.dtype)
                self._sin_k_cached = (torch.sin(freqs) / scale).to(x.dtype)

    def forward(self, qkv: torch.Tensor, **kwargs):
        
        return self._forward(qkv,  kwargs.pop("indexes"))



    def _forward(self, qkv: torch.Tensor, indexes=0) -> Tuple[torch.Tensor, torch.Tensor]:
        self._update_cos_sin_cache(qkv, indexes)

        if self.scale is None:
            return apply_rotary_emb_qkv_(qkv, self._cos_cached[indexes], self._sin_cached[indexes])
        else:
            return apply_rotary_emb_qkv_(
                qkv,
                self._cos_cached[indexes],
                self._sin_cached[indexes],
                self._cos_k_cached[indexes],
                self._sin_k_cached[indexes],
            )

    def _eval_forward(self, qkv, seqlen_offset=0):
        """
        seqlen_offset: can be used in generation where the qkv being passed in is only the last
        token in the batch.
        """
        self._update_cos_sin_cache(qkv, seqlen_offset + qkv.shape[1])
        if self.scale is None:
            return legacy_apply_rotary_embed_qkv(
                qkv, self._cos_cached[seqlen_offset:], self._sin_cached[seqlen_offset:]
            )
        else:
            return legacy_apply_rotary_embed_qkv(
                qkv,
                self._cos_cached[seqlen_offset:],
                self._sin_cached[seqlen_offset:],
                self._cos_k_cached[seqlen_offset:],
                self._sin_k_cached[seqlen_offset:],
            )

    def _single_forward(self, x, indexes=0):
        assert self.scale is None
        self._update_cos_sin_cache(x, indexes)
        x = x[None, ...]
        ret = legacy_apply_rotary_embed(x, self._cos_cached[indexes], self._sin_cached[indexes]).squeeze(0)
        return ret

    def _single_eval_forward(self, x, seqlen_offset=0):
        assert self.scale is None
        self._update_cos_sin_cache(x, seqlen_offset + x.shape[1])
        return legacy_apply_rotary_embed(x, self._cos_cached[seqlen_offset:], self._sin_cached[seqlen_offset:])




def _rotate_half(x):
    """
    change sign so the last dimension becomes [-odd, +even]
    """
    from einops import rearrange
    x = rearrange(x, '... (j d) -> ... j d', j=2)
    x1, x2 = x.unbind(dim=-2)
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
    t = (t * freqs.cos().to(t.dtype)) + (_rotate_half(t) * freqs.sin().to(t.dtype))
    return torch.cat((t, t_pass), dim=-1)


# [metax] end of change

module_path ="megatron.core.models.common.embeddings.rotary_pos_embedding"
module_dict ={"RotaryEmbedding":RotaryEmbedding}
add_patches_module(module_path,module_dict)


apply_rotary_emb_qkv_ = ApplyRotaryEmbQKV_.apply
legacy_apply_rotary_embed_qkv = LegacyApplyRotaryEmbQKV_.apply
legacy_apply_rotary_embed = LegacyApplyRotaryEmb.apply
