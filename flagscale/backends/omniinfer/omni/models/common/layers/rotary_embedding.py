# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

# coding=utf-8
# Adapted from
# https://github.com/huggingface/transformers/blob/v4.33.2/src/transformers/models/llama/modeling_llama.py
# Copyright 2023 The vLLM team.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Rotary Positional Embeddings."""
from typing import Any, Dict, Optional, Tuple, Union
import math
from dataclasses import dataclass
import torch_npu
import torch
import torch.nn as nn
import torch.nn.functional as F

from vllm.platforms import current_platform
from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding as GPURotaryEmbedding
from vllm.model_executor.layers.rotary_embedding import YaRNScalingRotaryEmbedding as GPUYaRNScalingRotaryEmbedding
from vllm.model_executor.layers.rotary_embedding import DeepseekScalingRotaryEmbedding as DeepseekScalingRotaryEmbeddingGPU
from vllm.model_executor.layers.rotary_embedding import (_yarn_find_correction_dim,
                                            _yarn_find_correction_range,
                                            _yarn_linear_ramp_mask,
                                            _yarn_get_mscale,
                                            _rotate_neox,
                                            _rotate_gptj)


SCALE_FACTOR = 8
LOW_FREQ_FACTOR = 1
HIGH_FREQ_FACTOR = 4
OLD_CONTEXT_LEN = 8192
ROPE_ROTARY_FACTOR = 64

NEOX_ROTARY_COEFF = 2



class RotaryEmbeddingTorchNpu(torch.nn.Module):
    _compute_inv_freq = GPURotaryEmbedding._compute_inv_freq

    def __init__(self,
                 head_size: int,
                 rotary_dim: int,
                 max_position_embeddings: int = 2048,
                 base: int = 10000,
                 is_neox_style: bool = False,
                 dtype: torch.dtype = None,
                 q_hidden_size=8192):
        super().__init__()
        self.dtype = dtype if dtype is not None else torch.get_default_dtype()
        self.rotary_dim = rotary_dim
        self.max_position_embeddings = max_position_embeddings
        self.max_len = self.max_position_embeddings
        self.base = base
        self.is_neox_style = is_neox_style
        self.rotary_coeff = NEOX_ROTARY_COEFF if is_neox_style else rotary_dim

        self.head_size = head_size
        self.cos, self.sin = self._compute_cos_sin_cache()
        self.cache_cos = None
        self.cache_sin = None
        self.cache_pos_shape = None

        if self.rotary_dim != self.head_size:
            self.rotary_pos_emb_cache = self.forward_impl(self.max_len, self.rotary_dim/2)
        else:
            self.embed = F.embedding
        self.org_position = None
        self.q_cache = None
        self.k_cache = None


    def _compute_cos_sin_cache(self) -> torch.Tensor:
        """Compute the cos and sin cache."""
        inv_freq = self._compute_inv_freq(self.base).npu()
        t = torch.arange(self.max_len, device=inv_freq.device, dtype=inv_freq.dtype)
        # Adapt: adapt for ascend rope
        if self.is_neox_style:
            freqs = torch.einsum("i,j->ij", t, inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
        else:
            freqs = torch.outer(t, inv_freq).float()
            emb = torch.stack((freqs, freqs), dim=-1)
        cos = torch.cos(emb).to(dtype=torch.get_default_dtype())
        sin = torch.sin(emb).to(dtype=torch.get_default_dtype())

        return cos, sin
        # Adapt end.

    def forward_impl(
            self, seq_len: int, n_elem: int):
        """Enhanced Transformer with Rotary Position Embedding.
        Derived from: https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/
        transformers/rope/__init__.py. MIT License:
        https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/license.
        """
        theta = 1.0 / (self.base ** (torch.arange(0, n_elem, 2, dtype=self.dtype) / n_elem))
        seq_idx = torch.arange(seq_len, dtype=self.dtype)
        idx_theta = torch.outer(seq_idx, theta).float().npu()
        cache = torch.stack([torch.cos(idx_theta), torch.sin(idx_theta)], dim=-1).to(self.dtype)
        return cache

    # use small ops
    def apply_rotary_pos_emb(self, x, cos, sin):
        x1, x2 = torch.chunk(x, 2, -1)
        x_new = torch.cat((-x2, x1), dim=-1)
        output = cos * x + sin * x_new
        return output

    # use small ops for chatglm
    def apply_rotary_pos_emb_glm(self, x: torch.Tensor, rope_cache: torch.Tensor) -> torch.Tensor:
        sq, b, np, hn = x.size(0), x.size(1), x.size(2), x.size(3)
        rot_dim = rope_cache.shape[-2] * 2
        x, x_pass = x[..., :rot_dim], x[..., rot_dim:]
        rope_cache = rope_cache[:sq]
        xshaped = x.reshape(sq, -1, np, rot_dim // 2, 2)
        rope_cache = rope_cache.view(sq, -1, 1, xshaped.size(3), 2)
        x_out2 = torch.stack(
            [
                xshaped[..., 0] * rope_cache[..., 0] - xshaped[..., 1] * rope_cache[..., 1],
                xshaped[..., 1] * rope_cache[..., 0] + xshaped[..., 0] * rope_cache[..., 1],
            ],
            -1,
        )
        x_out2 = x_out2.flatten(3)
        return torch.cat((x_out2, x_pass), dim=-1)

    # adapt chatglm : dim = head_size / 2
    def _forward_chatglm(self, position_ids, query, key):
        rotary_pos_emb = self.rotary_pos_emb_cache[position_ids]
        query = query.view(*query.shape[:-1], -1, self.head_size).contiguous()
        key = key.view(*key.shape[:-1], -1, self.head_size).contiguous()

        q_embed = self.apply_rotary_pos_emb_glm(query, rotary_pos_emb)
        k_embed = self.apply_rotary_pos_emb_glm(key, rotary_pos_emb)
        return q_embed.flatten(-2), k_embed.flatten(-2)

    # use ascend_ops to deal with torch_npu.npu_apply_rotary_pos_emb last dim is not 128 bug
    def _forward_ascend_ops_and_small_ops(self, position_ids, query, key):
        cos = torch.index_select(self.cos, dim=0, index=position_ids.squeeze(1)).unsqueeze(1)
        sin = torch.index_select(self.sin, dim=0, index=position_ids.squeeze(1)).unsqueeze(1)
        query = query.view(*query.shape[:-1], -1, self.head_size).contiguous()
        key = key.view(*key.shape[:-1], -1, self.head_size).contiguous()
        cos = cos.unsqueeze(-2)
        sin = sin.unsqueeze(-2)
        q_embed = self.apply_rotary_pos_emb(query, cos, sin)
        k_embed = self.apply_rotary_pos_emb(key, cos, sin)
        return q_embed.flatten(-2), k_embed.flatten(-2)

    # use torch_npu fused ops
    def _forward_fused_ops(self, position_ids, query, key):
        if query.shape[1] > 1:
            if self.cache_cos is None or position_ids.shape != self.cache_pos_shape:
                cos = self.embed(position_ids, self.cos)
                sin = self.embed(position_ids, self.sin)
                self.cache_cos = cos
                self.cache_sin = sin
                self.cache_pos_shape = position_ids.shape
            else:
                cos = self.cache_cos
                sin = self.cache_sin
        else:
            cos = torch.index_select(self.cos, dim=0, index=position_ids.squeeze(1)).unsqueeze(1)
            sin = torch.index_select(self.sin, dim=0, index=position_ids.squeeze(1)).unsqueeze(1)
        # head_dim use class variable, repair head_dim convert to symbol in dynamo
        query = query.view(*query.shape[:-1], -1, self.head_size).contiguous()
        key = key.view(*key.shape[:-1], -1, self.head_size).contiguous()
        cos = cos.unsqueeze(-2) # rotary_pos_emb api need cos/sin 4d input. only support BSND
        sin = sin.unsqueeze(-2)
        # npu_apply_rotary_pos_emb replace npu_rotary_mul, npu_rotary_mul will not support muti batch size
        q_embed, k_embed = torch_npu.npu_apply_rotary_pos_emb(query, key, cos, sin)
        return q_embed.flatten(-2), k_embed.flatten(-2)

    def forward(self, position_ids, query, key):
        # adapt chatglm : dim = head_size / 2
        if self.rotary_dim < self.head_size:
            q_embed, k_embed = self._forward_chatglm(position_ids, query, key)
        elif self.rotary_dim != 128:
            # use ascend_ops to deal with torch_npu.npu_apply_rotary_pos_emb last dim is not 128 bug
            q_embed, k_embed = self._forward_ascend_ops_and_small_ops(position_ids, query, key)
        else:
            q_embed, k_embed = self._forward_fused_ops(position_ids, query, key)
        return q_embed, k_embed



class YaRNScalingRotaryEmbedding(RotaryEmbeddingTorchNpu):
    """RotaryEmbedding extended with YaRN method.

    Credits to Peng et al. github.com/jquesnelle/yarn
    """

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: int,
        is_neox_style: bool,
        scaling_factor: float,
        dtype: torch.dtype,
        *,
        extrapolation_factor: float = 1,
        attn_factor: float = 1,
        beta_fast: int = 32,
        beta_slow: int = 1,
        q_hidden_size: int = 8192,
    ) -> None:
        self.scaling_factor = scaling_factor
        self.extrapolation_factor = extrapolation_factor
        self.attn_factor = attn_factor
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow
        # Get n-d magnitude scaling corrected for interpolation
        self.mscale = float(
            _yarn_get_mscale(self.scaling_factor) * self.attn_factor)
        super().__init__(head_size, rotary_dim, max_position_embeddings, base,
                         is_neox_style, dtype, q_hidden_size)

    _compute_inv_freq = GPUYaRNScalingRotaryEmbedding._compute_inv_freq

    def _compute_cos_sin_cache(self) -> torch.Tensor:
        inv_freq = self._compute_inv_freq(self.scaling_factor).npu()
        self.max_len = self.max_position_embeddings * self.scaling_factor
        t = torch.arange(self.max_len, device=inv_freq.device, dtype=inv_freq.dtype)
        # Adapt: adapt for ascend rope
        if self.is_neox_style:
            freqs = torch.einsum("i,j->ij", t, inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
        else:
            freqs = torch.outer(t, inv_freq).float()
            emb = torch.stack((freqs, freqs), dim=-1)
            emb = emb.reshape(emb.shape[0], -1)

        emb_cos = torch.cos(emb) * self.mscale
        emb_sin = torch.sin(emb) * self.mscale
        cos = emb_cos.to(dtype=torch.get_default_dtype())
        sin = emb_sin.to(dtype=torch.get_default_dtype())
        return cos, sin
        # Adapt end.


class LinearScalingRotaryEmbedding(RotaryEmbeddingTorchNpu):
    """RotaryEmbedding extended with linear scaling.

    Credits to the Reddit user /u/kaiokendev
    """

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: int,
        is_neox_style: bool,
        scaling_factor: float,
        dtype: torch.dtype,
        q_hidden_size: int = 8192,
    ) -> None:
        self.scaling_factor = scaling_factor
        super().__init__(head_size, rotary_dim, max_position_embeddings, base,
                         is_neox_style, dtype, q_hidden_size)

    def _compute_cos_sin_cache(self) -> torch.Tensor:
        inv_freq = self._compute_inv_freq(self.base).npu()
        self.max_len = self.max_position_embeddings * self.scaling_factor
        t = torch.arange(self.max_len, device=inv_freq.device, dtype=inv_freq.dtype)
        t = t / self.scaling_factor
        # Adapt: adapt for ascend rope
        if self.is_neox_style:
            freqs = torch.einsum("i,j->ij", t, inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
        else:
            freqs = torch.outer(t, inv_freq).float()
            emb = torch.stack((freqs, freqs), dim=-1)
            emb = emb.reshape(emb.shape[0], -1)
        cos = torch.cos(emb).to(dtype=torch.get_default_dtype())
        sin = torch.sin(emb).to(dtype=torch.get_default_dtype())
        return cos, sin
        # Adapt end.


class ExtendedRotaryEmbedding(RotaryEmbeddingTorchNpu):

    def _compute_inv_freq(self, base: Union[int, float]) -> torch.Tensor:
        inv_freqs = super()._compute_inv_freq(base)
        return self.apply_scaling(inv_freqs)

    def apply_scaling(self, freqs: torch.Tensor):
        low_freq_wavelen = OLD_CONTEXT_LEN / LOW_FREQ_FACTOR
        high_freq_wavelen = OLD_CONTEXT_LEN / HIGH_FREQ_FACTOR
        new_freqs = []
        for freq in freqs:
            wavelen = 2 * math.pi / freq
            if wavelen < high_freq_wavelen:
                new_freqs.append(freq)
            elif wavelen > low_freq_wavelen:
                new_freqs.append(freq / SCALE_FACTOR)
            else:
                smooth = (OLD_CONTEXT_LEN / wavelen - LOW_FREQ_FACTOR) / (
                    HIGH_FREQ_FACTOR - LOW_FREQ_FACTOR)
                new_freqs.append((1 - smooth) * freq / SCALE_FACTOR +
                                 smooth * freq)
        return torch.tensor(new_freqs, dtype=freqs.dtype, device=freqs.device)

class DeepseekScalingRotaryEmbedding(DeepseekScalingRotaryEmbeddingGPU):
    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: int,
        is_neox_style: bool,
        scaling_factor: float,
        dtype: torch.dtype,
        *,
        extrapolation_factor: float = 1,
        attn_factor: float = 1,
        beta_fast: int = 32,
        beta_slow: int = 1,
        mscale: float = 1,
        mscale_all_dim: float = 0,
    ) -> None:
        super().__init__(head_size, rotary_dim, max_position_embeddings, base,
                         is_neox_style, scaling_factor, dtype, extrapolation_factor=extrapolation_factor,
                         attn_factor=attn_factor,beta_fast=beta_fast,beta_slow=beta_slow,
                         mscale=mscale, mscale_all_dim=mscale_all_dim)

        self._set_cos_sin_cache(
            seq_len=max_position_embeddings,
            device=current_platform.device_type,
            dtype=torch.get_default_dtype(),
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        dim = self.rotary_dim

        freq_extra = 1.0 / (
            self.base
            ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim)
        )
        freq_inter = 1.0 / (
            self.scaling_factor
            * self.base
            ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim)
        )

        low, high = _yarn_find_correction_range(
            self.beta_fast,
            self.beta_slow,
            dim,
            self.base,
            self.max_position_embeddings,
        )
        inv_freq_mask = 1.0 - _yarn_linear_ramp_mask(low, high, dim // 2, dtype=torch.float32).to(
            device=device
        )
        inv_freq = freq_inter * (1 - inv_freq_mask) + freq_extra * inv_freq_mask
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        t = torch.arange(seq_len * self.scaling_factor, device=device, dtype=torch.float32)

        freqs = torch.outer(t, inv_freq)

        # _mscale = float(
        #     yarn_get_mscale(self.scaling_factor, self.mscale)
        #     / yarn_get_mscale(self.scaling_factor, self.mscale_all_dim)
        # )

        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer(
            "cos_cached", (emb.cos() * self.mscale).to(dtype), persistent=False
        )
        self.register_buffer(
            "sin_cached", (emb.sin() * self.mscale).to(dtype), persistent=False
        )

    def _compute_inv_freq(self, scaling_factor: float) -> torch.Tensor:
        pos_freqs = self.base ** (torch.arange(
            0, self.rotary_dim, 2, dtype=torch.float, device=current_platform.device_type) /
                                  self.rotary_dim)
        inv_freq_extrapolation = 1.0 / pos_freqs
        inv_freq_interpolation = 1.0 / (scaling_factor * pos_freqs)

        low, high = _yarn_find_correction_range(self.beta_fast, self.beta_slow,
                                                self.rotary_dim, self.base,
                                                self.max_position_embeddings)
        # Get n-d rotational scaling corrected for extrapolation
        inv_freq_mask = (1 - _yarn_linear_ramp_mask(
            low, high, self.rotary_dim // 2,
            dtype=torch.float)) * self.extrapolation_factor
        inv_freq_mask = inv_freq_mask.npu()
        inv_freq = inv_freq_interpolation * (
                1 - inv_freq_mask) + inv_freq_extrapolation * inv_freq_mask
        return inv_freq

    def _compute_cos_sin_cache(self) -> torch.Tensor:
        inv_freq = self._compute_inv_freq(self.scaling_factor)
        t = torch.arange(self.max_position_embeddings * self.scaling_factor,
                         device=inv_freq.device,
                         dtype=torch.float32)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = (freqs.cos() * self.mscale)
        sin = (freqs.sin() * self.mscale)
        if self.is_neox_style:
            cos = cos.repeat(1, 2)
            sin = sin.repeat(1, 2)
        else:
            cos = cos.repeat_interleave(2, dim=-1)
            sin = sin.repeat_interleave(2, dim=-1)
        cache = torch.cat((cos, sin), dim=-1)
        return cache

    def get_cos_sin(self, positions: torch.Tensor, offsets: Optional[torch.Tensor] = None):
        cos = self.cos_cached[torch.add(positions, offsets)
        if offsets is not None else positions]
        sin = self.sin_cached[torch.add(positions, offsets)
        if offsets is not None else positions]
        cos = cos.view(-1, 1, 1, cos.shape[-1])
        sin = sin.view(-1, 1, 1, sin.shape[-1])
        return cos, sin

    def forward(
            self,
            positions: torch.Tensor,
            query: torch.Tensor,
            key: torch.Tensor,
            offsets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """PyTorch-native implementation equivalent to forward()."""
        # adapt use split_rope_cat when deepseek_yarn rope rotary_dim = 64
        bs, _, hidden_size = query.shape

        cos_sin = self.cos_sin_cache[torch.add(positions, offsets)
        if offsets is not None else positions]
        cos, sin = cos_sin.chunk(2, dim=-1)
        # Adapt: adapt cos and sin shape
        cos = cos.view(-1, 1, cos.shape[-1])
        sin = sin.view(-1, 1, sin.shape[-1])
        # Adapt end.
        rotate_fn = _rotate_neox if self.is_neox_style else _rotate_gptj
        query_rot = query * cos + rotate_fn(query) * sin
        if key is not None:
            key_rot = key * cos + rotate_fn(key) * sin

        query = query_rot
        key = key_rot
        return query, key


_ROPE_DICT: Dict[Tuple, nn.Module] = {}


def get_rope(
        head_size: int,
        rotary_dim: int,
        max_position: int,
        base: int,
        is_neox_style: bool = True,
        rope_scaling: Optional[Dict[str, Any]] = None,
        dtype: Optional[torch.dtype] = None,
):
    if dtype is None:
        dtype = torch.get_default_dtype()
    key = (head_size, rotary_dim, max_position, base, is_neox_style,
           tuple(rope_scaling.items()) if rope_scaling is not None else None)
    if key in _ROPE_DICT:
        return _ROPE_DICT[key]
    # Adapt:
    # 1. do not support su
    # 2. support llama3.1 and deepseek_v2
    if rope_scaling is not None:
        # adapt Replacing legacy 'type' key with 'rope_type' in 0.6.3
        scaling_type = rope_scaling["rope_type"]
        if scaling_type != "su":
            scaling_factor = rope_scaling["factor"]
        if scaling_type == "linear":
            rotary_emb = LinearScalingRotaryEmbedding(head_size, rotary_dim,
                                                      max_position, base,
                                                      is_neox_style,
                                                      scaling_factor, dtype)
        elif scaling_type == "yarn":
            original_max_position = rope_scaling[
                "original_max_position_embeddings"]
            extra_kwargs = {
                k: v
                for k, v in rope_scaling.items()
                if k in ("extrapolation_factor", "attn_factor", "beta_fast",
                         "beta_slow")
            }
            rotary_emb = YaRNScalingRotaryEmbedding(head_size, rotary_dim,
                                                    original_max_position,
                                                    base, is_neox_style,
                                                    scaling_factor, dtype,
                                                    **extra_kwargs)
        elif scaling_type == "deepseek_yarn":
            original_max_position = rope_scaling[
                "original_max_position_embeddings"]
            extra_kwargs = {
                k: v
                for k, v in rope_scaling.items()
                if k in ("extrapolation_factor", "attn_factor", "beta_fast",
                         "beta_slow", "mscale", "mscale_all_dim")
            }
            rotary_emb = DeepseekScalingRotaryEmbedding(
                head_size, rotary_dim, original_max_position, base,
                is_neox_style, scaling_factor, dtype, **extra_kwargs)
        elif scaling_type == "llama3":
            rotary_emb = ExtendedRotaryEmbedding(head_size, rotary_dim,
                                                     max_position, base,
                                                     is_neox_style, dtype)
        else:
            scaling_type = rope_scaling["type"]
            raise ValueError(f"Unknown RoPE scaling type {scaling_type}, only support linear and dynamic now")
    else:
        rotary_emb = RotaryEmbeddingTorchNpu(head_size, rotary_dim, max_position, base,
                                             is_neox_style)
    _ROPE_DICT[key] = rotary_emb
    return rotary_emb

