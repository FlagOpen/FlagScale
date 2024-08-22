import copy
from typing import Union, Any, Tuple
from dataclasses import dataclass

import torch
import torch.distributed

from megatron.core import parallel_state 
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.attention import SelfAttention
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.models.common.embeddings.rotary_pos_embedding import apply_rotary_pos_emb


def post_all2all(input, scatter_idx, batch_dim_idx, seq_world_size, bs, seq_len, num_head, head_dim):

    if batch_dim_idx == 0:
        # b, s, n, h
        if scatter_idx < 2:
            output = input.permute(1, 2, 0, 3, 4).contiguous()
            output = output.reshape(bs, seq_len // seq_world_size, seq_world_size * num_head,
                                    head_dim).contiguous()
        else:
            output = input.permute(1, 0, 2, 3, 4).contiguous()
            output = output.reshape(bs, seq_world_size * seq_len, num_head // seq_world_size,
                                    head_dim).contiguous()
    else:
        # s, b, n, h
        if scatter_idx < 2:
            output = input.permute(1, 2, 0, 3, 4).contiguous()
            output = output.reshape(seq_len // seq_world_size, bs, seq_world_size * num_head,
                                    head_dim).contiguous()
        else:
            output = input.reshape(seq_len * seq_world_size, bs, num_head // seq_world_size, head_dim).contiguous()
    return output


def single_all_to_all(input, scatter_idx, gather_idx, batch_dim_idx, group):
    seq_world_size = parallel_state.get_ulysses_sp_parallel_world_size()
    if batch_dim_idx == 0:
        # b, s, hc, h
        if scatter_idx < 2: # all_to_all for output or backward
            bs, global_seq_len, num_local_head, head_dim = input.shape
            input_t = input.reshape([bs, seq_world_size, global_seq_len // seq_world_size, num_local_head,
                                     head_dim]).contiguous()
            input_t = input_t.permute(1, 0, 2, 3, 4).contiguous()
        else:
            bs, local_seq_len, num_total_head, head_dim = input.shape
            assert num_total_head % seq_world_size == 0, f"Number of heads ({num_total_head}) must be divisible by the sequence parallel size ({seq_world_size})!"
            input_t = input.reshape([bs, local_seq_len, seq_world_size, num_total_head // seq_world_size,
                                     head_dim]).contiguous()
            input_t = input_t.permute(2, 0, 1, 3, 4).contiguous()
    else:
        # s, b, hc, h
        if scatter_idx < 2: # all_to_all for output or backward
            global_seq_len, bs, num_local_head, head_dim = input.shape
            input_t = input.reshape([seq_world_size, global_seq_len // seq_world_size, bs, num_local_head,
                                     head_dim]).contiguous()
        else:
            local_seq_len, bs, num_total_head, head_dim = input.shape
            assert num_total_head % seq_world_size == 0, f"Number of heads ({num_total_head}) must be divisible by the sequence parallel size ({seq_world_size})!"
            input_t = input.reshape([local_seq_len, bs, seq_world_size, num_total_head // seq_world_size,
                                     head_dim]).contiguous()
            input_t = input_t.permute(2, 0, 1, 3, 4).contiguous()

    output = torch.empty_like(input_t)
    torch.distributed.all_to_all_single(output, input_t, group=group)
    
    if scatter_idx < 2:
        res = post_all2all(output, scatter_idx, batch_dim_idx, seq_world_size, bs, global_seq_len, num_local_head,
                                        head_dim)
    else:
        res = post_all2all(output, scatter_idx, batch_dim_idx, seq_world_size, bs, local_seq_len, num_total_head,
                                        head_dim)
    return res


class _SeqAllToAll(torch.autograd.Function):

    @staticmethod
    def forward(ctx: Any,
                group,
                input: torch.Tensor,
                scatter_idx: int = 0,
                gather_idx: int = 2,
                batch_dim_idx: int = 1) -> torch.Tensor:
        ctx.group = group
        ctx.scatter_idx = scatter_idx
        ctx.gather_idx = gather_idx
        ctx.batch_dim_idx = batch_dim_idx
        res = single_all_to_all(input, scatter_idx, gather_idx, batch_dim_idx, group)
        return res

    @staticmethod
    def backward(ctx: Any, *grad_output: torch.Tensor) -> Tuple[None, torch.Tensor, None, None, None]:

        return (None,
                _SeqAllToAll.apply(ctx.group, *grad_output, ctx.gather_idx, ctx.scatter_idx, ctx.batch_dim_idx),
                None,None,None)



@dataclass
class USPSelfAttentionSubmodules:
    linear_qkv: Union[ModuleSpec, type] = None
    core_attention: Union[ModuleSpec, type] = None
    linear_proj: Union[ModuleSpec, type] = None
    q_layernorm: Union[ModuleSpec, type] = None
    k_layernorm: Union[ModuleSpec, type] = None


class USPSelfAttention(SelfAttention):
    def __init__(
        self,
        config: TransformerConfig,
        submodules: USPSelfAttentionSubmodules,
        layer_number: int,
        attn_mask_type=AttnMaskType.padding,
    ):
        super().__init__(
            config=config,
            submodules=submodules,
            layer_number=layer_number,
            attn_mask_type=attn_mask_type,
        )
        self.usp_size = parallel_state.get_ulysses_sp_parallel_world_size()
        self.usp_group = parallel_state.get_ulysses_sp_parallel_group()
        te_attn_config = copy.deepcopy(config)
        assert config.num_attention_heads % self.usp_size == 0, \
               f"num_attention_heads[{config.num_attention_heads}] can't be divisived by usp_size[{self.usp_size}]"
        assert config.num_attention_heads % self.usp_size == 0, \
               f"num_query_groups[{config.num_query_groups}] can't be divisived by usp_size[{self.usp_size}]"
        te_attn_config.num_attention_heads = config.num_attention_heads // self.usp_size
        te_attn_config.num_query_groups = config.num_query_groups // self.usp_size

        self.core_attention = build_module(
            submodules.core_attention,
            config=te_attn_config,
            layer_number=self.layer_number,
            attn_mask_type=self.attn_mask_type,
            attention_type=self.attention_type,
        )

    def forward(
        self,
        hidden_states,
        attention_mask,
        key_value_states=None,
        inference_params=None,
        rotary_pos_emb=None,
        packed_seq_params=None,
    ):
        # hidden_states: [sq, b, h]

        # For self attention we just duplicate the rotary_pos_emb if it isn't already
        if rotary_pos_emb is not None and not isinstance(rotary_pos_emb, tuple):
            rotary_pos_emb = (rotary_pos_emb,) * 2

        # =====================
        # Query, Key, and Value
        # =====================
        # Get the query, key and value tensors based on the type of attention -
        # self or cross attn.
        query, key, value = self.get_query_key_value_tensors(hidden_states, key_value_states)

        # ===================================================
        # Adjust key, value, and rotary_pos_emb for inference
        # ===================================================
        key, value, rotary_pos_emb, attn_mask_type = self._adjust_key_value_for_inference(
            inference_params, key, value, rotary_pos_emb
        )

        if packed_seq_params is not None:
            query = query.squeeze(1)
            key = key.squeeze(1)
            value = value.squeeze(1)

        # ================================================
        # relative positional embedding (rotary embedding)
        # ================================================
        if rotary_pos_emb is not None:
            q_pos_emb, k_pos_emb = rotary_pos_emb

            if packed_seq_params is not None:
                cu_seqlens_q = packed_seq_params.cu_seqlens_q
                cu_seqlens_kv = packed_seq_params.cu_seqlens_kv
            else:
                cu_seqlens_q = cu_seqlens_kv = None
            query = apply_rotary_pos_emb(
                query, q_pos_emb, config=self.config, cu_seqlens=cu_seqlens_q,
            )
            key = apply_rotary_pos_emb(
                key, k_pos_emb, config=self.config, cu_seqlens=cu_seqlens_kv,
            )

        query = _SeqAllToAll.apply(self.usp_group, query, 2, 0)
        key = _SeqAllToAll.apply(self.usp_group, key, 2, 0)
        value = _SeqAllToAll.apply(self.usp_group, value, 2, 0)

        # ==================================
        # core attention computation
        # ==================================

        if self.checkpoint_core_attention and self.training:
            core_attn_out = self._checkpointed_attention_forward(
                query,
                key,
                value,
                attention_mask,
                attn_mask_type=attn_mask_type,
                packed_seq_params=packed_seq_params,
            )
        else:
            core_attn_out = self.core_attention(
                query,
                key,
                value,
                attention_mask,
                attn_mask_type=attn_mask_type,
                packed_seq_params=packed_seq_params,
            )

        # ================================================
        # scatter out along the sequence dimension(0) and gather along the head dimension(2)
        # ================================================
        
        core_attn_out = core_attn_out.view(query.shape)
        core_attn_out = _SeqAllToAll.apply(self.usp_group, core_attn_out, 0, 2)
        core_attn_out = core_attn_out.view(*core_attn_out.shape[:2], -1)
        
        # =================
        # Output. [sq, b, h]
        # =================

        if packed_seq_params is not None:
            # reshape to same output shape as unpacked case
            # (t, np, hn) -> (t, b=1, h=np*hn)
            # t is the pack size = sum (sq_i)
            # note that batch is a dummy dimension in the packed case
            core_attn_out = core_attn_out.reshape(core_attn_out.size(0), 1, -1)

        output, bias = self.linear_proj(core_attn_out)

        return output, bias