import copy

from typing import Union
from dataclasses import dataclass

from megatron.core import parallel_state 
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.attention import SelfAttention
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.models.common.embeddings.rotary_pos_embedding import apply_rotary_pos_emb

from .utils import _SeqAllToAll


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
        self.usp_size = parallel_state.get_ulysses_sequence_parallel_world_size()
        self.usp_group = parallel_state.get_ulysses_sequence_parallel_group()
        te_attn_config = copy.deepcopy(config)
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

            # TODO, can apply positional embedding to value_layer so it has
            # absolute positional embedding.
            # otherwise, only relative positional embedding takes effect
            # value_layer = apply_rotary_pos_emb(value_layer, k_pos_emb)

        # ================================================
        # scatter (q, k, v) along the head dimension(2) and gather along the sequence dimension(0)
        # ================================================

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

        core_attn_out = _SeqAllToAll.apply(self.usp_group, core_attn_out, 0, 2)

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









# class USPSelfAttention(Attention):
#     """
#     Refer to the USP paper (https://arxiv.org/pdf/2405.07719) for details.
#     """
#     def __init__(
#         self,
#         config: TransformerConfig,
#         submodules: USPSelfAttentionSubmodules,
#         layer_number: int,
#         attn_mask_type=AttnMaskType.padding,
#     ):

#         super().__init__(
#             config=config,
#             submodules=submodules,
#             layer_number=layer_number,
#             attn_mask_type=attn_mask_type,
#             attention_type="self",
#         )

#         te_attn_config = copy.deepcopy(config)
#         usp_size = parallel_state.get_ulysses_sequence_parallel_world_size()
#         te_attn_config.num_attention_heads = config.num_attention_heads // usp_size
#         te_attn_config.num_query_groups = config.num_query_groups // usp_size
#         self.core_attention = build_module(
#             submodules.core_attention,
#             config=te_attn_config,
#             layer_number=self.layer_number,
#             attn_mask_type=self.attn_mask_type,
#             attention_type=self.attention_type,
#         )

#         self.usp_group = parallel_state.get_ulysses_sequence_parallel_group()

#         self.linear_qkv = build_module(
#             submodules.linear_qkv,
#             self.config.hidden_size,
#             self.query_projection_size + 2 * self.kv_projection_size,
#             config=self.config,
#             init_method=self.config.init_method,
#             gather_output=False,
#             bias=self.config.add_bias_linear or self.config.add_qkv_bias,
#             skip_bias_add=False,
#             is_expert=False,
#             tp_comm_buffer_name='qkv',
#         )

#         if submodules.q_layernorm is not None:
#             self.q_layernorm = build_module(
#                 submodules.q_layernorm,
#                 hidden_size=self.hidden_size_per_attention_head,
#                 config=self.config,
#                 eps=self.config.layernorm_epsilon,
#             )
#         else:
#             self.q_layernorm = None

#         if submodules.k_layernorm is not None:
#             self.k_layernorm = build_module(
#                 submodules.k_layernorm,
#                 hidden_size=self.hidden_size_per_attention_head,
#                 config=self.config,
#                 eps=self.config.layernorm_epsilon,
#             )
#         else:
#             self.k_layernorm = None

#     def get_query_key_value_tensors(self, hidden_states, key_value_states=None):
#         """
#         Derives `query`, `key` and `value` tensors from `hidden_states`.
#         """
#         # Attention heads [sq, b, h] --> [sq, b, ng * (np/ng + 2) * hn)]
#         mixed_qkv, _ = self.linear_qkv(hidden_states)

#         # [sq, b, hp] --> [sq, b, ng, (np/ng + 2) * hn]
#         new_tensor_shape = mixed_qkv.size()[:-1] + (
#             self.num_query_groups_per_partition,
#             (
#                 (self.num_attention_heads_per_partition // self.num_query_groups_per_partition + 2)
#                 * self.hidden_size_per_attention_head
#             ),
#         )
#         mixed_qkv = mixed_qkv.view(*new_tensor_shape)

#         split_arg_list = [
#             (
#                 self.num_attention_heads_per_partition
#                 // self.num_query_groups_per_partition
#                 * self.hidden_size_per_attention_head
#             ),
#             self.hidden_size_per_attention_head,
#             self.hidden_size_per_attention_head,
#         ]

#         if SplitAlongDim is not None:

#             # [sq, b, ng, (np/ng + 2) * hn] --> [sq, b, ng, np/ng * hn], [sq, b, ng, hn], [sq, b, ng, hn]
#             (query, key, value) = SplitAlongDim(mixed_qkv, 3, split_arg_list,)
#         else:

#             # [sq, b, ng, (np/ng + 2) * hn] --> [sq, b, ng, np/ng * hn], [sq, b, ng, hn], [sq, b, ng, hn]
#             (query, key, value) = torch.split(mixed_qkv, split_arg_list, dim=3,)

#         # [sq, b, ng, np/ng * hn] -> [sq, b, np, hn]
#         query = query.reshape(query.size(0), query.size(1), -1, self.hidden_size_per_attention_head)

#         if self.q_layernorm is not None:
#             query = self.q_layernorm(query)

#         if self.k_layernorm is not None:
#             key = self.k_layernorm(key)

#         if self.config.test_mode:
#             self.run_realtime_tests()

#         return query, key, value

#     def forward(
#         self,
#         hidden_states,
#         attention_mask,
#         key_value_states=None,
#         inference_params=None,
#         rotary_pos_emb=None,
#         packed_seq_params=None,
#     ):
#         # hidden_states: [sq, b, h]

#         # For self attention we just duplicate the rotary_pos_emb if it isn't already
#         if rotary_pos_emb is not None and not isinstance(rotary_pos_emb, tuple):
#             rotary_pos_emb = (rotary_pos_emb,) * 2

#         # =====================
#         # Query, Key, and Value
#         # =====================
#         # Get the query, key and value tensors based on the type of attention -
#         # self or cross attn.
#         query, key, value = self.get_query_key_value_tensors(hidden_states, key_value_states)

#         # ===================================================
#         # Adjust key, value, and rotary_pos_emb for inference
#         # ===================================================
#         key, value, rotary_pos_emb, attn_mask_type = self._adjust_key_value_for_inference(
#             inference_params, key, value, rotary_pos_emb
#         )

#         if packed_seq_params is not None:
#             query = query.squeeze(1)
#             key = key.squeeze(1)
#             value = value.squeeze(1)

#         # ================================================
#         # relative positional embedding (rotary embedding)
#         # ================================================
#         if rotary_pos_emb is not None:
#             q_pos_emb, k_pos_emb = rotary_pos_emb

#             if packed_seq_params is not None:
#                 cu_seqlens_q = packed_seq_params.cu_seqlens_q
#                 cu_seqlens_kv = packed_seq_params.cu_seqlens_kv
#             else:
#                 cu_seqlens_q = cu_seqlens_kv = None
#             query = apply_rotary_pos_emb(
#                 query, q_pos_emb, config=self.config, cu_seqlens=cu_seqlens_q,
#             )
#             key = apply_rotary_pos_emb(
#                 key, k_pos_emb, config=self.config, cu_seqlens=cu_seqlens_kv,
#             )

#             # TODO, can apply positional embedding to value_layer so it has
#             # absolute positional embedding.
#             # otherwise, only relative positional embedding takes effect
#             # value_layer = apply_rotary_pos_emb(value_layer, k_pos_emb)

#         # ================================================
#         # gather q, k, v along the sequence dimension and scatter along the head dimension
#         # ================================================

#         query, key, value = [x.transpose(0, 1).contiguous() for x in (query, key, value)]
#         if value.shape == key.shape and value.shape[0] == 1 and value.stride() != key.stride():
#             value = value.as_strided(value.shape, key.stride())

#         query = _SeqAllToAll.apply(self.usp_group, query, 2, 1)
#         key = _SeqAllToAll.apply(self.usp_group, key, 2, 1)
#         value = _SeqAllToAll.apply(self.usp_group, value, 2, 1)

#         if not self.core_attention.is_bshd_format():
#             # convert to sbhd 
#             query, key, value = [x.transpose(0, 1).contiguous() for x in (query, key, value)]

#         # ==================================
#         # core attention computation
#         # ==================================

#         if self.checkpoint_core_attention and self.training:
#             core_attn_out = self._checkpointed_attention_forward(
#                 query,
#                 key,
#                 value,
#                 attention_mask,
#                 attn_mask_type=attn_mask_type,
#                 packed_seq_params=packed_seq_params,
#             )
#         else:
#             core_attn_out = self.core_attention(
#                 query,
#                 key,
#                 value,
#                 attention_mask,
#                 attn_mask_type=attn_mask_type,
#                 packed_seq_params=packed_seq_params,
#             )

#         # ================================================
#         # gather out along the sequence dimension and scatter along the head dimension
#         # ================================================
#         core_attn_out = core_attn_out.view(query.shape)
#         if not self.core_attention.is_bshd_format():
#             core_attn_out = core_attn_out.transpose(0, 1).contiguous()

#         core_attn_out = _SeqAllToAll.apply(self.usp_group, core_attn_out, 1, 2).transpose(0, 1)
#         core_attn_out = core_attn_out.view(core_attn_out.shape[0], core_attn_out.shape[1], -1)

#         if packed_seq_params is not None:
#             # reshape to same output shape as unpacked case
#             # (t, np, hn) -> (t, b=1, h=np*hn)
#             # t is the pack size = sum (sq_i)
#             # note that batch is a dummy dimension in the packed case
#             core_attn_out = core_attn_out.reshape(core_attn_out.size(0), 1, -1)

#         # =================
#         # Output. [sq, b, h]
#         # =================

#         output, bias = self.linear_proj(core_attn_out)

#         return output, bias
