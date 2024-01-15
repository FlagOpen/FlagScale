from typing import Optional
import torch
import megatron
from megatron.model.enums import AttnMaskType, LayerType, AttnType
from megatron.model.module import MegatronModule
from megatron.model.transformer import bias_dropout_add


class CoreAttention(MegatronModule):
    def __init__(self, layer_number, config,
                 attn_mask_type=AttnMaskType.padding):
        super(CoreAttention, self).__init__()

    def forward(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask
    ):
        input_size = (
            query_states[0], 
            query_states[1],
            query_states[2],
            query_states[3]
        )
        output_size = (
            query_states.size(0),
            query_states.size(1),
            query_states.size(2) * query_states.size(3)
        ) #seq_len, batch_size, head_num * head_dim

        query_states = query_states.permute(1, 2, 0, 3)
        key_states = key_states.permute(1, 2, 0, 3)
        value_states = value_states.permute(1, 2, 0, 3)

        bsz, num_heads, q_len, head_dim = query_states.size()
        kv_seq_len = key_states.size(2)

        with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False):
            attn_output = torch.nn.functional.scaled_dot_product_attention(
                query_states,#batch_size, head_num , seq_len,  head_size
                key_states,#batch_size * head_num * seq_len * head_size
                value_states,#batch_size * head_num * seq_len * head_size
                attn_mask=attention_mask,# bsz * num_heads, q_len, kv_seq_len
                dropout_p=0.0,
                is_causal=False,
            )

        if attn_output.size() != (bsz, num_heads, q_len, head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, num_heads, q_len, head_dim)}, but is"
                f" {attn_output.size()}"
            )
        attn_output = attn_output.permute(2, 0, 1, 3).reshape(*output_size)
        return attn_output


class FlashSelfAttention(MegatronModule):
    def __init__(self, layer_number, config,
                 attn_mask_type=AttnMaskType.padding):
        super(FlashSelfAttention, self).__init__()

    def forward(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask
    ):
        output_size = (
            query_states.size(0),
            query_states.size(1),
            query_states.size(2) * query_states.size(3)
        ) #seq_len, batch_size, head_num * head_dim

        query_states = query_states.permute(1, 2, 0, 3)
        key_states = key_states.permute(1, 2, 0, 3)
        value_states = value_states.permute(1, 2, 0, 3)

        bsz, num_heads, q_len, head_dim = query_states.size()
        kv_seq_len = key_states.size(2)

        with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False):
            attn_output = torch.nn.functional.scaled_dot_product_attention(
                query_states, # batch_size, head_num , seq_len,  head_size
                key_states, # batch_size, head_num, seq_len, head_size
                value_states, #batch_size, head_num, seq_len, head_size
                attn_mask=attention_mask,# bsz * num_heads, q_len, kv_seq_len
                dropout_p=0.0,
                is_causal=False,
            )

        if attn_output.size() != (bsz, num_heads, q_len, head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, num_heads, q_len, head_dim)}, but is"
                f" {attn_output.size()}"
            )
        attn_output = attn_output.permute(2, 0, 1, 3).reshape(*output_size).contiguous()
        return attn_output


def bias_dropout_add_fused_train(x: torch.Tensor,
                                 bias: Optional[torch.Tensor],
                                 residual: torch.Tensor,
                                 prob: float) -> torch.Tensor:
    return bias_dropout_add(x, bias, residual, prob, True) # TODO(mthreads)


megatron.model.transformer.CoreAttention = CoreAttention
megatron.model.transformer.FlashSelfAttention = FlashSelfAttention
megatron.model.transformer.bias_dropout_add_fused_train = bias_dropout_add_fused_train
