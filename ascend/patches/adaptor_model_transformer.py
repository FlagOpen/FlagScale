import os
import math
import torch
import torch.nn.functional as F
import megatron
from megatron import get_args, core
from megatron.core import mpu, tensor_parallel
from megatron.model.enums import AttnMaskType, AttnType
from megatron.model.transformer import ParallelAttention, CoreAttention
from megatron.core.models.common.rotary_pos_embedding import apply_rotary_pos_emb
from megatron.model.transformer import ParallelMLP
from megatron.model.utils import init_method_normal
from megatron.model.utils import openai_gelu, erf_gelu
import torch_npu

try:
    from einops import rearrange
except ImportError:
    rearrange = None


class FlashSelfAttention(torch.nn.Module):
    """Implement the scaled dot product attention with softmax.
    Arguments
    ---------
        softmax_scale: The temperature to use for the softmax attention.
                      (default: 1/sqrt(d_keys) where d_keys is computed at
                      runtime)
        attention_dropout: The dropout rate to apply to the attention
                           (default: 0.0)
    """

    def __init__(self, causal=False, softmax_scale=None, attention_dropout=0.0,
                 device=None, dtype=None, pre_tokens=None, next_tokens=None, shape_order='SBH'):
        super().__init__()
        assert rearrange is not None, 'Please install einops first, e.g., with pip install einops'
        self.causal = causal
        self.softmax_scale = softmax_scale
        self.dropout_p = attention_dropout
        self.pre_tokens = pre_tokens
        self.next_tokens = next_tokens
        self.shape_order = shape_order


    def forward(self, q, k, v, n, attention_mask):
        scale = 1.0/math.sqrt(q.size(2)/n) if self.softmax_scale is None else self.softmax_scale
        output = torch_npu.npu_fusion_attention(
            q, k, v, n, input_layout=self.shape_order,
            pse=None,
            padding_mask=None,
            atten_mask=attention_mask,
            scale=scale,
            pre_tockens=self.pre_tokens,
            next_tockens=self.next_tokens,
            keep_prob=1.,
            inner_precise=0
        )[0]        
        return output


def ParallelAttentionInit(self, config, layer_number,
                 attention_type=AttnType.self_attn,
                 attn_mask_type=AttnMaskType.padding):
    super(ParallelAttention, self).__init__()
    args = get_args()
    self.layer_number = max(1, layer_number)
    self.attention_type = attention_type
    self.attn_mask_type = attn_mask_type
    self.params_dtype = config.params_dtype
    self.sequence_parallel = config.sequence_parallel
    self.rotary_interleaved_patch = args.rotary_interleaved_patch
    self.shape_order = args.npu_fa_shape_order

    self.group_query_attention = args.group_query_attention
    self.num_query_groups = args.num_query_groups

    query_projection_size = config.kv_channels * config.num_attention_heads
    if self.group_query_attention:
        kv_projection_size = args.kv_channels * args.num_query_groups
    else:
        kv_projection_size = args.kv_channels * args.num_attention_heads

    self.use_flash_attn = args.use_flash_attn \
        and attention_type == AttnType.self_attn \
        and self.attn_mask_type == AttnMaskType.causal
    if self.use_flash_attn:
        assert attention_type == AttnType.self_attn, ('FlashAttention code path only supports '
                                                        'self-attention for now')
        assert self.attn_mask_type == AttnMaskType.causal, ('FlashAttention code path only '
                                                            'supports causal mask for now')
        if rearrange is None:
            raise ImportError('einops is not installed, please install with pip install einops')

    # Per attention head and per partition values.
    world_size = mpu.get_tensor_model_parallel_world_size()
    self.hidden_size_per_attention_head = core.utils.divide(
        query_projection_size, config.num_attention_heads)
    self.num_attention_heads_per_partition = core.utils.divide(
        config.num_attention_heads, world_size)

    if self.group_query_attention:
        if args.num_query_groups % world_size != 0:
            raise NotImplementedError('Currently the num_query_groups should be '
                                        'a multiple of the tensor parallel size')
        self.num_query_groups_per_partition = core.utils.divide(
                    args.num_query_groups, world_size)
    else:
        self.num_query_groups_per_partition = self.num_attention_heads_per_partition

    # TODO
    if args.apply_init_customized:
        init_method_attn_q = init_method_normal(
            args.init_method_std_scaled_attn_q[self.layer_number-1])
        init_method_attn_k = init_method_normal(
            args.init_method_std_scaled_attn_k[self.layer_number-1])
        init_method_attn_v = init_method_normal(
            args.init_method_std_scaled_attn_v[self.layer_number-1])

    # Strided linear layer.
    if attention_type == AttnType.self_attn:
        if args.mup is None:
            self.query_key_value = tensor_parallel.ColumnParallelLinear(
                config.hidden_size,
                query_projection_size + 2 * kv_projection_size,
                config=config,
                init_method=config.init_method,
                bias=args.add_bias_linear,
                gather_output=False)
            if args.apply_init_customized:
                with tensor_parallel.get_cuda_rng_tracker().fork():
                    # [ng, (np/ng + 2), hn, h]
                    tmp =  self.num_attention_heads_per_partition // self.num_query_groups_per_partition
                    new_tensor_shape = (self.num_query_groups_per_partition,
                                        tmp + 2,
                                        self.hidden_size_per_attention_head,
                                        self.query_key_value.weight.size()[-1])

                    wq = self.query_key_value.weight.view(new_tensor_shape)[:, 0:tmp        :, :]
                    wk = self.query_key_value.weight.view(new_tensor_shape)[:, tmp:tmp+1    :, :]
                    wv = self.query_key_value.weight.view(new_tensor_shape)[:, tmp+1:tmp+2, :, :]

                    init_method_attn_q(wq)
                    init_method_attn_k(wk)
                    init_method_attn_v(wv)
                if torch.distributed.get_rank() == 0:
                    print('Override ParallelAttention init_method.', flush=True)
        else:
            self.query = tensor_parallel.ColumnParallelLinear(
                config.hidden_size,
                query_projection_size,
                config=config,
                init_method=config.init_method,
                bias=args.add_bias_linear,
                gather_output=False)
            self.key = tensor_parallel.ColumnParallelLinear(
                config.hidden_size,
                kv_projection_size,
                config=config,
                init_method=config.init_method,
                bias=args.add_bias_linear,
                gather_output=False)
            self.value = tensor_parallel.ColumnParallelLinear(
                config.hidden_size,
                kv_projection_size,
                config=config,
                init_method=config.init_method,
                bias=args.add_bias_linear,
                gather_output=False)
        self.mup = args.mup
    else:
        assert attention_type == AttnType.cross_attn

        if self.group_query_attention:
            raise NotImplementedError("Grouped query attention not implemented for cross-attention.")
        assert query_projection_size == kv_projection_size

        self.query = tensor_parallel.ColumnParallelLinear(
            config.hidden_size,
            query_projection_size,
            config=config,
            init_method=config.init_method,
            bias=config.add_bias_linear,
            gather_output=False)

        self.key_value = tensor_parallel.ColumnParallelLinear(
            config.hidden_size,
            2 * kv_projection_size,
            config=config,
            init_method=config.init_method,
            bias=config.add_bias_linear,
            gather_output=False)

    self.core_attention = CoreAttention(self.layer_number, config,
                                        self.attn_mask_type)
    self.checkpoint_core_attention = config.recompute_granularity == 'selective'

    if self.use_flash_attn:
        self.core_attention_flash = FlashSelfAttention(
            causal=True, attention_dropout=args.attention_dropout,
            pre_tokens=args.npu_fa_pre_tokens, next_tokens=args.npu_fa_next_tokens,
            shape_order=args.npu_fa_shape_order
        )

    # Output.
    self.dense = tensor_parallel.RowParallelLinear(
        query_projection_size,
        config.hidden_size,
        config=config,
        init_method=config.output_layer_init_method,
        bias=args.add_bias_linear,
        input_is_parallel=True,
        skip_bias_add=True)

    if args.mup_coord_check:
        self.query_no_op = torch.nn.Identity() # just for coordcheck
        self.key_no_op = torch.nn.Identity() # just for coordcheck
        self.value_no_op = torch.nn.Identity() # just for coordcheck
        self.mup_coord_check = True
    else:
        self.mup_coord_check = False


def ParallelAttentionForward(self, hidden_states, attention_mask,
            encoder_output=None, inference_params=None,
            rotary_pos_emb=None):
    # hidden_states: [sq, b, h]

    # =================================================
    # Pre-allocate memory for key-values for inference.
    # =================================================
    is_first_step = False
    if inference_params:
        if self.layer_number not in inference_params.key_value_memory_dict:
            inf_max_seq_len = inference_params.max_sequence_length
            inf_max_batch_size = inference_params.max_batch_size
            inference_key_memory = self._allocate_memory(
                inf_max_seq_len, inf_max_batch_size,
                self.num_query_groups_per_partition)
            inference_value_memory = self._allocate_memory(
                inf_max_seq_len, inf_max_batch_size,
                self.num_query_groups_per_partition)

            inference_params.key_value_memory_dict[self.layer_number] = (
                inference_key_memory, inference_value_memory)
            is_first_step = True
        else:
            inference_key_memory, inference_value_memory = \
                inference_params.key_value_memory_dict[self.layer_number]

    # =====================
    # Query, Key, and Value
    # =====================
    if self.attention_type == AttnType.self_attn:
        if self.mup is None:
            # Attention heads [sq, b, h] --> [sq, b, ng * (np/ng + 2) * hn)]
            mixed_x_layer, _ = self.query_key_value(hidden_states)

            # [sq, b, hp] --> [sq, b, ng, (np/ng + 2) * hn]
            new_tensor_shape = mixed_x_layer.size()[:-1] + (
                self.num_query_groups_per_partition,
                (
                    (self.num_attention_heads_per_partition // self.num_query_groups_per_partition + 2)
                    * self.hidden_size_per_attention_head
                ),
            )
            mixed_x_layer = mixed_x_layer.view(*new_tensor_shape)

            # [sq, b, ng, (np/ng + 2) * hn] --> [sq, b, ng, np/ng * hn], [sq, b, ng, hn], [sq, b, ng, hn]
            (query_layer,
            key_layer,
            value_layer) = torch.split(
                mixed_x_layer,
                [
                    (
                        self.num_attention_heads_per_partition // self.num_query_groups_per_partition
                        * self.hidden_size_per_attention_head
                    ),
                    self.hidden_size_per_attention_head,
                    self.hidden_size_per_attention_head
                ],
                dim=3)
            # [sq, b, ng, np/ng * hn] -> [sq, b, np, hn] -
            query_layer = query_layer.reshape(query_layer.size(0), query_layer.size(1), -1, self.hidden_size_per_attention_head)
        else:
            query_layer, _ = self.query(hidden_states)
            key_layer, _ = self.key(hidden_states)
            value_layer, _ = self.value(hidden_states)
            new_tensor_shape1 = query_layer.size()[:-1] + \
                (self.num_attention_heads_per_partition,
                    self.hidden_size_per_attention_head)
            query_layer = query_layer.view(new_tensor_shape1)
            new_tensor_shape2 = key_layer.size()[:-1] + \
                (self.num_query_groups_per_partition,
                    self.hidden_size_per_attention_head)
            key_layer = key_layer.view(new_tensor_shape2)
            value_layer = value_layer.view(new_tensor_shape2)

        if self.mup_coord_check:
            query_layer = self.query_no_op(query_layer)
            key_layer = self.key_no_op(key_layer)
            value_layer = self.value_no_op(value_layer)
    else:
        # Attention heads [sk, b, h] --> [sk, b, (np * 2 * hn)]
        mixed_kv_layer, _ = self.key_value(encoder_output)

        # [sk, b, (np * 2 * hn)] --> [sk, b, np, 2 * hn]
        new_tensor_shape = mixed_kv_layer.size()[:-1] + \
            (self.num_attention_heads_per_partition,
            2 * self.hidden_size_per_attention_head)
        mixed_kv_layer = mixed_kv_layer.view(*new_tensor_shape)

        # [sk, b, np, 2 * hn] --> 2 [sk, b, np, hn]
        (key_layer,
        value_layer) = tensor_parallel.split_tensor_along_last_dim(mixed_kv_layer, 2)

        # Attention head [sq, b, h] --> [sq, b, hp]
        query_layer, _ = self.query(hidden_states)
        # [sq, b, hp] --> [sq, b, np, hn]
        new_tensor_shape = query_layer.size()[:-1] + \
            (self.num_attention_heads_per_partition,
            self.hidden_size_per_attention_head)
        query_layer = query_layer.view(*new_tensor_shape)

    # ==================================
    # Adjust key and value for inference
    # ==================================

    # duplicate the pos_emb for self attention
    if rotary_pos_emb is not None:
        if isinstance(rotary_pos_emb, tuple):
            rotary_pos_emb = rotary_pos_emb
        else:
            rotary_pos_emb = ((rotary_pos_emb,) * 2)

    if inference_params:
        batch_start = inference_params.batch_size_offset
        batch_end = batch_start + key_layer.size(1)
        assert batch_end <= inference_key_memory.size(1)
        sequence_start = inference_params.sequence_len_offset
        sequence_end = sequence_start + key_layer.size(0)
        assert sequence_end <= inference_key_memory.size(0)
        # Copy key and values.
        inference_key_memory[sequence_start:sequence_end,
                                batch_start:batch_end, ...] = key_layer
        inference_value_memory[sequence_start:sequence_end,
                                batch_start:batch_end, ...] = value_layer
        key_layer = inference_key_memory[
            :sequence_end, batch_start:batch_end, ...]
        value_layer = inference_value_memory[
            :sequence_end, batch_start:batch_end, ...]


        # adjust the key rotary positional embedding
        if rotary_pos_emb is not None:
            q_pos_emb, k_pos_emb = rotary_pos_emb
            # need to cross check this condition during inference
            # if not set_inference_key_value_memory:
            if not is_first_step:
                # In inference, we compute one token at a time.
                # Select the correct positional embedding
                # (only the last token in the sequence)
                q_pos_emb = q_pos_emb[sequence_end - 1 : sequence_end]
            else:
                # In the first forward pass of inference,
                # we use the entire provided prefix.
                # q_pos_emb here has the rope embeddings of the entire
                # prefix + to-be-generated output so
                # we slice to just the prefix.
                q_pos_emb = q_pos_emb[:sequence_end, :, :, :]
            k_pos_emb = k_pos_emb[:sequence_end, :, :, :]
            rotary_pos_emb = (q_pos_emb, k_pos_emb)

    # ==================================
    # core attention computation
    # ==================================

    # expand the key_layer and value_layer [sk, b, ng, hn] -> [sk, b, np, hn]
    if self.num_attention_heads_per_partition // self.num_query_groups_per_partition > 1:
        key_layer = repeat_interleave(
            key_layer,
            self.num_attention_heads_per_partition // self.num_query_groups_per_partition,
            dim=2
        )
        value_layer = repeat_interleave(
            value_layer,
            self.num_attention_heads_per_partition // self.num_query_groups_per_partition,
            dim=2
        )

    # apply relative positional encoding (rotary embedding)
    if rotary_pos_emb is not None:
        q_pos_emb, k_pos_emb = rotary_pos_emb
        query_layer = apply_rotary_pos_emb(query_layer, q_pos_emb)
        key_layer = apply_rotary_pos_emb(key_layer, k_pos_emb)
        if self.rotary_interleaved_patch:
            # TODO, better ops to reduce overhead
            assert rearrange is not None, 'Please install einops first, e.g., with pip install einops'
            query_layer = rearrange(query_layer, '... (two headdim) -> ... two headdim', two=2)
            query_layer = rearrange(query_layer.transpose(-2, -1), '... headdim two -> ... (headdim two)', two=2)
            key_layer = rearrange(key_layer, '... (two headdim) -> ... two headdim', two=2)
            key_layer = rearrange(key_layer.transpose(-2, -1), '... headdim two -> ... (headdim two)', two=2)
            value_layer = rearrange(value_layer, '... (two headdim) -> ... two headdim', two=2)
            value_layer = rearrange(value_layer.transpose(-2, -1), '... headdim two -> ... (headdim two)', two=2)

        # TODO, can apply positional embedding to value_layer so it has
        # absolute positional embedding.
        # otherwise, only relative positional embedding takes effect
        # value_layer = apply_rotary_pos_emb(value_layer, k_pos_emb)

    if not self.use_flash_attn:
        if self.checkpoint_core_attention:
            context_layer = self._checkpointed_attention_forward(
                query_layer, key_layer, value_layer, attention_mask)
        else:
            context_layer = self.core_attention(
                query_layer, key_layer, value_layer, attention_mask)
    else:
        hidden_head_num = query_layer.size(2)
        if self.shape_order == 'BSH':
            q, k, v = [rearrange(x, 's b h d -> b s (h d)').contiguous()
                for x in (query_layer, key_layer, value_layer)]
        elif self.shape_order == 'SBH':
            q, k, v = [rearrange(x, 's b h d -> s b (h d)').contiguous()
                for x in (query_layer, key_layer, value_layer)]
        elif self.shape_order == 'BNSD':
            q, k, v = [rearrange(x, 's b h d -> b h s d').contiguous()
                       for x in (query_layer, key_layer, value_layer)]
        elif self.shape_order == 'BSND':
            q, k, v = [rearrange(x, 's b h d -> b s h d').contiguous()
                       for x in (query_layer, key_layer, value_layer)]
        else:
            raise ImportError('flash attention shape order must be SBH or BSH, please add args shape-order')
            
        if not self.sequence_parallel:
            with tensor_parallel.get_cuda_rng_tracker().fork():
                context_layer = self.core_attention_flash(q, k, v, hidden_head_num, attention_mask)
        else:
            context_layer = self.core_attention_flash(q, k, v, hidden_head_num, attention_mask)
        
        if self.shape_order == 'BSH':
            context_layer = torch.tensor(1.0).to(context_layer.dtype).npu() * context_layer
            context_layer = rearrange(context_layer, 'b s D -> s b D').contiguous()
        elif self.shape_order == 'BNSD':
            context_layer = torch.tensor(1.0).to(context_layer.dtype).npu() * context_layer
            context_layer = rearrange(context_layer, 'b h s d -> s b (h d)').contiguous()
        elif self.shape_order == 'BSND':
            context_layer = torch.tensor(1.0).to(context_layer.dtype).npu() * context_layer
            context_layer = rearrange(context_layer, 'b s h d -> s b (h d)').contiguous()

    # =================
    # Output. [sq, b, h]
    # =================

    output, bias = self.dense(context_layer)

    return output, bias

def repeat_interleave(inputs, repeats, dim):
    shape = inputs.shape
    new_shape = shape[:dim + 1] + (repeats,) + shape[dim + 1:]
    out_shape = shape[:dim] + (shape[dim] * repeats,) + shape[dim + 1:]
    return inputs.unsqueeze(dim + 1).expand(new_shape).reshape(out_shape)

def CoreAttentionForward(self, query_layer, key_layer,
            value_layer, attention_mask):

    # ===================================
    # Raw attention scores. [b, np, s, s]
    # ===================================

    # [b, np, sq, sk]
    output_size = (query_layer.size(1),
                query_layer.size(2),
                query_layer.size(0),
                key_layer.size(0))

    # [sq, b, np, hn] -> [sq, b * np, hn]
    query_layer = query_layer.reshape(output_size[2],
                                    output_size[0] * output_size[1], -1)
    # [sk, b, np, hn] -> [sk, b * np, hn]
    key_layer = key_layer.view(output_size[3],
                            output_size[0] * output_size[1], -1)

    # preallocting input tensor: [b * np, sq, sk]
    matmul_input_buffer = mpu.get_global_memory_buffer().get_tensor(
        (output_size[0]*output_size[1], output_size[2], output_size[3]),
        query_layer.dtype, "mpu")

    # Raw attention scores. [b * np, sq, sk]
    # matmul_result = torch.baddbmm(
    #     matmul_input_buffer,
    #     query_layer.transpose(0, 1),   # [b * np, sq, hn]
    #     key_layer.transpose(0, 1).transpose(1, 2),  # [b * np, hn, sk]
    #     beta=0.0, alpha=(1.0/self.norm_factor))
    matmul_result = torch.bmm(query_layer.transpose(0, 1), key_layer.permute(1, 2, 0))
    matmul_result *= 1.0 / self.norm_factor

    # change view to [b, np, sq, sk]
    attention_scores = matmul_result.view(*output_size)

    if self.mup_coord_check:
        attention_scores = self.attn_score_no_op(attention_scores)

    # ===========================
    # Attention probs and dropout
    # ===========================

    # attention scores and attention mask [b, np, sq, sk]
    attention_probs = self.scale_mask_softmax(attention_scores,
                                            attention_mask)

    # This is actually dropping out entire tokens to attend to, which might
    # seem a bit unusual, but is taken from the original Transformer paper.
    if not self.sequence_parallel:
        with tensor_parallel.get_cuda_rng_tracker().fork():
            attention_probs = self.attention_dropout(attention_probs)
    else:
        attention_probs = self.attention_dropout(attention_probs)

    # =========================
    # Context layer. [sq, b, hp]
    # =========================

    # value_layer -> context layer.
    # [sk, b, np, hn] --> [b, np, sq, hn]

    # context layer shape: [b, np, sq, hn]
    output_size = (value_layer.size(1),
                value_layer.size(2),
                query_layer.size(0),
                value_layer.size(3))

    # change view [sk, b * np, hn]
    value_layer = value_layer.view(value_layer.size(0),
                                output_size[0] * output_size[1], -1)

    # change view [b * np, sq, sk]
    attention_probs = attention_probs.view(output_size[0] * output_size[1],
                                        output_size[2], -1)

    # matmul: [b * np, sq, hn]
    context_layer = torch.bmm(attention_probs, value_layer.transpose(0, 1))

    # change view [b, np, sq, hn]
    context_layer = context_layer.view(*output_size)

    # [b, np, sq, hn] --> [sq, b, np, hn]
    context_layer = context_layer.permute(2, 0, 1, 3).contiguous()

    # [sq, b, np, hn] --> [sq, b, hp]
    new_context_layer_shape = context_layer.size()[:-2] + \
        (self.hidden_size_per_partition,)
    context_layer = context_layer.view(*new_context_layer_shape)

    return context_layer


def ParallelMLPInit(self, config, layer_number):
    super(ParallelMLP, self).__init__()
    args = get_args()

    self.layer_number = max(1, layer_number)
    self.add_bias = config.add_bias_linear

    ffn_hidden_size = config.ffn_hidden_size
    if config.gated_linear_unit:
        ffn_hidden_size *= 2

    # TODO
    if args.apply_init_customized:
        assert args.swiglu, "Only support for ParallelMLP using swiglu."
        init_method_ffn_w1 = init_method_normal(
            args.init_method_std_scaled_ffn_w1[self.layer_number-1])
        init_method_ffn_w2 = init_method_normal(
            args.init_method_std_scaled_ffn_w2[self.layer_number-1])
        init_method_ffn_w3 = init_method_normal(
            args.init_method_std_scaled_ffn_w3[self.layer_number-1])

    if args.mup is None:
        # Project to 4h. If using swiglu double the output width, see https://arxiv.org/pdf/2002.05202.pdf
        self.dense_h_to_4h = tensor_parallel.ColumnParallelLinear(
            config.hidden_size,
            ffn_hidden_size,
            config=config,
            init_method=config.init_method,
            bias=self.add_bias,
            gather_output=False,
            skip_bias_add=True,
        )
    else:
        assert args.swiglu == True, "Only support for ParallelMLP using swiglu."
        self.dense_h_to_4h1 = tensor_parallel.ColumnParallelLinear(
            config.hidden_size,
            config.ffn_hidden_size,
            config=config,
            init_method=config.init_method,
            bias=self.add_bias,
            gather_output=False,
            skip_bias_add=True,
        )
        self.dense_h_to_4h2 = tensor_parallel.ColumnParallelLinear(
            config.hidden_size,
            config.ffn_hidden_size,
            config=config,
            init_method=config.init_method,
            bias=self.add_bias,
            gather_output=False,
            skip_bias_add=True,
        )
    self.mup = args.mup

    self.bias_gelu_fusion = False
    self.activation_func = None
    self.swiglu = args.swiglu

    if args.openai_gelu:
        self.activation_func = openai_gelu
    elif args.onnx_safe:
        self.activation_func = erf_gelu
    elif args.swiglu:
        if args.mup is None:
            if args.use_npu_swiglu:
                assert args.bf16, "only support bf16 when use npu swiglu"
                def swiglu(x):
                    return torch_npu.npu_swiglu(x, dim=-1)
            else:
                def swiglu(x):
                    x = torch.chunk(x, 2, dim=-1)
                    return F.silu(x[0]) * x[1]
        else:
            def swiglu(x1, x2):
                return F.silu(x1) * x2
        self.activation_func = swiglu
    elif args.squared_relu:
        def squared_relu(x):
            return torch.pow(F.relu(x), 2)
        self.activation_func = squared_relu
    else:
        self.bias_gelu_fusion = args.bias_gelu_fusion
        self.activation_func = F.gelu

    # Project back to h.
    self.dense_4h_to_h = tensor_parallel.RowParallelLinear(
        config.ffn_hidden_size,
        config.hidden_size,
        config=config,
        init_method=config.output_layer_init_method,
        bias=self.add_bias,
        input_is_parallel=True
    )

    if args.apply_init_customized:
        with tensor_parallel.get_cuda_rng_tracker().fork():
            w1, w3 = self.dense_h_to_4h.weight.chunk(2, dim=0)
            init_method_ffn_w1(w1)
            init_method_ffn_w3(w3)
            init_method_ffn_w2(self.dense_4h_to_h.weight)
            if torch.distributed.get_rank() == 0:
                print('Override ParallelMLP init_method.', flush=True)

megatron.model.transformer.FlashSelfAttention = FlashSelfAttention
megatron.model.transformer.ParallelAttention.__init__ = ParallelAttentionInit
megatron.model.transformer.ParallelAttention.forward = ParallelAttentionForward
megatron.model.transformer.CoreAttention.forward = CoreAttentionForward
megatron.model.transformer.ParallelMLP.__init__ = ParallelMLPInit
