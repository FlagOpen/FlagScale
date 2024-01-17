import torch
import megatron
from megatron.model.enums import AttnMaskType, AttnType
from megatron.model.module import MegatronModule
from megatron import get_args, core
from megatron.core import mpu, tensor_parallel
from megatron.core.models.common.rotary_pos_embedding import apply_rotary_pos_emb
from megatron.model.transformer import CoreAttention
try:
    from einops import rearrange
except ImportError:
    rearrange = None

class FlashSelfAttention(MegatronModule):
    def __init__(self, causal=False, softmax_scale=None, attention_dropout=0.0,
                 device=None, dtype=None):
        super().__init__()
        assert rearrange is not None, 'Please install einops first, e.g., with pip install einops'
        self.causal = causal
        self.softmax_scale = softmax_scale
        self.dropout_p = attention_dropout


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
        attn_output = attn_output.permute(2, 0, 1, 3).reshape(*output_size)
        return attn_output


class ParallelAttention(MegatronModule):
    """Parallel self-attention layer abstract class.

    Self-attention layer takes input with size [s, b, h]
    and returns output of the same size.
    """

    def __init__(self, config, layer_number,
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
            if args.mup != "apply":
                self.core_attention_flash = FlashSelfAttention(
                    causal=True, attention_dropout=config.attention_dropout
                )
            else:
                softmax_scale =  args.mup_attn_multiplier / float(self.hidden_size_per_attention_head)
                self.core_attention_flash = FlashSelfAttention(
                    causal=True, softmax_scale=softmax_scale, attention_dropout=config.attention_dropout
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

    def _checkpointed_attention_forward(self, query_layer, key_layer,
                                        value_layer, attention_mask,
                                        rotary_pos_emb=None):
        """Forward method with activation checkpointing."""
        def custom_forward(*inputs):
            query_layer = inputs[0]
            key_layer = inputs[1]
            value_layer = inputs[2]
            attention_mask = inputs[3]
            output_ = self.core_attention(query_layer, key_layer,
                                          value_layer, attention_mask)
            return output_

        q_pos_emb, k_pos_emb = (None, None) if rotary_pos_emb is None \
            else rotary_pos_emb

        hidden_states = tensor_parallel.checkpoint(
            custom_forward,
            False, query_layer, key_layer, value_layer, attention_mask,
            q_pos_emb, k_pos_emb)

        return hidden_states

    def _allocate_memory(self, inference_max_sequence_len, batch_size, num_attention_heads):
        return torch.empty(
            inference_max_sequence_len,
            batch_size,
            num_attention_heads,
            self.hidden_size_per_attention_head,
            dtype=self.params_dtype,
            device=torch.cuda.current_device())

    def forward(self, hidden_states, attention_mask,
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
            key_layer = key_layer.repeat_interleave(
                self.num_attention_heads_per_partition // self.num_query_groups_per_partition,
                dim=2
            )
            value_layer = value_layer.repeat_interleave(
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

        if not self.use_flash_attn:
            if self.checkpoint_core_attention:
                context_layer = self._checkpointed_attention_forward(
                    query_layer, key_layer, value_layer, attention_mask)
            else:
                context_layer = self.core_attention(
                    query_layer, key_layer, value_layer, attention_mask)
        else:
            if not self.sequence_parallel:
                with tensor_parallel.get_cuda_rng_tracker().fork():
                    context_layer = self.core_attention_flash(
                        query_layer, key_layer, value_layer, attention_mask)
            else:
                context_layer = self.core_attention_flash(
                    query_layer, key_layer, value_layer, attention_mask)


        # =================
        # Output. [sq, b, h]
        # =================

        output, bias = self.dense(context_layer)

        return output, bias

megatron.model.transformer.ParallelAttention = ParallelAttention
