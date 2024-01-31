import numpy as np
from contextlib import nullcontext
import torch
import megatron
from megatron.model.enums import AttnMaskType, AttnType
from megatron.model.module import MegatronModule
from megatron import get_args, core
from megatron.core import mpu, tensor_parallel
from megatron.core.models.common.rotary_pos_embedding import apply_rotary_pos_emb
from megatron.model.transformer import CoreAttention
from megatron.core import mpu, tensor_parallel
from megatron.model.transformer import (
                                        DropPath,
                                        ParallelMLP,
                                        SwitchMLP,
                                        ParallelTransformer)

from megatron.model.module import MegatronModule                                        
from megatron.model.transformer import (_get_num_layers,
                                        bias_dropout_add_fused_train,
                                        bias_dropout_add_fused_inference,
                                        get_bias_dropout_add)
from megatron.model.enums import AttnMaskType, LayerType, AttnType
from megatron.core.enums import ModelType
from megatron import  get_args, get_retro_args, core
from megatron.model import LayerNorm
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

class ParallelTransformerLayer(MegatronModule):
    """A single transformer layer.

    Transformer layer takes input with size [s, b, h] and returns an
    output of the same size.
    """

    def __init__(self, config,
                 layer_number, layer_type=LayerType.encoder,
                 self_attn_mask_type=AttnMaskType.padding,
                 drop_path_rate=0.):
                 # retriever=None):
        args = get_args()

        super(ParallelTransformerLayer, self).__init__()
        self.layer_number = layer_number
        self.layer_type = layer_type
        self.recompute_granularity = config.recompute_granularity
        self.recompute_method = config.recompute_method
        self.recompute_num_layers = config.recompute_num_layers
        self.num_layers = _get_num_layers(args, ModelType.encoder_or_decoder,
                                          layer_type==LayerType.decoder)
        self.pipeline_offset = mpu.get_pipeline_model_parallel_rank() * self.num_layers
        self.pp_layer_num = self.layer_number - self.pipeline_offset
        if torch.distributed.get_rank() % 8 == 0:
            print('self.pipeline_offset', self.pipeline_offset, 'self.pp_layer_num', self.pp_layer_num, 'self.num_layers', \
                self.num_layers, 'self.recompute_num_layers', self.recompute_num_layers, \
                    'config.recompute_granularity', config.recompute_granularity, 'config.recompute-method', config.recompute_method, flush=True)
       
        # TODO
        self.init_weight_attn_norm = args.layernorm_init_weight
        self.init_weight_ffn_norm = args.layernorm_init_weight
        if args.apply_init_norm_customized:
            self.init_weight_attn_norm = args.init_weight_attn_norm[self.layer_number-1];
            self.init_weight_ffn_norm = args.init_weight_ffn_norm[self.layer_number-1];

        self.apply_residual_connection_post_layernorm \
            = config.apply_residual_connection_post_layernorm

        self.bf16 = config.bf16
        self.fp32_residual_connection = config.fp32_residual_connection

        # Layernorm on the input data.
        self.input_layernorm = LayerNorm(
            config.hidden_size,
            eps=config.layernorm_epsilon,
            no_persist_layer_norm=args.no_persist_layer_norm,
            sequence_parallel=config.sequence_parallel,
            apply_layernorm_1p=args.apply_layernorm_1p,
            apply_layernorm_rms=args.apply_layernorm_rms,
            init_weight=self.init_weight_attn_norm)

        # Self attention.
        self.self_attention = ParallelAttention(
            config,
            layer_number,
            attention_type=AttnType.self_attn,
            attn_mask_type=self_attn_mask_type)
        self.hidden_dropout = config.hidden_dropout
        self.bias_dropout_fusion = config.bias_dropout_fusion
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else None

        # Layernorm on the attention output
        self.post_attention_layernorm = LayerNorm(
            config.hidden_size,
            eps=config.layernorm_epsilon,
            no_persist_layer_norm=not config.persist_layer_norm,
            sequence_parallel=config.sequence_parallel,
            apply_layernorm_1p=args.apply_layernorm_1p,
            apply_layernorm_rms=args.apply_layernorm_rms,
            init_weight=self.init_weight_ffn_norm)

        # Cross attention.
        if self.layer_type in (LayerType.decoder,
                               LayerType.retro_decoder,
                               LayerType.retro_decoder_with_retriever,
                               LayerType.retro_encoder):
            self.inter_attention = ParallelAttention(
                config,
                layer_number,
                attention_type=AttnType.cross_attn)
            # Layernorm on the attention output.
            self.post_inter_attention_layernorm = LayerNorm(
                config.hidden_size,
                eps=config.layernorm_epsilon,
                no_persist_layer_norm=not config.persist_layer_norm,
                sequence_parallel=config.sequence_parallel,
                apply_layernorm_1p=args.apply_layernorm_1p)

        # MLP
        if args.num_experts is not None:
            self.mlp = SwitchMLP(config, layer_number)
        else:
            self.mlp = ParallelMLP(config, layer_number)

        # Set bias+dropout+add fusion grad_enable execution handler.
        TORCH_MAJOR = int(torch.__version__.split('.')[0])
        TORCH_MINOR = int(torch.__version__.split('.')[1])
        use_nvfuser = TORCH_MAJOR > 1 or (TORCH_MAJOR == 1 and TORCH_MINOR >= 10)
        self.bias_dropout_add_exec_handler = \
                nullcontext if use_nvfuser else torch.enable_grad

        if args.retro_add_retriever:
            retro_args = get_retro_args()
            self.retro_num_neighbors = args.retro_num_neighbors
            self.retro_chunk_length = retro_args.retro_gpt_chunk_length
            self.retro_retrieved_length = retro_args.retro_gpt_retrieved_length

        # Retriever (bi-directional transformer with cross attention)
        if layer_type == LayerType.retro_decoder_with_retriever:
            self.retriever = ParallelTransformer(
                config=config,
                model_type=ModelType.retro_encoder,
                self_attn_mask_type=AttnMaskType.padding,
                pre_process=True,
                post_process=False,
            )
            self._retriever_key = 'retriever'
        else:
            self.retriever = None

    def default_decoder_cross_attention(self,
                                        encoder_output,
                                        enc_dec_attn_mask,
                                        layernorm_input,
                                        layernorm_output,
                                        bias_dropout_add_func):
        '''Cross attention for a standard encoder-decoder model.'''

        # Attention.
        attention_output, attention_bias = \
            self.inter_attention(layernorm_output,
                                 enc_dec_attn_mask,
                                 encoder_output=encoder_output)

        # Residual connection.
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = layernorm_input

        if attention_bias is not None:
            attention_bias = attention_bias.expand_as(residual)

        # Bias-dropout-add.
        with self.bias_dropout_add_exec_handler():
            layernorm_input = bias_dropout_add_func(
                attention_output,
                attention_bias,
                residual,
                self.hidden_dropout)

        # Layer norm.
        layernorm_output = self.post_inter_attention_layernorm(layernorm_input)

        return layernorm_input, layernorm_output

    def retro_encoder_cross_attention(self,
                                      retriever_output,
                                      layernorm_input,
                                      layernorm_output,
                                      bias_dropout_add_func):
        """Cross attention for Retro encoder.

        Notation:
            ns : Sequence length.
            bs : Batch size.
            d  : Hidden size.
            l  : Number of chunks per sample (i.e., seq_length/chunk_length).
            k  : Number of neighbors.
            r  : Number of retrieved tokens (neighbors + continuation).
        """

        ns, bs, d = layernorm_output.shape # [r, bs * l * k, d]

        # Divide sequence dimension into chunks.
        chunked_outputs = layernorm_output.reshape(self.retro_retrieved_length,
                                                   -1,
                                                   self.retro_num_neighbors,
                                                   d)
        chunked_outputs_before_layer_norm = \
            layernorm_input.reshape(self.retro_retrieved_length, -1,
                                    self.retro_num_neighbors, d) # [r, bs*l, k, d]

        # Per-chunk attention.
        layernorm_inputs = []
        layernorm_outputs = []
        for k in range(self.retro_num_neighbors):

            # Attention.
            chunked_output = chunked_outputs[:,:,k].contiguous()
            attention_output, attention_bias = \
                self.inter_attention(
                    chunked_output, # Q (neighbor embedding)
                    None,
                    encoder_output=retriever_output) # K, V (hidden act)

            # Residual connection.
            if self.apply_residual_connection_post_layernorm:
                residual = chunked_output
            else:
                residual = chunked_outputs_before_layer_norm[:,:,k]

            # Re-enable torch grad to enable fused optimization.
            with torch.enable_grad():
                layernorm_input = bias_dropout_add_func(
                    attention_output,
                    None if attention_bias is None else attention_bias.expand_as(residual),
                    residual,
                    self.hidden_dropout)
                layernorm_inputs.append(layernorm_input)

            # Layer norm.
            layernorm_output = \
                self.post_inter_attention_layernorm(layernorm_input)
            layernorm_outputs.append(layernorm_output)

        # Concatenate layer norms.
        # layernorm_input : [r, k * bs * l, d]
        # layernorm_output : [r, k * bs * l, d]
        layernorm_input = \
            torch.stack(layernorm_inputs, dim=1).reshape(ns, bs, d)
        layernorm_output = \
            torch.stack(layernorm_outputs, dim=1).reshape(ns, bs, d)

        return layernorm_input, layernorm_output

    def retro_decoder_cross_attention(self,
                                      retriever_input,
                                      retriever_output,
                                      retriever_attn_mask,
                                      layernorm_input,
                                      layernorm_output,
                                      inference_params,
                                      bias_dropout_add_func):
        """Cross attention for Retro decoder.

        Notation:
            ns : Sequence length.
            bs : Batch size.
            d  : Hidden size.
            l  : Number of chunks per sample (i.e., seq_length/chunk_length).
            m  : Number of tokens per chunk.
            k  : Number of neighbors.
            r  : Number of retrieved tokens (neighbors + continuation).
        """

        ns, bs, d = layernorm_output.shape
        l = int(np.ceil(ns / self.retro_chunk_length))

        # Retrieve neighbors.
        if self.layer_type == LayerType.retro_decoder_with_retriever:
            first_ns = ns % self.retro_chunk_length
            if first_ns > 0:
                raise Exception("test this case.")
                first_chunk, rest_chunk = \
                    layernorm_output[:first_ns], layernorm_output[first_ns:]
                first_chunk = torch.nn.functional.pad(
                    first_chunk,
                    (0, 0, 0, 0, 0, self.retro_chunk_length - first_ns),
                    'constant',
                    0)
                chunked_output = \
                    torch.cat((first_chunk, rest_chunk), dim=0) # [l * m, bs, d]
            else:
                chunked_output = layernorm_output # [l * m, bs, d]
            chunked_output = chunked_output \
                .reshape(l, self.retro_chunk_length, bs, d) \
                .permute(1, 2, 0, 3) \
                .reshape(self.retro_chunk_length, bs * l, d) \
                .contiguous()

            # Get Encoder Output
            retriever_output = self.retriever(
                hidden_states=retriever_input,
                attention_mask=retriever_attn_mask,
                retriever_output=chunked_output,
                retriever_attn_mask=retriever_attn_mask,
                inference_params=inference_params) # [r, k * bs * l , d]
            retriever_output = retriever_output.reshape(
                self.retro_retrieved_length * self.retro_num_neighbors, bs * l, d) # [r * k, bs * l, d]

        # Chunks.
        pad = (ns - 1) % self.retro_chunk_length
        attending_chunks = layernorm_output[pad:]
        padded_chunks = torch.nn.functional.pad(
            attending_chunks,
            (0, 0, 0, 0, 0, self.retro_chunk_length - 1),
            'constant', 0)
        padded_chunked_output = padded_chunks \
            .reshape(l, self.retro_chunk_length, bs, d) \
            .permute(1, 2, 0, 3)
        padded_chunked_output = padded_chunked_output.reshape(
            self.retro_chunk_length, bs * l, d).contiguous()

        # Encoder output.
        attention_output, attention_bias = \
            self.inter_attention(padded_chunked_output,
                                 None,
                                 encoder_output=retriever_output)

        # Residual connection.
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = layernorm_input

        # Re-enable torch grad to enable fused optimization.
        with torch.enable_grad():
            layernorm_input = bias_dropout_add_func(
                attention_output,
                None if attention_bias is None else attention_bias.expand_as(attention_output),
                torch.zeros_like(attention_output),
                self.hidden_dropout)
            layernorm_input = layernorm_input \
                .reshape(self.retro_chunk_length, bs, l, d) \
                .permute(2, 0, 1, 3) # [l, m, bs, d]
            layernorm_input = layernorm_input.reshape(self.retro_chunk_length * l, bs, d)
            layernorm_input = torch.nn.functional.pad(
                layernorm_input,
                (0, 0, 0, 0, pad, 0),
                'constant', 0)[:ns] # [ns, b, d]
            layernorm_input = layernorm_input + residual

        # Layer norm post the decoder attention
        layernorm_output = self.post_inter_attention_layernorm(layernorm_input)

        return retriever_output, layernorm_input, layernorm_output

    
    def _checkpointed_layer_norm(self, hidden_states, layrnorm_func):
        """Forward method with activation checkpointing."""
        def custom_forward(*inputs):
            hidden_states = inputs[0]
            layrnorm_function = inputs[1]
            output_ = layrnorm_function(hidden_states)
            return output_

        hidden_states = tensor_parallel.checkpoint(
            custom_forward,
            False, hidden_states, layrnorm_func)

        return hidden_states
    
    def _checkpointed_attention(self, query_layer, key_layer,
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

    def checkpoint_attention_layernorm(self, hidden_states, attention_mask,
                encoder_output=None, enc_dec_attn_mask=None,
                retriever_input=None,
                retriever_output=None,
                retriever_attn_mask=None,
                inference_params=None,
                rotary_pos_emb=None):
        def custom_forward(*inputs):
            hidden_states = inputs[0]
            attention_mask = inputs[1]
            rotary_pos_emb = inputs[-1]
            layernorm_output = self.input_layernorm(hidden_states)

            # Self attention.
            attention_output, attention_bias = \
                self.self_attention(
                    layernorm_output,
                    attention_mask,
                    inference_params=inference_params,
                    rotary_pos_emb=rotary_pos_emb)

            # Residual connection.
            if self.apply_residual_connection_post_layernorm:
                residual = layernorm_output
            else:
                residual = hidden_states

            if self.drop_path is None:
                # jit scripting for a nn.module (with dropout) is not
                # trigerring the fusion kernel. For now, we use two
                # different nn.functional routines to account for varying
                # dropout semantics during training and inference phases.
                if self.bias_dropout_fusion:
                    if self.training:
                        bias_dropout_add_func = bias_dropout_add_fused_train
                    else:
                        bias_dropout_add_func = bias_dropout_add_fused_inference
                else:
                    bias_dropout_add_func = get_bias_dropout_add(self.training)

                if attention_bias is not None:
                    attention_bias = attention_bias.expand_as(residual)
                with self.bias_dropout_add_exec_handler():
                    layernorm_input = bias_dropout_add_func(
                        attention_output,
                        attention_bias,
                        residual,
                        self.hidden_dropout)
            else:
                out = torch.nn.functional.dropout(attention_output + attention_bias,
                                                p=self.hidden_dropout,
                                                training=self.training)
                layernorm_input = residual + self.drop_path(out)

            # Layer norm post the self attention.
            layernorm_output = self.post_attention_layernorm(layernorm_input)
            return layernorm_input, layernorm_output

        hidden_states = tensor_parallel.checkpoint(
            custom_forward,
            False, hidden_states, attention_mask, encoder_output, enc_dec_attn_mask, retriever_input, retriever_output, retriever_attn_mask,
            inference_params, rotary_pos_emb)

        return hidden_states
            
    def forward(self, hidden_states, attention_mask,
                encoder_output=None, enc_dec_attn_mask=None,
                retriever_input=None,
                retriever_output=None,
                retriever_attn_mask=None,
                inference_params=None,
                rotary_pos_emb=None):
        # hidden_states: [s, b, h]
        if self.recompute_num_layers != 1 and self.pp_layer_num > self.recompute_num_layers:
            layernorm_input, layernorm_output = self.checkpoint_attention_layernorm(hidden_states, attention_mask,
                encoder_output, enc_dec_attn_mask,
                retriever_input, retriever_output,
                retriever_attn_mask, inference_params,
                rotary_pos_emb)
            bias_dropout_add_func = bias_dropout_add_fused_train
        else:

            # Layer norm at the beginning of the transformer layer.
            layernorm_output = self.input_layernorm(hidden_states)

            # Self attention.
            attention_output, attention_bias = \
                self.self_attention(
                    layernorm_output,
                    attention_mask,
                    inference_params=inference_params,
                    rotary_pos_emb=rotary_pos_emb)

            # Residual connection.
            if self.apply_residual_connection_post_layernorm:
                residual = layernorm_output
            else:
                residual = hidden_states

            if self.drop_path is None:
                # jit scripting for a nn.module (with dropout) is not
                # trigerring the fusion kernel. For now, we use two
                # different nn.functional routines to account for varying
                # dropout semantics during training and inference phases.
                if self.bias_dropout_fusion:
                    if self.training:
                        bias_dropout_add_func = bias_dropout_add_fused_train
                    else:
                        bias_dropout_add_func = bias_dropout_add_fused_inference
                else:
                    bias_dropout_add_func = get_bias_dropout_add(self.training)

                if attention_bias is not None:
                    attention_bias = attention_bias.expand_as(residual)
                with self.bias_dropout_add_exec_handler():
                    layernorm_input = bias_dropout_add_func(
                        attention_output,
                        attention_bias,
                        residual,
                        self.hidden_dropout)
            else:
                out = torch.nn.functional.dropout(attention_output + attention_bias,
                                                p=self.hidden_dropout,
                                                training=self.training)
                layernorm_input = residual + self.drop_path(out)

            # Layer norm post the self attention.
            layernorm_output = self.post_attention_layernorm(layernorm_input)

        # Cross attention.
        if self.layer_type == LayerType.encoder:
            pass
        elif self.layer_type == LayerType.decoder:
            layernorm_input, layernorm_output = \
                self.default_decoder_cross_attention(
                    encoder_output,
                    enc_dec_attn_mask,
                    layernorm_input,
                    layernorm_output,
                    bias_dropout_add_func)
        elif self.layer_type == LayerType.retro_encoder:
            layernorm_input, layernorm_output = \
                self.retro_encoder_cross_attention(
                    retriever_output,
                    layernorm_input,
                    layernorm_output,
                    bias_dropout_add_func)
        elif self.layer_type in (LayerType.retro_decoder,
                                 LayerType.retro_decoder_with_retriever):
            retriever_output, layernorm_input, layernorm_output = \
                self.retro_decoder_cross_attention(
                    retriever_input,
                    retriever_output,
                    retriever_attn_mask,
                    layernorm_input,
                    layernorm_output,
                    inference_params,
                    bias_dropout_add_func)
        else:
            raise Exception("Unsupported layer type, '%s'." %
                            self.layer_type.name)

        # MLP.
        mlp_output, mlp_bias = self.mlp(layernorm_output)

        # Second residual connection.
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = layernorm_input

        if self.drop_path is None:
            if mlp_bias is not None:
                mlp_bias = mlp_bias.expand_as(residual)
            with self.bias_dropout_add_exec_handler():
                output = bias_dropout_add_func(
                    mlp_output,
                    mlp_bias,
                    residual,
                    self.hidden_dropout)

            # Jit compiled function creates 'view' tensor. This tensor
            # potentially gets saved in the MPU checkpoint function context,
            # which rejects view tensors. While making a viewless tensor here
            # won't result in memory savings (like the data loader, or
            # p2p_communication), it serves to document the origin of this
            # 'view' tensor.
            output = core.utils.make_viewless_tensor(inp = output,
                                                     requires_grad = output.requires_grad,
                                                     keep_graph = True)

        else:
            if mlp_bias is not None:
                mlp_output = mlp_output + mlp_bias
            out = torch.nn.functional.dropout(mlp_output,
                                              p=self.hidden_dropout,
                                              training=self.training)
            output = residual + self.drop_path(out)

        if self.layer_type == LayerType.retro_decoder_with_retriever:
            return output, retriever_output
        else:
            return output
        
megatron.model.transformer.ParallelAttention = ParallelAttention
megatron.model.transformer.ParallelTransformerLayer = ParallelTransformerLayer
