# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import torch
import torch_mlu
from torch import nn
import megatron
from torch.nn.modules.normalization import LayerNorm
from megatron.model.enums import AttnMaskType, LayerType, AttnType, ModelType
from megatron import get_timers, get_args, get_retro_args, core, get_num_microbatches, get_hetero_context
from megatron.model.transformer import ParallelTransformerLayer, ParallelAttention, ParallelMLP, ParallelTransformer, _get_num_layers, _get_layer_type
from megatron.core import mpu, tensor_parallel
from contextlib import nullcontext
from megatron.core.models.common.rotary_pos_embedding import apply_rotary_pos_emb

try:
    from einops import rearrange
except ImportError:
    rearrange = None

# Try FlashAttn2 first
try:
    from flash_attn.flash_attn_interface import flash_attn_varlen_func as flash_attn_unpadded_func
except ImportError:
    try:
        from flash_attn.flash_attn_interface import flash_attn_unpadded_func
    except ImportError:
        flash_attn_unpadded_func = None


def _repeat_interleave_op(x, num_repeats):
    # only support 4D tensor, dim=2
    dim0 = x.shape[0]
    dim1 = x.shape[1]
    dim2 = x.shape[2]
    dim3 = x.shape[3]
    x = x[:, :, :, None, :].expand(dim0, dim1, dim2, num_repeats, dim3)
    x = x.reshape(dim0, dim1, dim2*num_repeats, dim3)
    return x

class RMSNorm(torch.nn.Module):

    def __init__(self,
                 dim: int,
                 eps: float = 1e-6,
                 sequence_parallel: bool = False):
        """RMS Normaliation module

        Arguments:
            dim (int): The width of input, i.e. hidden size
            eps (float): epsilon to use for the norm, default to 1e-6
            sequence_parallel (bool): Set to true if sequence parallelism is being used,
              this marks the weights as needing to be allreduced.
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

        setattr(self.weight, 'sequence_parallel', sequence_parallel)

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

def get_norm(config):
    args = get_args()
    if args.normalization == "LayerNorm" and args.apply_layernorm_rms == False:
        return LayerNorm(
            config.hidden_size,
            eps=config.layernorm_epsilon)
    elif args.normalization == "RMSNorm" or args.apply_layernorm_rms == True:
        if args.apply_layernorm_1p:
            raise NotImplementedError('RMSNorm does not currently support the layernorm_1p formulation.')

        return RMSNorm(dim=config.hidden_size,
                       eps=config.layernorm_epsilon,
                       sequence_parallel=config.sequence_parallel)
    else:
        raise Exception(f"unsupported norm type '{args.normalization}'.")

def ParallelTransformerLayerInit(self, config,
                 layer_number, layer_type=LayerType.encoder,
                 self_attn_mask_type=AttnMaskType.padding,
                 drop_path_rate=0.):
                 # retriever=None):
        super(ParallelTransformerLayer, self).__init__()
        args = get_args()

        self.layer_number = layer_number
        self.layer_type = layer_type

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

        # self.input_layernorm = LayerNorm(
        #     config.hidden_size,
        #     eps=config.layernorm_epsilon,
        #     no_persist_layer_norm=args.no_persist_layer_norm,
        #     sequence_parallel=config.sequence_parallel,
        #     apply_layernorm_1p=args.apply_layernorm_1p,
        #     apply_layernorm_rms=args.apply_layernorm_rms,
        #     init_weight=self.init_weight_attn_norm)
        self.input_layernorm = get_norm(config)



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
        # self.post_attention_layernorm = LayerNorm(
        #     config.hidden_size,
        #     eps=config.layernorm_epsilon,
        #     no_persist_layer_norm=not config.persist_layer_norm,
        #     sequence_parallel=config.sequence_parallel,
        #     apply_layernorm_1p=args.apply_layernorm_1p,
        #     apply_layernorm_rms=args.apply_layernorm_rms,
        #     init_weight=self.init_weight_ffn_norm)
        self.post_attention_layernorm = get_norm(config)

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
            # self.post_inter_attention_layernorm = LayerNorm(
            #     config.hidden_size,
            #     eps=config.layernorm_epsilon,
            #     no_persist_layer_norm=not config.persist_layer_norm,
            #     sequence_parallel=config.sequence_parallel,
            #     apply_layernorm_1p=args.apply_layernorm_1p)
            self.post_inter_attention_layernorm = get_norm(config)

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

def ParallelTransformerInit(self, config,
                 model_type, layer_type=LayerType.encoder,
                 self_attn_mask_type=AttnMaskType.padding,
                 post_layer_norm=True,
                 pre_process=True,
                 post_process=True,
                 drop_path_rate=0.0):
        super(ParallelTransformer, self).__init__()
        args = get_args()

        self.layer_type = layer_type
        self.model_type = model_type
        self.bf16 = config.bf16
        self.fp32_residual_connection = config.fp32_residual_connection
        self.post_layer_norm = post_layer_norm
        self.pre_process = pre_process
        self.post_process = post_process
        self.input_tensor = None
        self.drop_path_rate = drop_path_rate
        self.transformer_impl = args.transformer_impl
        self.retro_add_retriever = args.retro_add_retriever

        # Store activation checkpoiting flag.
        self.recompute_granularity = config.recompute_granularity
        self.recompute_method = config.recompute_method
        self.recompute_num_layers = config.recompute_num_layers
        self.distribute_saved_activations = \
            config.distribute_saved_activations and not config.sequence_parallel

        self.sequence_parallel = config.sequence_parallel

        # Transformer Engine Init.
        self.transformer_engine_v_0_10 = False
        self.transformer_engine_v_0_11 = False
        self.transformer_engine_v_0_8 = False
        if self.transformer_impl == 'transformer_engine':
            global transformer_engine
            import transformer_engine
            from importlib.metadata import version
            from pkg_resources import packaging

            te_version = packaging.version.Version(version("transformer-engine"))
            if te_version >= packaging.version.Version("0.8.0"):
                self.transformer_engine_v_0_8 = True
            if te_version >= packaging.version.Version("0.10.0"):
                self.transformer_engine_v_0_10 = True
            if te_version >= packaging.version.Version("0.11.0"):
                self.transformer_engine_v_0_11 = True

            del version, packaging

            assert not args.squared_relu, "TransformerEngine does not support squared relu activation."

        self.use_fp8 = args.fp8 is not None
        self.fp8_recipe = None
        self.fp8_group = None
        if self.use_fp8:
            assert args.transformer_impl == 'transformer_engine', \
                'transformer-engine required for fp8 training and inference'
            self.fp8_group = mpu.get_amax_reduction_group()
            if args.fp8 == "e4m3":
                fp8_format = transformer_engine.common.recipe.Format.E4M3
            elif args.fp8 == "hybrid":
                fp8_format = transformer_engine.common.recipe.Format.HYBRID
            else:
                raise ValueError("The DelayedScaling recipe only supports E4M3 and HYBRID formats.")
            self.fp8_recipe = transformer_engine.common.recipe.DelayedScaling(
                margin=args.fp8_margin,
                interval=args.fp8_interval,
                fp8_format=fp8_format,
                amax_history_len=args.fp8_amax_history_len,
                amax_compute_algo=args.fp8_amax_compute_algo,
                override_linear_precision=(False, False, not args.fp8_wgrad),
            )

        self.num_microbatches_in_previous_step = -1
        self.microbatch_count = 0
        self.checkpoint_core_attention = config.recompute_granularity == 'selective'

        # Number of layers.
        self.num_layers = _get_num_layers(args, model_type,
                                          layer_type==LayerType.decoder)

        self.drop_path_rates = [
            rate.item() for rate in
            torch.linspace(0, self.drop_path_rate, config.num_layers)]

        self.retro_layer_numbers = None
        if model_type == ModelType.retro_decoder:
            retro_layer_start = 6 if config.num_layers <= 15 else 9
            self.retro_layer_numbers = \
                np.arange(retro_layer_start, args.num_layers + 1, 3).tolist()
        if model_type == ModelType.retro_encoder:
            self.retro_layer_numbers = [1]

        # Transformer layers.
        if args.retro_add_retriever:
            assert self.recompute_granularity != 'full', \
                "Full recompute not supported for Retro."
            assert args.transformer_impl == 'local', \
                "Transformer engine does not support Retro layers."
        def build_layer(layer_number):
            if args.transformer_impl == 'local':
                current_layer_type = _get_layer_type(
                    model_type, layer_type, self.retro_layer_numbers,
                    layer_number)
                return ParallelTransformerLayer(
                    config,
                    layer_number,
                    layer_type=current_layer_type,
                    self_attn_mask_type=self_attn_mask_type,
                    drop_path_rate=self.drop_path_rates[layer_number - 1])
            else:
                # This argument is only available from TE v0.10 onwards.
                extra_transformer_engine_kwargs = {}
                if self.transformer_engine_v_0_8:
                    extra_transformer_engine_kwargs["bias"] = args.add_bias_linear
                if self.transformer_engine_v_0_10:
                    extra_transformer_engine_kwargs["activation"] = "swiglu" if args.swiglu else "gelu"
                if self.transformer_engine_v_0_11:
                    extra_transformer_engine_kwargs["normalization"] = args.normalization
                return transformer_engine.pytorch.TransformerLayer(
                    config.hidden_size,
                    config.ffn_hidden_size,
                    config.num_attention_heads,
                    layernorm_epsilon=config.layernorm_epsilon,
                    hidden_dropout=config.hidden_dropout,
                    attention_dropout=config.attention_dropout,
                    init_method=config.init_method,
                    output_layer_init_method=config.output_layer_init_method,
                    layer_number=layer_number,
                    kv_channels=config.kv_channels,
                    self_attn_mask_type=self_attn_mask_type.name,
                    tp_group=mpu.get_tensor_model_parallel_group(),
                    get_rng_state_tracker=tensor_parallel.get_cuda_rng_tracker,
                    fuse_wgrad_accumulation=config.gradient_accumulation_fusion,
                    apply_query_key_layer_scaling=config.apply_query_key_layer_scaling,
                    attention_softmax_in_fp32=config.attention_softmax_in_fp32,
                    seq_length=args.seq_length,
                    micro_batch_size=args.micro_batch_size,
                    sequence_parallel=config.sequence_parallel,
                    params_dtype=config.params_dtype,
                    apply_residual_connection_post_layernorm=config.apply_residual_connection_post_layernorm,
                    output_layernorm=False,
                    layer_type="encoder",
                    drop_path_rate=self.drop_path_rates[layer_number - 1],
                    set_parallel_mode=True,
                    fuse_qkv_params=True,
                    **extra_transformer_engine_kwargs)

        if config.virtual_pipeline_model_parallel_size is not None:
            assert config.num_layers % config.virtual_pipeline_model_parallel_size == 0, \
                'num_layers_per_stage must be divisible by ' \
                'virtual_pipeline_model_parallel_size'
            assert args.model_type != ModelType.encoder_and_decoder
            assert args.hetero_mode != "pp", \
                "Heterogenous pipeline parallelism is not supported for virtual pipeline model parallel."
            # Number of layers in each model chunk is the number of layers in the stage,
            # divided by the number of model chunks in a stage.
            self.num_layers = self.num_layers // config.virtual_pipeline_model_parallel_size
            # With 8 layers, 2 stages, and 4 model chunks, we want an assignment of
            # layers to stages like (each list is a model chunk):
            # Stage 0: [0]  [2]  [4]  [6]
            # Stage 1: [1]  [3]  [5]  [7]
            # With 8 layers, 2 stages, and 2 virtual stages, we want an assignment of
            # layers to stages like (each list is a model chunk):
            # Stage 0: [0, 1]  [4, 5]
            # Stage 1: [2, 3]  [6, 7]
            offset = mpu.get_virtual_pipeline_model_parallel_rank() * (
                config.num_layers // config.virtual_pipeline_model_parallel_size) + \
                (mpu.get_pipeline_model_parallel_rank() * self.num_layers)
        else:
            # Each stage gets a contiguous set of layers.
            if args.model_type == ModelType.encoder_and_decoder and \
                    mpu.get_pipeline_model_parallel_world_size() > 1:
                assert args.hetero_mode != "pp", \
                    "Heterogenous pipeline parallelism is not supported for encoder-decoder models."
                pipeline_rank = mpu.get_pipeline_model_parallel_rank()
                if layer_type == LayerType.encoder:
                    offset = pipeline_rank * self.num_layers
                else:
                    num_ranks_in_enc = args.pipeline_model_parallel_split_rank
                    offset = (pipeline_rank - num_ranks_in_enc) * self.num_layers
            else:
                if args.hetero_mode != "pp":
                    offset = mpu.get_pipeline_model_parallel_rank() * self.num_layers
                else:
                    offset, self.num_layers = _get_layer_info(args)

        if self.num_layers == 0:
            # When a standalone embedding stage is used (e.g.,
            # args.standalone_embedding_stage == True), virtual pipeline ranks
            # on pipeline rank 0 will have zero transformer layers assigned to
            # them. This results in the model's input and output tensors to be
            # the same, which will cause failure for certain output tensor
            # optimizations (e.g., pipeline output deallocation). To remedy
            # this, we assign a 'no-op' layer on these ranks, which will
            # disconnect the input tensor from the output tensor.
            self.num_layers = 1
            self.layers = torch.nn.ModuleList([ NoopTransformerLayer(1) ])
        else:
            self.layers = torch.nn.ModuleList(
                [build_layer(i + 1 + offset) for i in range(self.num_layers)])

            # Update dropout rate for Retro encoder.
            if model_type == ModelType.retro_encoder:
                for layer in self.layers:
                    if layer.self_attention.use_flash_attn:
                        layer.self_attention.core_attention_flash.dropout_p = \
                            torch.nn.Dropout(args.retro_encoder_attention_dropout)
                    else:
                        layer.self_attention.core_attention.attention_dropout.p =\
                            args.retro_encoder_attention_dropout
                    layer.hidden_dropout = args.retro_encoder_hidden_dropout

        # TODO
        self.init_weight_output_norm = args.layernorm_init_weight
        if args.apply_init_norm_customized:
            self.init_weight_output_norm = args.init_weight_output_norm

        if self.post_process and self.post_layer_norm:
            # Final layer norm before output.
            # self.final_layernorm = LayerNorm(
            #     config.hidden_size,
            #     eps=config.layernorm_epsilon,
            #     no_persist_layer_norm=args.no_persist_layer_norm,
            #     sequence_parallel=config.sequence_parallel,
            #     apply_layernorm_1p=args.apply_layernorm_1p,
            #     apply_layernorm_rms=args.apply_layernorm_rms,
            #     init_weight=self.init_weight_output_norm)
            self.final_layernorm = get_norm(config)

def FlashSelfAttentionForward(self, q, k, v):
    """Implements the multihead softmax attention.
    Arguments
    ---------
        q, k, v: The tensor containing the query, key, and value. (B, S, H, D)
    """

    assert all((i.dtype in [torch.float16, torch.bfloat16] for i in (q,k,v)))
    assert all((i.is_mlu for i in (q,k,v)))

    batch_size, seqlen_q = q.shape[0], q.shape[1]
    seqlen_k = k.shape[1]

    q, k, v = [rearrange(x, 'b s ... -> (b s) ...') for x in [q, k, v]]
    cu_seqlens_q = torch.arange(0, (batch_size + 1) * seqlen_q, step=seqlen_q, dtype=torch.int32,
                                device=q.device)

    if self.training:
        # during training q,k,v always have same seqlen
        assert seqlen_k == seqlen_q

        is_causal = self.causal
        cu_seqlens_k = cu_seqlens_q
        dropout_p = self.dropout_p
    else:
        # turn off FA causal mask after first inference autoregressive iteration
        # only on first autoregressive step q,k,v have same seqlen
        is_causal = seqlen_q == seqlen_k
        cu_seqlens_k = torch.arange(0, (batch_size + 1) * seqlen_k, step=seqlen_k, dtype=torch.int32,
                    device=q.device)
        dropout_p = 0

    output = flash_attn_unpadded_func(
        q, k, v, cu_seqlens_q, cu_seqlens_k, seqlen_q, seqlen_k,
        dropout_p,
        softmax_scale=self.softmax_scale, causal=is_causal
    )

    output = rearrange(output, '(b s) ... -> b s ...', b=batch_size)
    return output


def _allocate_memory(self, inference_max_sequence_len, batch_size, num_attention_heads):
    return torch.empty(
        inference_max_sequence_len,
        batch_size,
        num_attention_heads,
        self.hidden_size_per_attention_head,
        dtype=self.params_dtype,
        device=torch.mlu.current_device())

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
        # key_layer = key_layer.repeat_interleave(
        #     self.num_attention_heads_per_partition // self.num_query_groups_per_partition,
        #     dim=2
        # )
        # value_layer = value_layer.repeat_interleave(
        #     self.num_attention_heads_per_partition // self.num_query_groups_per_partition,
        #     dim=2
        # )
        #TODO: repeat_interleave does not support bf16.
        if not args.bf16:
            key_layer = key_layer.repeat_interleave(
                self.num_attention_heads_per_partition // self.num_query_groups_per_partition,
                dim = 2
            )
            value_layer = value_layer.repeat_interleave(
                self.num_attention_heads_per_partition // self.num_query_groups_per_partition,
                dim = 2
            )
        else:
            key_layer = _repeat_interleave_op(key_layer,
                 self.num_attention_heads_per_partition // self.num_query_groups_per_partition)
            value_layer = _repeat_interleave_op(value_layer,
                 self.num_attention_heads_per_partition // self.num_query_groups_per_partition)

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
        q, k, v = [rearrange(x, 's b ... -> b s ...').contiguous()
                   for x in (query_layer, key_layer, value_layer)]
        if not self.sequence_parallel:
            with tensor_parallel.get_cuda_rng_tracker().fork():
                context_layer = self.core_attention_flash(q, k, v)
        else:
            context_layer = self.core_attention_flash(q, k, v)
        context_layer = rearrange(context_layer, 'b s h d -> s b (h d)').contiguous()

    # =================
    # Output. [sq, b, h]
    # =================

    output, bias = self.dense(context_layer)

    return output, bias



megatron.model.transformer.ParallelTransformerLayer.__init__ = ParallelTransformerLayerInit
megatron.model.transformer.ParallelTransformer.__init__ = ParallelTransformerInit
megatron.model.transformer.FlashSelfAttention.forward = FlashSelfAttentionForward
megatron.model.transformer.ParallelAttention._allocate_memory = _allocate_memory
megatron.model.transformer.ParallelAttention.forward = ParallelAttentionForward 
