
import torch
import megatron
from torch import Tensor

from megatron.core import parallel_state, tensor_parallel
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.enums import AttnMaskType
from megatron.training import get_args
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.module import MegatronModule
from flagscale.patches_utils import add_patches_module_
# [metax] start of change
try:
    from einops import rearrange
except ImportError:
    rearrange = None

# Try FlashAttn2 first
try:
    from flash_attn.flash_attn_interface import flash_attn_varlen_qkvpacked_func as flash_attn_varlen_qkvpacked_func
    from flash_attn.flash_attn_interface import flash_attn_varlen_func as flash_attn_unpadded_func
except ImportError:
        flash_attn_varlen_qkvpacked_func = None
        flash_attn_unpadded_func = None

def get_current_device() -> torch.device:
    """
    Returns currently selected device (gpu/cpu).
    If cuda available, return gpu, otherwise return cpu.
    """
    if torch.cuda.is_available():
        return torch.device(f"cuda:{torch.cuda.current_device()}")
    else:
        return torch.device("cpu")


class FlashSelfAttention_packed(torch.nn.Module):
    """Implement the scaled dot product attention with softmax.
    Arguments
    ---------
        softmax_scale: The temperature to use for the softmax attention.
                      (default: 1/sqrt(d_keys) where d_keys is computed at
                      runtime)
        attention_dropout: The dropout rate to apply to the attention
                           (default: 0.0)
    """
    def __init__(self, causal=False, softmax_scale=None, attention_dropout=0.0, micro_batch_size = None, 
                 device=None, dtype=None):
        super().__init__()
        assert flash_attn_varlen_qkvpacked_func is not None, ('Please install FlashAttention first, '
                                                      'e.g., with pip install flash-attn')
        assert rearrange is not None, 'Please install einops first, e.g., with pip install einops'
        self.causal = causal
        self.softmax_scale = softmax_scale
        self.dropout_p = attention_dropout
        self.micro_batch_size = micro_batch_size

    def forward(self, qkv):
        """Implements the multihead softmax attention.
        Arguments
        ---------
            qkv: The tensor containing the query, key, and value. [(b s) three h d]
        """

        assert qkv.dtype in [torch.float16, torch.bfloat16]
        assert qkv.is_cuda
        #batch_size, seqlen_q = q.shape[0], q.shape[1]
        #seqlen_k = k.shape[1]
        seq_len = qkv.shape[0] // self.micro_batch_size
        cu_seqlens = torch.arange(0, (self.micro_batch_size + 1) * seq_len, step=seq_len, dtype=torch.int32,
                                    device=qkv.device)

        if self.training:
            is_causal = self.causal
            dropout_p = self.dropout_p
        else:
            # turn off FA causal mask after first inference autoregressive iteration
            # only on first autoregressive step q,k,v have same seqlen
            is_causal = self.causal
            cu_seqlens_k = cu_seqlens
            dropout_p = 0

        output = flash_attn_varlen_qkvpacked_func(
            qkv, cu_seqlens, seq_len, dropout_p,
            softmax_scale=self.softmax_scale, causal=is_causal
        )

        output = rearrange(output, '(b s) ... -> b s ...', b=self.micro_batch_size)
        return output


class FlashSelfAttention_unpacked(torch.nn.Module):
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
                 device=None, dtype=None):
        super().__init__()
        assert flash_attn_unpadded_func is not None, ('Please install FlashAttention first, '
                                                      'e.g., with pip install flash-attn')
        assert rearrange is not None, 'Please install einops first, e.g., with pip install einops'
        self.causal = causal
        self.softmax_scale = softmax_scale
        self.dropout_p = attention_dropout

    def forward(self, q, k, v):
        """Implements the multihead softmax attention.
        Arguments
        ---------
            q, k, v: The tensor containing the query, key, and value. (B, S, H, D)
        """

        assert all((i.dtype in [torch.float16, torch.bfloat16] for i in (q,k,v)))
        assert all((i.is_cuda for i in (q,k,v)))

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
# [metax] end of change

class DotProductAttention(MegatronModule):
    """
    Region where selective activation recomputation is applied.
    This region is memory intensive but less compute intensive which
    makes activation checkpointing more efficient for LLMs (20B+).
    See Reducing Activation Recomputation in Large Transformer Models: https://arxiv.org/abs/2205.05198 for more details.

    We use the following notation:
     h: hidden size
     n: number of attention heads
     p: number of tensor model parallel partitions
     b: batch size
     s: sequence length
    """

    def __init__(
        self,
        config: TransformerConfig,
        layer_number: int,
        attn_mask_type: AttnMaskType,
        attention_type: str,
        attention_dropout: float = None,
    ):
        super().__init__(config=config)

        self.config: TransformerConfig = config

        assert (
            self.config.context_parallel_size == 1
        ), "Context parallelism is only supported by TEDotProductAttention!"

        assert (
            self.config.window_size is None
        ), "Sliding Window Attention is only supported by TEDotProductAttention!"
        # [metax] start of change
        self.args = get_args()
        self.seq_length = self.args.seq_length
        self.micro_batch_size = self.args.micro_batch_size
        self.head_dim = self.args.hidden_size // self.args.num_attention_heads
        if self.args.use_rotary_emb_implement == 'flash_attn': 
            assert self.args.rotary_percent == 1.0, ('rotary_emb implemented by flash attn only supported rotary_percent = 1.0')
            self.rotary_emb = RotaryEmbedding_Fuse(self.head_dim, device=get_current_device())

        self.use_flash_attn = self.args.use_flash_attn \
            and attention_type == 'self' \
            and attn_mask_type == AttnMaskType.causal
        if self.use_flash_attn:
            if flash_attn_varlen_qkvpacked_func is None:
                raise ImportError('FlashAttention is not installed, please install with '
                                  'pip install flash-attn')
            assert attention_type == 'self', ('FlashAttention code path only supports '
                                                          'self-attention for now')
            assert attn_mask_type == AttnMaskType.causal, ('FlashAttention code path only '
                                                                'supports causal mask for now')
           
            if self.args.use_rotary_emb_implement == 'flash_attn':        
                self.core_attention_flash = FlashSelfAttention_packed(causal=True, softmax_scale=1 / math.sqrt(self.head_dim), attention_dropout=self.config.attention_dropout,
                    micro_batch_size = self.micro_batch_size)
            elif self.args.use_rotary_emb_implement == 'apex':
                self.core_attention_flash = FlashSelfAttention_unpacked(
                    causal=True, attention_dropout=config.attention_dropout)
            else:
                self.core_attention_flash = None
        # [metax] end of change

        self.layer_number = max(1, layer_number)
        self.attn_mask_type = attn_mask_type
        self.attention_type = attention_type  # unused for now

        projection_size = self.config.kv_channels * self.config.num_attention_heads

        # Per attention head and per partition values.
        world_size = parallel_state.get_tensor_model_parallel_world_size()
        self.hidden_size_per_partition = divide(projection_size, world_size)
        self.hidden_size_per_attention_head = divide(projection_size, config.num_attention_heads)
        self.num_attention_heads_per_partition = divide(self.config.num_attention_heads, world_size)
        self.num_query_groups_per_partition = divide(self.config.num_query_groups, world_size)

        coeff = None
        self.norm_factor = math.sqrt(self.hidden_size_per_attention_head)
        if self.config.apply_query_key_layer_scaling:
            coeff = self.layer_number
            self.norm_factor *= coeff

        self.scale_mask_softmax = FusedScaleMaskSoftmax(
            input_in_fp16=self.config.fp16,
            input_in_bf16=self.config.bf16,
            attn_mask_type=self.attn_mask_type,
            scaled_masked_softmax_fusion=self.config.masked_softmax_fusion,
            mask_func=attention_mask_func,
            softmax_in_fp32=self.config.attention_softmax_in_fp32,
            scale=coeff,
        )

        # Dropout. Note that for a single iteration, this layer will generate
        # different outputs on different number of parallel partitions but
        # on average it should not be partition dependent.
        self.attention_dropout = torch.nn.Dropout(
            self.config.attention_dropout if attention_dropout is None else attention_dropout
        )

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Tensor,
        attn_mask_type: AttnMaskType = None,
        packed_seq_params: PackedSeqParams = None,
    ):
        assert packed_seq_params is None, (
            "Packed sequence is not supported by DotProductAttention."
            "Please use TEDotProductAttention instead."
        )

        # ===================================
        # Raw attention scores. [b, n/p, s, s]
        # ===================================

        # expand the key and value [sk, b, ng, hn] -> [sk, b, np, hn]
        # This is a noop for normal attention where ng == np. When using group query attention this
        # creates a view that has the keys and values virtually repeated along their dimension to
        # match the number of queries.

        # attn_mask_type is not used.
        if self.num_attention_heads_per_partition // self.num_query_groups_per_partition > 1:
            key = key.repeat_interleave(
                self.num_attention_heads_per_partition // self.num_query_groups_per_partition, dim=2
            )
            value = value.repeat_interleave(
                self.num_attention_heads_per_partition // self.num_query_groups_per_partition, dim=2
            )
        # [metax] start of change
        if self.args.use_rotary_emb_implement == 'flash_attn':
            indexes = torch.arange(key.shape[0]*key.shape[1])
            kwargs = {"arg3":3, "indexes":indexes}
            qkv = torch.stack([query, key, value], dim=2)
            qkv = rearrange(qkv, "s b three h d -> (b s) three h d")
            qkv = self.rotary_emb(qkv, **kwargs)
            if not self.use_flash_attn:
                qkv = rearrange(qkv, "(b s) three h d -> s b three h d", b = self.micro_batch_size)
                query = qkv[:,:,0,:,:]
                key = qkv[:,:,1,:,:]
                value = qkv[:,:,2,:,:]
            kwargs.pop("indexes")
        
        if self.use_flash_attn:
            if self.args.use_rotary_emb_implement == 'flash_attn':
                if not self.config.sequence_parallel:
                    with tensor_parallel.get_cuda_rng_tracker().fork():
                        context_layer = self.core_attention_flash(qkv)
                else:
                    context_layer = self.core_attention_flash(qkv)
                context_layer = rearrange(context_layer, 'b s h d -> s b (h d)').contiguous()

                return context_layer #[s, b, h]

            elif self.args.use_rotary_emb_implement == 'apex':
                q, k, v = [rearrange(x, 's b ... -> b s ...').contiguous()
                            for x in (query, key, value)]
                if not self.config.sequence_parallel:
                    with tensor_parallel.get_cuda_rng_tracker().fork():
                        context_layer = self.core_attention_flash(q, k, v)
                else:
                    context_layer = self.core_attention_flash(q, k, v)

                context_layer = rearrange(context_layer, 'b s h d -> s b (h d)').contiguous()

                return context_layer #[s, b, h]

            else:
                raise Exception('Only apex and flash_attn implement are curently supported')
        # [metax] end of change

        # [b, np, sq, sk]
        output_size = (
            query.size(1),
            query.size(2),
            query.size(0),
            key.size(0),
        )

        # [sq, b, np, hn] -> [sq, b * np, hn]
        # This will be a simple view when doing normal attention, but in group query attention
        # the key and value tensors are repeated to match the queries so you can't use simple strides
        # to extract the queries.
        query = query.reshape(output_size[2], output_size[0] * output_size[1], -1)
        # [sk, b, np, hn] -> [sk, b * np, hn]
        key = key.view(output_size[3], output_size[0] * output_size[1], -1)

        # preallocting input tensor: [b * np, sq, sk]
        matmul_input_buffer = parallel_state.get_global_memory_buffer().get_tensor(
            (output_size[0] * output_size[1], output_size[2], output_size[3]), query.dtype, "mpu",
        )

        # Raw attention scores. [b * np, sq, sk]
        matmul_result = torch.baddbmm(
            matmul_input_buffer,
            query.transpose(0, 1),  # [b * np, sq, hn]
            key.transpose(0, 1).transpose(1, 2),  # [b * np, hn, sk]
            beta=0.0,
            alpha=(1.0 / self.norm_factor),
        )

        # change view to [b, np, sq, sk]
        attention_scores = matmul_result.view(*output_size)

        # ===========================
        # Attention probs and dropout
        # ===========================

        # attention scores and attention mask [b, np, sq, sk]
        attention_probs: Tensor = self.scale_mask_softmax(attention_scores, attention_mask)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.

        if not self.config.sequence_parallel:
            with tensor_parallel.get_cuda_rng_tracker().fork():
                attention_probs = self.attention_dropout(attention_probs)
        else:
            attention_probs = self.attention_dropout(attention_probs)

        # =========================
        # Context layer. [sq, b, hp]
        # =========================

        # value -> context layer.
        # [sk, b, np, hn] --> [b, np, sq, hn]

        # context layer shape: [b, np, sq, hn]
        output_size = (
            value.size(1),
            value.size(2),
            query.size(0),
            value.size(3),
        )

        # change view [sk, b * np, hn]
        value = value.view(value.size(0), output_size[0] * output_size[1], -1)

        # change view [b * np, sq, sk]
        attention_probs = attention_probs.view(output_size[0] * output_size[1], output_size[2], -1)

        # matmul: [b * np, sq, hn]
        context = torch.bmm(attention_probs, value.transpose(0, 1))

        # change view [b, np, sq, hn]
        context = context.view(*output_size)

        # [b, np, sq, hn] --> [sq, b, np, hn]
        context = context.permute(2, 0, 1, 3).contiguous()

        # [sq, b, np, hn] --> [sq, b, hp]
        new_context_shape = context.size()[:-2] + (self.hidden_size_per_partition,)
        context = context.view(*new_context_shape)

        return context







module_path = "megatron.core.transformer"
module_dict = {"DotProductAttention":DotProductAttention,
               "FlashSelfAttention_packed":FlashSelfAttention_packed,
               "FlashSelfAttention_unpacked":FlashSelfAttention_unpacked}
add_patches_module_(module_path,module_dict)


# megatron.core.transformer.dot_product_attention.DotProductAttention = DotProductAttention
# import sys
# for k in sys.modules:
#     if k.startswith('megatron.core.transformer'):
#         if getattr(sys.modules[k], 'FlashSelfAttention_packed', None):
#             setattr(sys.modules[k], 'FlashSelfAttention_packed', FlashSelfAttention_packed)
#         if getattr(sys.modules[k], 'FlashSelfAttention_unpacked', None):
#             setattr(sys.modules[k], 'FlashSelfAttention_unpacked', FlashSelfAttention_unpacked)



