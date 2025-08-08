import math
from typing import Optional

import torch
from torch import Tensor
from einops import rearrange

from megatron.core import parallel_state, tensor_parallel
from megatron.core.fusions.fused_softmax import FusedScaleMaskSoftmax
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.utils import attention_mask_func
from megatron.core.utils import divide
from megatron.core.process_groups_config import ModelCommProcessGroups
from magi_attention.api import calc_attn, get_position_ids,flex_flash_attn_func
from magi_attention.dist_attn_runtime_mgr import DistAttnRuntimeKey

class MagiAttention(MegatronModule):
    def __init__(
        self,
        config: TransformerConfig,
        layer_number: int,
        attn_mask_type: AttnMaskType,
        attention_type: str,
        attention_dropout: Optional[float] = None,
        softmax_scale: Optional[float] = None,
        k_channels: Optional[int] = None,
        v_channels: Optional[int] = None,
        cp_comm_type: str = "p2p",
        model_comm_pgs: ModelCommProcessGroups = None,
    ):
        super().__init__(config=config)

        self.config: TransformerConfig = config
        assert (
            self.config.window_size is None
        ), "Sliding Window Attention is only supported by TEDotProductAttention!"

        self.layer_number = max(1, layer_number)
        self.attn_mask_type = attn_mask_type
        self.attention_type = attention_type  # unused for now

    def forward(        
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Tensor,
        attn_mask_type: AttnMaskType = None,
        attention_bias: Tensor = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
        magi_attention_key: DistAttnRuntimeKey = None,
    ):
        """Forward."""
        # print(f"MagiAttention.forward, query shape is {query.shape}")
        # print(f"MagiAttention.forward, key shape is {key.shape}")
        # print(f"MagiAttention.forward, value shape is {value.shape}")

        # reshape input from (b, s, nh, hd) to (s, nh, hd)
        dtype = query.dtype
        seq_len = query.shape[0]
        batch_size = query.shape[1]
        num_heads = query.shape[2]
        head_dim = query.shape[3]

        query, key, value = [
            rearrange(e, "s b nh hd -> (b s) nh hd").to(torch.bfloat16)
            for e in (query, key, value)
        ]
        # print(f"MagiAttention.forward, after rearrange, query shape is {query.shape}")
        # print(f"MagiAttention.forward, after rearrange, key shape is {key.shape}")
        # print(f"MagiAttention.forward, after rearrange, value shape is {value.shape}")

        o = calc_attn(query, key, value, magi_attention_key)[0]
        # print(f"MagiAttention.forward, after calc_attn, o shape is {o.shape}")
        # o = rearrange(o, "(b s) nh hd -> s b (nh hd)").to(dtype)
        o = o.contiguous().view(batch_size, seq_len, num_heads, head_dim).view(seq_len, batch_size, -1).to(dtype)
        # print(f"MagiAttention.forward, after rearrange, o shape is {o.shape}")

        return o