# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

import math
from typing import Callable, List, Optional, Dict, Any

import torch, torch_npu
#from mindspeed.ops import quant_gmm

from vllm.attention import AttentionMetadata
from vllm.platforms import current_platform

__all__ = ["AscendCompressedTensorsW8A8Int8MoEMethod"]

from omni.adaptors.vllm.distributed.parallel_state import (
    get_expert_parallel_rank, get_expert_parallel_world_size)
from omni.models.common.config.model_config import model_extra_config

from omni.models.common.layers.fused_moe.fused_moe import (
    fused_experts_w8a8_moe_dispatch_combine, 
    moe_infer_fusion,
    fused_experts_w8a8_allgather_ep,
)

# OMNI_PLANNER: import omni planner instance, all layers share the same instance(singleton instance)
if model_extra_config.operator_opt_config.use_omni_placement:
    from omni_planner import OmniPlanner

SUPPORTED_BITS = 8
SEQ_SPLIT_LENGTH = 4096
torch.npu.config.allow_internal_format = True

class AscendCompressedTensorsW8A8Int8MoEMethod:

    LAST_SEQ_LEN = None
    BEST_EXPERT_TOKENS = None

    def __init__(self):
        self.initialized = False
        self.warm_up = True
    
    @staticmethod
    def get_weight(num_experts: int, intermediate_size_per_partition: int,
                   hidden_sizes: int,
                   params_dtype: torch.dtype) -> Dict[str, Any]:
        param_dict = {}
        param_dict["w13_weight"] = torch.empty(num_experts,
                                                    2 * intermediate_size_per_partition,
                                                    hidden_sizes,
                                                    dtype=torch.int8)
        param_dict["w2_weight"] = torch.empty(num_experts,
                                                   hidden_sizes,
                                                   intermediate_size_per_partition,
                                                   dtype=torch.int8)
        return param_dict

    @staticmethod
    def get_dynamic_quant_param(num_experts: int,
                                intermediate_size_per_partition: int,
                                hidden_sizes: int,
                                params_dtype: torch.dtype) -> Dict[str, Any]:
        param_dict = {}
        param_dict["w13_weight_scale"] = torch.ones(num_experts,
                                                  2 * intermediate_size_per_partition,
                                                  dtype=torch.float32
                                                  if params_dtype == torch.float16 else torch.bfloat16)
        param_dict["w13_weight_offset"] = torch.zeros(num_experts,
                                                  2 * intermediate_size_per_partition,
                                                  dtype=torch.float32
                                                  if params_dtype == torch.float16 else torch.bfloat16)
        param_dict["w2_weight_scale"] = torch.ones(num_experts,
                                                 hidden_sizes,
                                                 dtype=torch.float32
                                                 if params_dtype == torch.float16 else torch.bfloat16)
        param_dict["w2_weight_offset"] = torch.zeros(num_experts,
                                                  hidden_sizes,
                                                  dtype=torch.float32
                                                  if params_dtype == torch.float16 else torch.bfloat16)
        return param_dict

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        layer.w13_weight = torch.nn.Parameter(layer.w13_weight.transpose(1,2).contiguous(), requires_grad=False)
        layer.w2_weight = torch.nn.Parameter(layer.w2_weight.transpose(1,2).contiguous(), requires_grad=False)
        if model_extra_config.operator_opt_config.gmm_nz:
            layer.w13_weight.data = torch_npu.npu_format_cast(layer.w13_weight, 29)
            layer.w2_weight.data = torch_npu.npu_format_cast(layer.w2_weight, 29)
        if not model_extra_config.operator_opt_config.opt_w2_scale_cast:
            layer.w2_weight_scale = torch.nn.Parameter(layer.w2_weight_scale.to(torch.float32), requires_grad=False)
        layer.w13_weight_scale = torch.nn.Parameter(layer.w13_weight_scale.to(torch.float32), requires_grad=False)
        self.n_routed_experts = len(layer.w13_weight)
        self.local_expert_indices_offset = (
                get_expert_parallel_rank() * self.n_routed_experts
        )
        self.local_expert_indices = [
            self.local_expert_indices_offset + i for i in range(self.n_routed_experts)
        ]
        self.initialized = True


    def apply(
            self,
            layer: torch.nn.Module,
            x: torch.Tensor,
            topk_weights: torch.Tensor,
            topk_ids: torch.Tensor,
            pertoken_scale: torch.Tensor,
            attn_metadata: AttentionMetadata
    ) -> torch.Tensor:
        #ENABLE_OMNI_PLANNER
        max_num_deployed_expert_per_rank = self.n_routed_experts
        if model_extra_config.operator_opt_config.use_omni_placement:
            max_num_deployed_expert_per_rank = layer.planner.get_max_num_deployed_expert_per_rank()

        if model_extra_config.operator_opt_config.enable_moe_expert_parallel:
            is_prefill = attn_metadata is None or attn_metadata.prefill is not None
            if model_extra_config.operator_opt_config.prefill_dispatch_combine or (model_extra_config.operator_opt_config.moe_dispatch_combine and is_prefill):
                if is_prefill and model_extra_config.operator_opt_config.enable_pd_separated:
                    row_idx = torch.arange(topk_ids.numel(), device=current_platform.device_type,
                                       dtype=torch.int32).view(-1,x.shape[0]).transpose(0,1)
                    out = moe_infer_fusion(layer, x, topk_ids, topk_weights, layer.w13_weight, layer.w2_weight,
                                           layer.w13_weight_scale, layer.w2_weight_scale, row_idx, self.warm_up, is_prefill)
                else:
                    out = fused_experts_w8a8_moe_dispatch_combine(layer,
                                                                    x,
                                                                    layer.w13_weight,
                                                                    layer.w2_weight,
                                                                    layer.w13_weight_scale,
                                                                    layer.w2_weight_scale,
                                                                    topk_weights,
                                                                    topk_ids,
                                                                    n_routed_experts=self.n_routed_experts * get_expert_parallel_world_size(), 
                                                                    max_num_deployed_expert=max_num_deployed_expert_per_rank * get_expert_parallel_world_size(),
                                                                    is_prefill=is_prefill #ENABLE_OMNI_PLANNER
                                                                    )
            else:
                if model_extra_config.operator_opt_config.best_ep and (
                        AscendCompressedTensorsW8A8Int8MoEMethod.LAST_SEQ_LEN is None or AscendCompressedTensorsW8A8Int8MoEMethod.LAST_SEQ_LEN !=
                        x.shape[0]):
                    avg_num_tokens = math.ceil(topk_ids.numel() / get_expert_parallel_world_size())
                    AscendCompressedTensorsW8A8Int8MoEMethod.BEST_EXPERT_TOKENS = torch.ones(self.n_routed_experts,
                                                                                             dtype=torch.int64,
                                                                                             device=current_platform.device_type) * avg_num_tokens
                    AscendCompressedTensorsW8A8Int8MoEMethod.LAST_SEQ_LEN = x.shape[0]

                out = fused_experts_w8a8_allgather_ep(hidden_states=x,
                                                      pertoken_scale=pertoken_scale,
                                                      w1=layer.w13_weight,
                                                      w2=layer.w2_weight,
                                                      w1_scale=layer.w13_weight_scale,
                                                      w2_scale=layer.w2_weight_scale,
                                                      topk_weights=topk_weights,
                                                      topk_ids=topk_ids,
                                                      n_routed_experts=self.n_routed_experts,
                                                      attn_metadata=attn_metadata,
                                                      max_num_deployed_expert_per_rank=max_num_deployed_expert_per_rank #ENABLE_OMNI_PLANNER
                                                      )
            if self.warm_up:
                self.warm_up = False
            return out
        else:
            row_idx = torch.arange(topk_ids.numel(), device=current_platform.device_type,
                                   dtype=torch.int32).view(-1, x.shape[0]).transpose(0,1)
            token_num = x.shape[0]
            if token_num > SEQ_SPLIT_LENGTH:  # Split seq to reduce memory usage
                x_list = x.split(SEQ_SPLIT_LENGTH)
                topk_weights_list = topk_weights.split(SEQ_SPLIT_LENGTH)
                topk_ids_list = topk_ids.split(SEQ_SPLIT_LENGTH)
                out = []
                for i in range(len(x_list)):
                    split_token, top_k = topk_weights_list[i].shape
                    row_idx = torch.arange(split_token * top_k).to(torch.int32).view(
                        (top_k, split_token)).T.contiguous().npu()
                    out.append(fused_experts_w8a8(x_list[i],
                                                    layer.w13_weight,
                                                    layer.w2_weight,
                                                    layer.w13_weight_scale,
                                                    layer.w2_weight_scale,
                                                    layer.w13_weight_offset,
                                                    layer.w2_weight_offset,
                                                    topk_weights_list[i],
                                                    topk_ids_list[i],
                                                    row_idx))
                return torch.concat(out)
            return fused_experts_w8a8(x,
                                        layer.w13_weight,
                                        layer.w2_weight,
                                        layer.w13_weight_scale,
                                        layer.w2_weight_scale,
                                        layer.w13_weight_offset,
                                        layer.w2_weight_offset,
                                        topk_weights,
                                        topk_ids,
                                        row_idx)

def fused_experts_w8a8(hidden_states: torch.Tensor,
                       w1: torch.Tensor,
                       w2: torch.Tensor,
                       w1_scale: torch.Tensor,
                       w2_scale: torch.Tensor,
                       w1_offset: torch.Tensor,
                       w2_offset: torch.Tensor,
                       topk_weights: torch.Tensor,
                       topk_ids: torch.Tensor,
                       row_idx: torch.Tensor,
                       ):

    num_tokens, hidden_size = hidden_states.shape
    n_routed_experts = len(w1)
    sorted_tokens, expanded_src_to_dst_row, expanded_expert_idx = \
        torch_npu.npu_moe_init_routing(hidden_states, row_idx, topk_ids, num_tokens)
    expert_tokens = torch_npu.npu_moe_compute_expert_tokens(expanded_expert_idx, n_routed_experts).to(torch.int64)
    act_dtype = hidden_states.dtype
    w1_scale = w1_scale.to(torch.bfloat16)
    w2_scale = w2_scale.to(torch.bfloat16)
    sorted_tokens, pertoken_scale = torch_npu.npu_dynamic_quant(sorted_tokens)
    gate_up_proj = torch_npu.npu_grouped_matmul([sorted_tokens], [w1], scale=[w1_scale], per_token_scale=[pertoken_scale],
                                          bias=None, group_list=expert_tokens, split_item=3, output_dtype=act_dtype, group_type=0,
                                          group_list_type=0)[0]

    gate_up_proj = torch_npu.npu_swiglu(gate_up_proj)
    gate_up_proj, pertoken_scale = torch_npu.npu_dynamic_quant(gate_up_proj)  # , smooth_scales=scale_2)

    out = torch_npu.npu_grouped_matmul([gate_up_proj], [w2], scale=[w2_scale], per_token_scale=[pertoken_scale],
                                 bias=None, group_list=expert_tokens, split_item=3, output_dtype=act_dtype, group_type=0,
                                 group_list_type=0)[0]
    out = out.float()
    return torch_npu.npu_moe_finalize_routing(out, None, None, None, topk_weights,
                                              expanded_src_to_dst_row, topk_ids).to(torch.bfloat16)
