# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

import math
from typing import Optional, Callable

import torch, torch_npu

from vllm.distributed import get_world_group, get_pp_group
from vllm.attention import AttentionMetadata
from vllm.platforms import current_platform
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.fused_moe.layer import FusedMoE as GPUFusedMoE
from vllm.model_executor.layers.fused_moe.layer import UnquantizedFusedMoEMethod as GPUUnquantizedFusedMoEMethod
from vllm.model_executor.layers.fused_moe.layer import FusedMoeWeightScaleSupported
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig, QuantizeMethodBase)
from omni.models.common.config.model_config import model_extra_config
from omni.adaptors.vllm.distributed.parallel_state import (
    get_expert_parallel_rank, 
    get_expert_parallel_world_size
)
from omni.adaptors.vllm.distributed.communication_op import expert_parallel_all_reduce

from omni.models.common.layers.fused_moe.fused_moe import (
    fused_experts_ep_best_alltoall, 
    fused_experts_alltoall_ep, 
    fused_experts_allgather_ep, 
    fused_topk,
    grouped_topk
)

__all__ = ['_prune_hidden_states']

if model_extra_config.operator_opt_config.use_omni_placement:
    from omni_planner import OmniPlanner

_MAX_NUM_TOKEN=100000
UNQUANT_MODE = 0
STATIC_QUANT_MODE = 1
DYNAMIC_QUANT_MODE = 2


class UnquantizedFusedMoEMethod(GPUUnquantizedFusedMoEMethod):
    LAST_SEQ_LEN = None
    BEST_EXPERT_TOKENS = None

    def __init__(self):
        super().__init__()
        self.initialized = False
        self.warm_up = True

    def apply(
            self,
            layer: torch.nn.Module,
            x: torch.Tensor,
            router_logits: torch.Tensor,
            top_k: int,
            renormalize: bool,
            use_grouped_topk: bool,
            topk_group: Optional[int] = None,
            num_expert_group: Optional[int] = None,
            custom_routing_function: Optional[Callable] = None,
            scoring_func: str = "softmax",
            e_score_correction_bias: Optional[torch.Tensor] = None
    ) -> torch.Tensor:

        return self.forward(x=x,
                            layer=layer,
                            router_logits=router_logits,
                            top_k=top_k,
                            renormalize=renormalize,
                            use_grouped_topk=use_grouped_topk,
                            topk_group=topk_group,
                            num_expert_group=num_expert_group,
                            custom_routing_function=custom_routing_function,
                            scoring_func=scoring_func,
                            e_score_correction_bias=e_score_correction_bias
                            )

    def forward_cuda(
            self,
            layer: torch.nn.Module,
            x: torch.Tensor,
            use_grouped_topk: bool,
            top_k: int,
            router_logits: torch.Tensor,
            renormalize: bool,
            topk_group: Optional[int] = None,
            num_expert_group: Optional[int] = None,
            custom_routing_function: Optional[Callable] = None,
            scoring_func: str = "softmax",
            e_score_correction_bias: Optional[torch.Tensor] = None
    ) -> torch.Tensor:

        if not self.initialized:
            layer.w13_weight = torch.nn.Parameter(layer.w13_weight.transpose(1,2).contiguous(), requires_grad=False)
            layer.w2_weight = torch.nn.Parameter(layer.w2_weight.transpose(1,2).contiguous(), requires_grad=False)
            self.n_routed_experts = len(layer.w13_weight)
            self.local_expert_indices_offset = (
                    get_expert_parallel_rank() * self.n_routed_experts
            )
            self.local_expert_indices = [
                self.local_expert_indices_offset + i for i in range(self.n_routed_experts)
            ]
            self.initialized = True

        topk_weights, topk_ids, row_idx = FusedMoE.select_experts(
            hidden_states=x,
            router_logits=router_logits,
            use_grouped_topk=use_grouped_topk,
            top_k=top_k,
            renormalize=renormalize,
            topk_group=topk_group,
            num_expert_group=num_expert_group,
            custom_routing_function=custom_routing_function,
            scoring_func=scoring_func,
            e_score_correction_bias=e_score_correction_bias,
            layer=layer
        )

        if model_extra_config.operator_opt_config.enable_moe_expert_parallel:
            if model_extra_config.operator_opt_config.enable_alltoall:
                if not model_extra_config.operator_opt_config.best_ep:
                    out = fused_experts_alltoall_ep(hidden_states=x,
                                                     w1=layer.w13_weight,
                                                     w2=layer.w2_weight,
                                                     topk_weights=topk_weights,
                                                     topk_ids=topk_ids,
                                                     row_idx=row_idx,
                                                     warm_up=self.warm_up)
                else:
                    out = fused_experts_ep_best_alltoall(hidden_states=x,
                                                     w1=layer.w13_weight,
                                                     w2=layer.w2_weight,
                                                     topk_weights=topk_weights,
                                                     topk_ids=topk_ids,
                                                     row_idx=row_idx)
            else:
                if model_extra_config.operator_opt_config.best_ep and (UnquantizedFusedMoEMethod.LAST_SEQ_LEN is None or UnquantizedFusedMoEMethod.LAST_SEQ_LEN != x.shape[0]):
                    avg_num_tokens = math.ceil(topk_ids.numel() / get_expert_parallel_world_size())
                    UnquantizedFusedMoEMethod.BEST_EXPERT_TOKENS = torch.ones(self.n_routed_experts, dtype=torch.int64, device=current_platform.device_type) * avg_num_tokens
                    UnquantizedFusedMoEMethod.LAST_SEQ_LEN = x.shape[0]

                out = fused_experts_allgather_ep(hidden_states=x,
                                    w1=layer.w13_weight,
                                    w2=layer.w2_weight,
                                    topk_weights=topk_weights,
                                    topk_ids=topk_ids,
                                    row_idx=row_idx,
                                    warm_up=self.warm_up,
                                    n_routed_experts=self.n_routed_experts,
                                    local_expert_indices=self.local_expert_indices,
                                    best_expert_tokens=UnquantizedFusedMoEMethod.BEST_EXPERT_TOKENS)
            if self.warm_up:
                self.warm_up = False
            return out
        else:
            raise ValueError("expert_parallel_size should be larger than 1 if enable_moe_expert_parallel")

class FusedMoE(torch.nn.Module):
    _load_w13=GPUFusedMoE._load_w13
    _load_w2=GPUFusedMoE._load_w2
    _load_single_value=GPUFusedMoE._load_single_value
    _load_g_idx=GPUFusedMoE._load_g_idx
    make_expert_params_mapping=GPUFusedMoE.make_expert_params_mapping
    _load_per_tensor_weight_scale=GPUFusedMoE._load_per_tensor_weight_scale
    _load_model_weight_or_group_weight_scale = GPUFusedMoE._load_model_weight_or_group_weight_scale
    # _load_fp8_scale = GPUFusedMoE._load_fp8_scale

    def __init__(
        self,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        params_dtype: Optional[torch.dtype] = None,
        reduce_results: bool = False,
        renormalize: bool = True,
        use_grouped_topk: bool = False,
        num_expert_group: Optional[int] = None,
        topk_group: Optional[int] = None,
        quant_config: Optional[QuantizationConfig] = None,
        tp_size: Optional[int] = None,
        prefix: str = "",
        custom_routing_function: Optional[Callable] = None,
        scoring_func: str = "softmax",
        e_score_correction_bias: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        # ENABLE_OMNI_PLANNER
        # OMNI_PLANNER: import omni planner instance, all layers share the same instance(singleton instance)
        if model_extra_config.operator_opt_config.use_omni_placement:
            self.planner = OmniPlanner(config_file= model_extra_config.operator_opt_config.omni_placement_config_path)
            self.moe_layer_idx = OmniPlanner.get_deepseek_v3_moe_layer_idx(prefix)
            self.expert_mapping = self.planner.expert_mapping_on_current_layer(self.moe_layer_idx)

        if model_extra_config.operator_opt_config.enable_moe_expert_parallel:
            ep_size = get_expert_parallel_world_size()
            num_experts = int(num_experts / ep_size)
            tp_size = 1

        if params_dtype is None:
            params_dtype = torch.get_default_dtype()

        self.tp_size = (tp_size if tp_size is not None else
                        get_expert_parallel_world_size())
        self.top_k = top_k
        self.num_experts = num_experts
        self.intermediate_size_per_partition = intermediate_size // self.tp_size
        self.reduce_results = reduce_results
        self.renormalize = renormalize
        self.use_grouped_topk = use_grouped_topk
        if self.use_grouped_topk:
            if num_expert_group is None or topk_group is None:
                raise RuntimeError("num_expert_group and topk_group must not be None")
        self.num_expert_group = num_expert_group
        self.topk_group = topk_group
        self.custom_routing_function = custom_routing_function

        if quant_config is None:
            self.quant_method: Optional[QuantizeMethodBase] = (
                UnquantizedFusedMoEMethod())
            self.quant_mode = UNQUANT_MODE
        else:
            self.quant_method = quant_config.get_quant_method(self, prefix)
            self.quant_mode = DYNAMIC_QUANT_MODE # static_quant_mode is not supported now
        if self.quant_method is None:
            raise RuntimeError("self.quant_method must not be None")

        #ENABLE_OMNI_PLANNER
        num_of_redundant_experts = 0
        if model_extra_config.operator_opt_config.use_omni_placement:
            num_of_redundant_experts = self.planner.get_num_of_redundant_experts(moe_layer_idx = self.moe_layer_idx,
                                                                                 num_expert_per_device_origin = num_experts,
                                                                                 rank_device = get_expert_parallel_rank())
        self.quant_method.create_weights(
            layer=self,
            num_experts=num_experts + num_of_redundant_experts,  #ENABLE_OMNI_PLANNER
            hidden_size=hidden_size,
            intermediate_size_per_partition=self.intermediate_size_per_partition,
            params_dtype=params_dtype,
            weight_loader=self.weight_loader)
        self.scoring_func = scoring_func
        self.e_score_correction_bias = e_score_correction_bias
        if self.scoring_func != "softmax" and not self.use_grouped_topk:
            raise ValueError("Only softmax scoring function is supported for "
                             "non-grouped topk.")

        if model_extra_config.operator_opt_config.moe_dispatch_combine:
            # Adapt the dispatch combine operator
            self.ep_size = get_expert_parallel_world_size()
            self.global_rank = get_world_group().rank_in_group
            self.world_size = get_world_group().world_size
            # self.n_shared_experts = n_shared_experts

            self.moe_all_to_all_group = get_world_group().device_group
            self.moe_all_to_all_group_name = self.moe_all_to_all_group._get_backend(torch.device(current_platform.device_type)).get_hccl_comm_name(
                self.global_rank)
            self.moe_rs_group = get_pp_group().device_group
            self.moe_rs_group_rank = get_pp_group().rank_in_group
            self.moe_rs_group_name = self.moe_rs_group._get_backend(torch.device(current_platform.device_type)).get_hccl_comm_name(
                                                 self.moe_rs_group_rank)


    @staticmethod
    def select_experts(hidden_states: torch.Tensor,
                       router_logits: torch.Tensor,
                       top_k: int,
                       use_grouped_topk: bool,
                       renormalize: bool,
                       topk_group: Optional[int] = None,
                       num_expert_group: Optional[int] = None,
                       custom_routing_function: Optional[Callable] = None,
                       scoring_func: str = "softmax",
                       e_score_correction_bias: Optional[torch.Tensor] = None,
                       routed_scaling_factor: Optional[torch.Tensor] = None,
                       layer: torch.nn.Module = None
                       ):
        # DeekSeekv2 uses grouped_top_k
        # adapt: When num_expert_group=1, it degenerates to fused_topk.
        if use_grouped_topk:# and num_expert_group != 1:
        # adapt end.
            if topk_group is None:
                raise ValueError(f"Unsupported topk_group is None")
            if num_expert_group is None:
                raise ValueError(f"Unsupported num_expert_group is None")

            if e_score_correction_bias is None:
                topk_weights, topk_ids, row_idx = grouped_topk(
                    hidden_states=hidden_states,
                    gating_output=router_logits,
                    topk=top_k,
                    renormalize=renormalize,
                    num_expert_group=num_expert_group,
                    topk_group=topk_group,
                    scoring_func=scoring_func,
                    e_score_correction_bias=e_score_correction_bias)
                topk_weights = topk_weights * routed_scaling_factor

            else:
                is_prefill = get_forward_context().attn_metadata is None or get_forward_context().attn_metadata.prefill is not None
                if is_prefill:
                    topk_weights, topk_ids, _ = torch_npu.npu_moe_gating_top_k(
                        router_logits.float(),
                        k=top_k,  # topk is currently 8
                        bias=e_score_correction_bias,    # float32
                        k_group=topk_group,  # fix: 4
                        group_count=num_expert_group,  # fix 8
                        group_select_mode=1,  # 0: maximum in group; 1: topk2.sum(fix)
                        renorm=0,  # 0: softmax->topk(fix); 1: topk->softmax
                        norm_type=1,  # 0: softmax; 1: sigmoid(fix)
                        routed_scaling_factor=routed_scaling_factor,
                        eps=float(1e-20))
                else:
                    # Only single operator is supported.
                    # topk_weights, topk_ids, _ = torch.ops.npu_inference.npu_moe_gating_top_k(
                    # Support single operator + graph mode
                    topk_weights, topk_ids, _ = torch_npu.npu_moe_gating_top_k(
                        router_logits,
                        k=top_k,  # topk is currently 8
                        bias=e_score_correction_bias,  # float32
                        k_group=topk_group,  # fix: 4
                        group_count=num_expert_group,  # fix 8
                        group_select_mode=1,  # 0: maximum in group; 1: topk2.sum(fix)
                        renorm=0,  # 0: softmax->topk(fix); 1: topk->softmax
                        norm_type=1,  # 0: softmax; 1: sigmoid(fix)
                        routed_scaling_factor=routed_scaling_factor,
                        eps=float(1e-20))
                row_idx = torch.arange(topk_ids.numel(), device=current_platform.device_type, dtype=torch.int32).view(-1, router_logits.shape[
                    0]).transpose(0, 1)
        elif custom_routing_function is None:
            topk_weights, topk_ids, row_idx = fused_topk(gating_output=router_logits,
                                                         topk=top_k,
                                                         renormalize=renormalize)
        else:
            topk_weights, topk_ids, row_idx = custom_routing_function(
                hidden_states=hidden_states,
                gating_output=router_logits,
                topk=top_k,
                renormalize=renormalize)
        
        #ENABLE_OMNI_PLANNER
        if model_extra_config.operator_opt_config.use_omni_placement and layer is not None:
            hidden_states, topk_ids, topk_weights = layer.planner.plan(layer_idx_moe=layer.moe_layer_idx, 
                                                           tokens=hidden_states, 
                                                           token_expert_ids=topk_ids, 
                                                           token_expert_scores=topk_weights,
                                                           top_k=layer.top_k,
                                                           expert_mapping=layer.expert_mapping,
                                                           is_prefill=is_prefill)

        if is_prefill and model_extra_config.operator_opt_config.best_ep:
            # Forced load balance
            t = (topk_ids.shape[0] * 8) // 256
            topk_ids = torch.arange(256, device=current_platform.device_type, dtype=torch.int32).unsqueeze(0).repeat(t + 1, 1).view(-1, 8)[:topk_ids.shape[0]]
        
        return topk_weights, topk_ids, row_idx

    def _load_per_channel_weight_scale(self, expert_data: torch.Tensor,
                                       shard_dim: int, shard_id: str,
                                       loaded_weight: torch.tensor,
                                       tp_rank: int):
        # adapt loaded_weight shape
        loaded_weight = loaded_weight.squeeze(-1)
        # adapt end
        # for per channel weight quantization
        if shard_id == "w2":
            expert_data.copy_(loaded_weight)
        elif shard_id in ("w1", "w3"):
            self._load_w13(shard_id=shard_id,
                           shard_dim=shard_dim,
                           loaded_weight=loaded_weight,
                           expert_data=expert_data,
                           tp_rank=tp_rank)

    def forward(self, hidden_states: torch.Tensor,
                topk_weights: torch.Tensor,
                topk_ids: torch.Tensor,
                pertoken_scale: torch.Tensor,
                attn_metadata: AttentionMetadata
                ):
        if self.quant_method is None:
            raise RuntimeError("self.quant_method must not be None")

        # Matrix multiply.
        final_hidden_states = self.quant_method.apply(
            layer=self,
            x=hidden_states,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            pertoken_scale=pertoken_scale,
            attn_metadata=attn_metadata
        )

        if self.reduce_results and self.tp_size > 1:
            final_hidden_states = expert_parallel_all_reduce(
                final_hidden_states)

        return final_hidden_states

    def weight_loader(self, param: torch.nn.Parameter,
                      loaded_weight: torch.Tensor, weight_name: str,
                      shard_id: str, expert_id: int) -> None:


        if model_extra_config.operator_opt_config.enable_moe_expert_parallel:
            ep_rank = get_expert_parallel_rank()
            # ENABLE_OMNI_PLANNER
            if model_extra_config.operator_opt_config.use_omni_placement:
                # OMNI_PLANNER: determine the expert deployment based on the pattern
                exists_locally, local_pos = self.planner.is_expert_on_current_rank(self.moe_layer_idx, expert_id, ep_rank, self.num_experts)
                # if the re-deployed expert is not on the current rank, then skip the weight_loader
                if not exists_locally:
                    return
                # if the re-deployed expert is on the current rank, then update the id of the expert
                else:
                    expert_id = ep_rank * self.num_experts + local_pos
            else:
                if expert_id < ep_rank * self.num_experts or expert_id >= (ep_rank + 1) * self.num_experts:
                    return
            tp_rank = 0
            expert_id -= ep_rank * self.num_experts
        else:
            tp_rank = get_expert_parallel_rank()
        # compressed-tensors checkpoints with packed weights are stored flipped
        loaded_weight = loaded_weight.t().contiguous() if (
            self.quant_method.__class__.__name__
            == "CompressedTensorsWNA16MoEMethod") else loaded_weight

        if shard_id not in ("w1", "w2", "w3"):
            raise ValueError(f"shard_id must be ['w1','w2','w3'] but "
                             f"got {shard_id}.")

        WEIGHT_SCALE_SUPPORTED = [
            e.value for e in FusedMoeWeightScaleSupported
        ]
        # Fetch the dim to shard the parameter/loaded weight
        # based on the shard id. This will be whatever
        # dimension intermediate_size is used.
        SHARD_ID_TO_SHARDED_DIM = {"w1": 0, "w2": 1, "w3": 0}

        expert_data = param.data[expert_id]

        # is_transposed: if the dim to shard the weight
        # should be flipped. Required by GPTQ, compressed-tensors
        # should be whatever dimension intermediate_size is
        is_transposed = getattr(param, "is_transposed", False)
        shard_dim = SHARD_ID_TO_SHARDED_DIM[shard_id]
        if is_transposed:
            shard_dim = ~shard_dim

        # Case input scale: input_scale loading is only supported for fp8
        if "input_scale" in weight_name:
            # this is needed for compressed-tensors only
            loaded_weight = loaded_weight.to(param.data.device)

            if param.data[expert_id] != 1 and (param.data[expert_id] -
                                               loaded_weight).abs() > 1e-5:
                raise ValueError(
                    "input_scales of w1 and w3 of a layer "
                    f"must be equal. But got {param.data[expert_id]} "
                    f"vs. {loaded_weight}")

            self._load_single_value(param=param,
                                    loaded_weight=loaded_weight,
                                    expert_id=expert_id)
            return

        # Case g_idx
        if "g_idx" in weight_name:
            self._load_g_idx(shard_dim=0,
                             shard_id=shard_id,
                             loaded_weight=loaded_weight,
                             expert_data=expert_data,
                             tp_rank=tp_rank)
            return

        # Case weight scales and zero_points
        if ("scale" in weight_name or "zero" in weight_name or "offset" in weight_name):
            # load the weight scales and zp based on the quantization scheme
            # supported weight scales/zp can be found in
            # FusedMoeWeightScaleSupported
            quant_method = getattr(param, "quant_method", None)
            if quant_method == FusedMoeWeightScaleSupported.CHANNEL.value:
                self._load_per_channel_weight_scale(
                    shard_id=shard_id,
                    shard_dim=shard_dim,
                    loaded_weight=loaded_weight,
                    expert_data=expert_data,
                    tp_rank=tp_rank)
            elif quant_method == FusedMoeWeightScaleSupported.GROUP.value:
                self._load_model_weight_or_group_weight_scale(
                    shard_id=shard_id,
                    shard_dim=shard_dim,
                    loaded_weight=loaded_weight,
                    expert_data=expert_data,
                    tp_rank=tp_rank)
            elif quant_method == FusedMoeWeightScaleSupported.TENSOR.value:
                self._load_per_tensor_weight_scale(shard_id=shard_id,
                                                   param=param,
                                                   loaded_weight=loaded_weight,
                                                   expert_id=expert_id)
            else:
                raise ValueError(
                    f"quant method must be one of {WEIGHT_SCALE_SUPPORTED}")
            return

        # Case weight_shape
        if "weight_shape" in weight_name:
            # only required by compressed-tensors
            self._load_single_value(param=param,
                                    loaded_weight=loaded_weight,
                                    expert_id=expert_id)
            return

        # Case model weights
        if "weight" in weight_name:
            self._load_model_weight_or_group_weight_scale(
                shard_id=shard_id,
                shard_dim=shard_dim,
                loaded_weight=loaded_weight,
                expert_data=expert_data,
                tp_rank=tp_rank)
            return
