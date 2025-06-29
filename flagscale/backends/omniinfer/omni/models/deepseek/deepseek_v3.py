# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

# Adapted from
# https://github.com/huggingface/transformers/blob/v4.28.0/src/transformers/models/llama/modeling_llama.py
# Copyright 2023 The vLLM team.
# Copyright 2023 DeepSeek-AI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Inference-only DeepseekV3 model."""
import itertools
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union
from contextlib import nullcontext
import torch, torch_npu
from torch import nn
from transformers import PretrainedConfig
import torch.distributed as dist
import torchair as tng
torch._logging.set_logs(recompiles=True)
# vllm adaptor
from vllm.platforms import current_platform
from vllm.config import CacheConfig, QuantizationConfig, VllmConfig
from vllm.attention import Attention, AttentionMetadata
from vllm.sequence import IntermediateTensors
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.model_executor.layers.sampler import Sampler, SamplerOutput
from vllm.distributed import (
    get_dp_group,
    get_pp_group,
    get_world_group,
    get_tensor_model_parallel_world_size,
)
from vllm.distributed.communication_op import tensor_model_parallel_all_gather

from vllm.model_executor.models.utils import (
    PPMissingLayer, 
    is_pp_missing_parameter, 
    make_layers, 
    make_empty_intermediate_tensors_factory,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.layers.linear import (
    ReplicatedLinear,
    ColumnParallelLinear
)
# Huawei Cloud
from omni.models.common.layers.rotary_embedding import get_rope
from omni.models.common.layers.linear import (
    MergedReplicatedLinear,
    RowParallelLinearWithReduceScatter,
    AscendMergedColumnParallelLinear,
    AscendRowParallelLinear
)
from omni.models.common.layers.vocab_parallel_embedding import (
    ParallelLMHead, 
    VocabParallelEmbedding
)
from omni.models.common.layers.logits_processor import LogitsProcessor
from omni.models.common.layers.activation import SiluAndMul
from omni.models.common.layers.layernorm import RMSNorm
from omni.adaptors.vllm.distributed.parallel_state import (
    get_data_parallel_world_size,
    get_data_parallel_rank,
    get_expert_parallel_world_size
)
from omni.adaptors.vllm.distributed.communication_op import (
    mlp_all_gather,
    mlp_reduce_scatter,
    all_gather_two_stage,
    reduce_scatter_two_stage,
)
from omni.models.common.layers.fused_moe.layer import FusedMoE
from omni.models.common.layers.logits_processor import _prune_hidden_states
from omni.models.common.config.model_config import model_extra_config
from omni.adaptors.vllm.worker.npu_model_runner import GraphCompileConfiguration


"""MLP module activation split length, split by 64G VRAM, need to confirm the optimal split length based on sequence length and performance"""
SEQ_SPLIT_LENGTH = 4096
SEQ_SPLIT_LENGTH_BEFORE_ALL_GATHER = 64 if model_extra_config.operator_opt_config.prefill_dispatch_combine else 256
KVCACHE_NZ_DIM = 16
# Load FFN weights into L1Cache in advance, and add them after debugging is stable.
FFN1_PREFETCH_SIZE = 56 * 1024 * 1024


def yarn_get_mscale(scale: float = 1, mscale: float = 1) -> float:
    import math
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0

def condition_context(context, enable=True):
    return context if (enable and model_extra_config.operator_opt_config.use_super_kernel) else nullcontext()


class ReplicatedDeepseekMLP(nn.Module):
    """Replicates the inputs and weights across multiple GPUs. No memory saving."""
    def __init__(
            self,
            hidden_size: int,
            intermediate_size: int,
            hidden_act: str,
            quant_config: Optional[QuantizationConfig] = None,
            reduce_results: bool = True,
            prefix: str = "",
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedReplicatedLinear(
            hidden_size, [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj")
        self.gate_up_proj.throw_dequant = True
        self.down_proj = ReplicatedLinear(intermediate_size,
                                          hidden_size,
                                          bias=False,
                                          quant_config=quant_config,
                                          prefix=f"{prefix}.down_proj")
        if hidden_act != "silu":
            raise ValueError(f"Unsupported activation: {hidden_act}. "
                             "Only silu is supported for now.")
        self.act_fn_obj = SiluAndMul()
        self.quant_symbol = True if quant_config else False

    def act_fn(self, x, quant_symbol):
        if quant_symbol and isinstance(x, tuple):
            x = dict(zip(['x_int8', 'pertoken_scale'], x))
            x['out_scale'] = self.gate_up_proj.weight_scale
        return self.act_fn_obj(x, quant_symbol)

    def forward(self, x, attn_metadata):
        token_num = x.shape[0]
        if token_num > SEQ_SPLIT_LENGTH:  # Split seq to reduce memory usage
            x_list = x.split(SEQ_SPLIT_LENGTH)
            out = []
            for i in range(len(x_list)):
                x = x_list[i]
                gate_up, _ = self.gate_up_proj.forward(x)
                x = self.act_fn(gate_up, self.quant_symbol)
                x, _ = self.down_proj.forward(x)
                out.append(x)
            return torch.concat(out)
        gate_up, _ = self.gate_up_proj.forward(x)
        x = self.act_fn(gate_up, self.quant_symbol)
        x, _ = self.down_proj.forward(x)
        return x


class ParallelDeepseekMLP(nn.Module):

    def __init__(
            self,
            hidden_size: int,
            intermediate_size: int,
            hidden_act: str,
            quant_config: Optional[QuantizationConfig] = None,
            reduce_results: bool = True,
            prefix: str = "",
    ) -> None:
        super().__init__()
        self.prefix = prefix
        self.gate_up_proj = AscendMergedColumnParallelLinear(
            hidden_size, [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj")
        self.down_proj = AscendRowParallelLinear(intermediate_size,
                                           hidden_size,
                                           bias=False,
                                           quant_config=quant_config,
                                           reduce_results=False,
                                           prefix=f"{prefix}.down_proj")
        if hidden_act != "silu":
            raise ValueError(f"Unsupported activation: {hidden_act}. "
                             "Only silu is supported for now.")
        self.act_fn_obj = SiluAndMul()
        self.quant_symbol = True if quant_config else False

    def act_fn(self, x, quant_symbol):
        if quant_symbol and isinstance(x, tuple):
            x = dict(zip(['x_int8', 'pertoken_scale'], x))
            x['out_scale'] = self.gate_up_proj.weight_scale
        return self.act_fn_obj(x, quant_symbol)


    def forward(self, x, residual, attn_metadata, layerid=None):
        is_prefill = attn_metadata is None or attn_metadata.prefill
        if not is_prefill:
            # P and D are both cut, and are concave at the node
            x = mlp_all_gather(x, dim=0)
        else:
            pad_size = 0
            # pd mix use
            if model_extra_config.parall_config.dp_size > 1:
                local_length = x.shape[0]
                reduce_length = torch.tensor(x.shape[0], dtype=torch.int64, device=current_platform.device_type)
                dist.all_reduce(reduce_length, op=dist.ReduceOp.MAX, async_op=False)
                global_max_length = reduce_length.item()
                pad_size = global_max_length - x.shape[0]

                x = torch.nn.functional.pad(
                    x, (0, 0, 0, pad_size)
                )
            x = mlp_all_gather(x, dim=0)

        gate_up, _ = self.gate_up_proj.forward(x)
        x = self.act_fn(gate_up, self.quant_symbol)
        x, _ = self.down_proj.forward(x)

        # P and D are both cut, and are concave at the node (16)
        x = mlp_reduce_scatter(x)
        if is_prefill and pad_size > 0:
            x = x[:local_length, :]
        return x, residual


class DeepseekMoE(nn.Module):

    def __init__(
            self,
            config: PretrainedConfig,
            quant_config: Optional[QuantizationConfig] = None,
            prefix: str = "",
    ):
        super().__init__()
        self.prefix = prefix
        self.ep_size = get_expert_parallel_world_size()
        self.routed_scaling_factor = config.routed_scaling_factor
        self.n_shared_experts = config.n_shared_experts
        self.n_routed_experts = config.n_routed_experts
        self.device_count = torch.npu.device_count()
        self.routed_scaling_factor = config.routed_scaling_factor
        if self.ep_size > config.n_routed_experts:
            raise ValueError(
                f"Tensor parallel size {self.ep_size} is greater than "
                f"the number of experts {config.n_routed_experts}.")

        if config.hidden_act != "silu":
            raise ValueError(f"Unsupported activation: {config.hidden_act}. "
                             "Only silu is supported for now.")

        self.gate = ReplicatedLinear(config.hidden_size,
                                     config.n_routed_experts,
                                     bias=False,
                                     quant_config=None,
                                     params_dtype=torch.float32,
                                     prefix=f"{prefix}.gate")
        if config.topk_method == "noaux_tc":
            self.gate.e_score_correction_bias = nn.Parameter(
                torch.empty(config.n_routed_experts, dtype=torch.float), requires_grad=False)
        else:
            self.gate.e_score_correction_bias = None

        self.experts = FusedMoE(
            num_experts=config.n_routed_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            reduce_results=False,
            renormalize=config.norm_topk_prob,
            quant_config=quant_config,
            use_grouped_topk=True,
            num_expert_group=config.n_group,
            topk_group=config.topk_group,
            prefix=f"{prefix}.experts",
            scoring_func=config.scoring_func,
            e_score_correction_bias=self.gate.e_score_correction_bias,
        )
        self.warm_up = True
        if config.n_shared_experts is not None:
            intermediate_size = (config.moe_intermediate_size *
                                 config.n_shared_experts)

            self.shared_experts = ReplicatedDeepseekMLP(
                hidden_size=config.hidden_size,
                intermediate_size=intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                reduce_results=False,
                prefix=f"{prefix}.shared_experts",
            )

        self.local_expert_num = self.n_routed_experts // get_expert_parallel_world_size()
        self.in_scale_2 = torch.ones((self.local_expert_num, self.experts.w13_weight_scale.shape[-1] // 2), dtype=torch.float32, device=current_platform.device_type)
        torch._dynamo.mark_static(self.in_scale_2)

    def forward(self, hidden_states: torch.Tensor, residual: torch.Tensor, attn_metadata: AttentionMetadata, layer_id: int) -> torch.Tensor:
        is_prefill = (attn_metadata is None or attn_metadata.prefill is not None)

        if (model_extra_config.operator_opt_config.moe_multi_stream_tune and
            not is_prefill and 
            self.n_shared_experts is not None and 
            model_extra_config.operator_opt_config.moe_dispatch_combine):
            if self.warm_up:
                self.warm_up = False

            router_logits, _ = self.gate.forward(hidden_states.float())
            # Here, we do a 2D to 3D conversion, and then convert back to 2D to trigger the fusion rule, fusing add rms and cast into AddRmsNormCast.
            hidden_states_3d = hidden_states.unsqueeze(1)
            hidden_states = hidden_states_3d.squeeze(1)

            with tng.scope.npu_stream_switch('21'):
                hidden_states = tng.scope.npu_wait_tensor(hidden_states, router_logits)
                # shared_experts w13
                gate_up_share, _ = self.shared_experts.gate_up_proj.forward(hidden_states)

            topk_weights, topk_ids, _ = FusedMoE.select_experts(hidden_states, router_logits,
                                                                self.experts.top_k, self.experts.use_grouped_topk,
                                                                self.experts.renormalize,
                                                                self.experts.topk_group, self.experts.num_expert_group,
                                                                self.experts.custom_routing_function,
                                                                self.experts.scoring_func,
                                                                self.experts.e_score_correction_bias,
                                                                self.routed_scaling_factor,
                                                                layer=self.experts  # ENABLE_OMNI_PLANNER
                                                                )

            if model_extra_config.operator_opt_config.best_ep and attn_metadata.decode.best_topk is not None:
                faka_topk_ids = attn_metadata.decode.best_topk
                topk_ids = tng.scope.npu_wait_tensor(faka_topk_ids, topk_ids)

            mc2_mask = attn_metadata.decode.mc2_mask if attn_metadata is not None and attn_metadata.decode is not None else None
            layer = self.experts
            max_num_deployed_expert = self.n_routed_experts
            if model_extra_config.operator_opt_config.use_omni_placement:
                max_num_deployed_expert_per_rank = self.experts.planner.get_max_num_deployed_expert_per_rank()
                max_num_deployed_expert = max_num_deployed_expert_per_rank * get_expert_parallel_world_size()
            act_dtype = hidden_states.dtype
            shared_expert_rank_num = 0
            kwargs = {
                "x": hidden_states,
                "expert_ids": topk_ids,  # [n*topk]
                "expert_shard_type": 0,  # Set it to 0 for now
                "shared_expert_rank_num": shared_expert_rank_num,  # 32
                "moe_expert_num": max_num_deployed_expert, #ENABLE_OMNI_PLANNER, 0 redundancy 256, 1 redundancy expert 320
                "global_bs": 0,  # 0 Default (all); all tokens can be set
            }

            experts_tp_size = layer.tp_size
            world_size = get_world_group().world_size
            # In fact, what we get is the die number, and the ep group is not adapted by default.
            # The default ep group is experts_num/die_num.
            global_rank = get_world_group().rank_in_group
            all_to_all_group_size = world_size // experts_tp_size

            kwargs.update({
                "scales": None,  # Quantization coefficient
                "quant_mode": layer.quant_mode,  # 0: Non-quantization; 1: Static quantization; 2: Dynamic quantization
                "group_ep": layer.moe_all_to_all_group_name,  # Unlike torch, it is obtained by name.
                "ep_world_size": all_to_all_group_size,
                "ep_rank_id": global_rank // experts_tp_size,
                "group_tp": layer.moe_rs_group_name,
                "tp_world_size": experts_tp_size,
                "tp_rank_id": global_rank % experts_tp_size,
                "x_active_mask": mc2_mask,
            })

            output = torch_npu.npu_moe_distribute_dispatch(**kwargs)
            expand_x, dynamic_scale, expand_idx, expert_token_nums, ep_recv_counts = output[0:5]

            group_list = expert_token_nums.to(torch.int64)
            if model_extra_config.operator_opt_config.use_omni_placement and layer.planner.enable_dump and self.experts.moe_layer_idx < 58:
                if is_prefill:
                    layer.planner.npu_activation_count[layer.moe_layer_idx:layer.moe_layer_idx+1].add_(group_list[None])
                else:
                    with tng.scope.npu_stream_switch('22'):
                        layer.planner.npu_activation_count[layer.moe_layer_idx:layer.moe_layer_idx+1].add_(group_list[None])

            group_list = group_list[:len(self.experts.w13_weight)] # Adapt to redundant and non-redundant layers, #ENABLE_OMNI_PLANNER
            # cal experts
            weight1_3 = self.experts.w13_weight
            weight2 = self.experts.w2_weight
            weight_scale1_3 = self.experts.w13_weight_scale
            weight_scale2 = self.experts.w2_weight_scale

            if self.experts.quant_mode:  # 0: no quant 1: static quant 2: dynamic quant
                pertoken_scale = dynamic_scale
            else:
                expand_x, pertoken_scale = torch_npu.npu_dynamic_quant(expand_x)

            with tng.scope.npu_stream_switch('21'):
                wait_gate = gate_up_share if isinstance(gate_up_share, torch.Tensor) else gate_up_share[0]
                wait_gate = tng.scope.npu_wait_tensor(wait_gate, expand_x)
                if not isinstance(gate_up_share, torch.Tensor):
                    gate_up_share = (wait_gate, gate_up_share[1])
                intermediate_hiddenstates_share = self.shared_experts.act_fn(gate_up_share, self.shared_experts.quant_symbol)

            gate_up_proj = torch_npu.npu_grouped_matmul([expand_x], [weight1_3], bias=None, group_list=group_list,
                                                        split_item=3, output_dtype=torch.int32, group_type=0,
                                                        group_list_type=1)[0]
            
            gate_up_proj, pertoken_scale = torch_npu.npu_dequant_swiglu_quant(
                gate_up_proj, weight_scale=weight_scale1_3, activate_scale=pertoken_scale, bias=None, quant_scale=self.in_scale_2, quant_offset=None,
                group_index=group_list, activate_left=True, quant_mode=1)

            hidden_states_experts = torch_npu.npu_grouped_matmul([gate_up_proj], [weight2], scale=[weight_scale2],
                                            per_token_scale=[pertoken_scale],bias=None,
                                            group_list=group_list, split_item=3, output_dtype=act_dtype,
                                            group_type=0,
                                            group_list_type=1)[0]

            # moeCombine
            kwargs = {
                "expand_x": hidden_states_experts,
                "expert_ids": topk_ids,  # [n*topk]
                "expand_idx": expand_idx,
                "expert_scales": topk_weights.to(torch.float32),  # weight [n*topk]
                "expert_shard_type": 0,
                "shared_expert_rank_num": shared_expert_rank_num,
                "moe_expert_num":  max_num_deployed_expert, #ENABLE_OMNI_PLANNER, 0 redundancy 256, 1 redundancy expert 320
                "global_bs": 0,  # 0 Default (all); all tokens can be set
            }
            tp_recv_counts = output[5]
            stage3_kwargs = {
                "ep_send_counts": ep_recv_counts,  # dispatch's send_counts
                "group_ep": layer.moe_all_to_all_group_name,  # Unlike torch, it is obtained by name.
                "ep_world_size": all_to_all_group_size,
                "ep_rank_id": global_rank // experts_tp_size,
                "tp_send_counts": tp_recv_counts,
                "group_tp": layer.moe_rs_group_name,
                "tp_world_size": experts_tp_size,
                "tp_rank_id": global_rank % experts_tp_size,
                "x_active_mask": mc2_mask,
            }
            kwargs.update(stage3_kwargs)

            with tng.scope.npu_stream_switch('21'):
                if isinstance(intermediate_hiddenstates_share, dict):
                    intermediate_hiddenstates_share['x_int8'] = tng.scope.npu_wait_tensor(intermediate_hiddenstates_share.get('x_int8'), hidden_states_experts)
                else:
                    intermediate_hiddenstates_share = tng.scope.npu_wait_tensor(intermediate_hiddenstates_share, hidden_states_experts)
                shared_output, _ = self.shared_experts.down_proj.forward(intermediate_hiddenstates_share)

            hidden_states_route = torch_npu.npu_moe_distribute_combine(**kwargs)

            if shared_output is not None:
                final_hidden_states = (hidden_states_route, shared_output)

            return final_hidden_states, residual

        if self.n_shared_experts is not None:
            if is_prefill or not model_extra_config.operator_opt_config.enable_kv_rmsnorm_rope_cache:
                shared_output = self.shared_experts(hidden_states, attn_metadata)
            else:
                if model_extra_config.operator_opt_config.moe_multi_stream_tune:
                    with tng.scope.npu_stream_switch('21'):
                        hidden_states = tng.scope.npu_wait_tensor(hidden_states, hidden_states)
                        shared_output = self.shared_experts(hidden_states, attn_metadata)
                else:
                    shared_output = self.shared_experts(hidden_states, attn_metadata)

        # skip when use dispatch&combine
        if (is_prefill and
            not model_extra_config.operator_opt_config.prefill_dispatch_combine) or (
            not is_prefill and
            not model_extra_config.operator_opt_config.moe_dispatch_combine):
            hidden_states_int8, pertoken_scale = torch_npu.npu_dynamic_quant(hidden_states)
            global_hidden_states = all_gather_two_stage(hidden_states_int8, idx=0, dim=0)
        else:
            global_hidden_states = hidden_states
            global_pertoken_scale = None

        if self.warm_up:
            self.warm_up = False

        if is_prefill or not model_extra_config.operator_opt_config.enable_kv_rmsnorm_rope_cache:
            router_logits, _ = self.gate.forward(hidden_states.float())
            topk_weights, topk_ids, _ = FusedMoE.select_experts(hidden_states, router_logits,
                                                                      self.experts.top_k, self.experts.use_grouped_topk, self.experts.renormalize,
                                                                      self.experts.topk_group, self.experts.num_expert_group, self.experts.custom_routing_function,
                                                                      self.experts.scoring_func, self.experts.e_score_correction_bias, self.routed_scaling_factor,
                                                                      layer=self.experts  # ENABLE_OMNI_PLANNER
                                                                      )
            if not is_prefill and model_extra_config.operator_opt_config.best_ep and attn_metadata.decode.best_topk is not None:
                topk_ids = attn_metadata.decode.best_topk
            # skip when use dispatch&combine
            if (is_prefill and
                not model_extra_config.operator_opt_config.prefill_dispatch_combine) or (
                not is_prefill and
                not model_extra_config.operator_opt_config.moe_dispatch_combine):
                topk_cat = torch.cat((topk_weights, topk_ids.to(torch.float), pertoken_scale.unsqueeze(-1)), dim=-1)
                topk_all = all_gather_two_stage(topk_cat, idx=1, dim=0)
                topk_weights, topk_ids, global_pertoken_scale = torch.split(
                    topk_all, [topk_weights.shape[-1], topk_ids.shape[-1], 1], dim=-1)
                topk_ids = torch.round(topk_ids).to(torch.int32)
                global_pertoken_scale = global_pertoken_scale.squeeze(-1)
        else:
            if model_extra_config.operator_opt_config.moe_multi_stream_tune:
                with tng.scope.npu_stream_switch('22'):
                    hidden_states = tng.scope.npu_wait_tensor(hidden_states, hidden_states)
                    router_logits, _ = self.gate.forward(hidden_states.float())
                    # Here, we do a 2d-3d conversion and then convert back to 2d to trigger the fusion rule, fusing add rms and cast into AddRmsNormCast.
                    hidden_states_3d = hidden_states.unsqueeze(1)
                    hidden_states = hidden_states_3d.squeeze(1)
                    topk_weights, topk_ids, _ = FusedMoE.select_experts(hidden_states, router_logits,
                                                                        self.experts.top_k, self.experts.use_grouped_topk,
                                                                        self.experts.renormalize,
                                                                        self.experts.topk_group, self.experts.num_expert_group,
                                                                        self.experts.custom_routing_function,
                                                                        self.experts.scoring_func,
                                                                        self.experts.e_score_correction_bias,
                                                                        self.routed_scaling_factor,
                                                                        layer=self.experts  # ENABLE_OMNI_PLANNER
                                                                        )
                    if model_extra_config.operator_opt_config.best_ep and attn_metadata.decode.best_topk is not None:
                        topk_ids = attn_metadata.decode.best_topk
                    if not model_extra_config.operator_opt_config.moe_dispatch_combine:
                        pertoken_scale = tng.scope.npu_wait_tensor(pertoken_scale, pertoken_scale)
                        topk_cat = torch.cat((topk_weights, topk_ids.to(torch.float), pertoken_scale.unsqueeze(-1)), dim=-1)
                with tng.scope.npu_stream_switch('23'):
                    if not model_extra_config.operator_opt_config.moe_dispatch_combine:
                        topk_cat = tng.scope.npu_wait_tensor(topk_cat, topk_cat)
                        topk_all = all_gather_two_stage(topk_cat, idx=1, dim=0, reverse=True)
                with tng.scope.npu_stream_switch('22'):
                    if not model_extra_config.operator_opt_config.moe_dispatch_combine:
                        topk_all = tng.scope.npu_wait_tensor(topk_all, topk_all)
                        topk_all = topk_all.view(-1, self.device_count, topk_weights.shape[0], topk_all.shape[-1]) \
                                        .transpose(0, 1) \
                                        .reshape(-1, topk_all.shape[-1])
                        topk_weights, topk_ids, global_pertoken_scale = torch.split(topk_all, [topk_weights.shape[-1], topk_ids.shape[-1], 1], dim=-1)
                        topk_ids = torch.round(topk_ids).to(torch.int32)
                        global_pertoken_scale = global_pertoken_scale.squeeze(-1)
            else:
                router_logits, _ = self.gate.forward(hidden_states.float())
                # Here, we do a 2d-3d conversion and then convert back to 2d to trigger the fusion rule, fusing add rms and cast into AddRmsNormCast.
                hidden_states_3d = hidden_states.unsqueeze(1)
                hidden_states = hidden_states_3d.squeeze(1)
                topk_weights, topk_ids, _ = FusedMoE.select_experts(hidden_states, router_logits,
                                                                    self.experts.top_k, self.experts.use_grouped_topk,
                                                                    self.experts.renormalize,
                                                                    self.experts.topk_group, self.experts.num_expert_group,
                                                                    self.experts.custom_routing_function,
                                                                    self.experts.scoring_func,
                                                                    self.experts.e_score_correction_bias,
                                                                    self.routed_scaling_factor,
                                                                    layer=self.experts  # ENABLE_OMNI_PLANNER
                                                                    )
                if model_extra_config.operator_opt_config.best_ep and attn_metadata.decode.best_topk is not None:
                    topk_ids = attn_metadata.decode.best_topk
                if not model_extra_config.operator_opt_config.moe_dispatch_combine:
                    topk_cat = torch.cat((topk_weights, topk_ids.to(torch.float), pertoken_scale.unsqueeze(-1)), dim=-1)
                    topk_all = all_gather_two_stage(topk_cat, idx=1, dim=0, reverse=True)

                    topk_all = topk_all.view(-1, self.device_count, topk_weights.shape[0], topk_all.shape[-1]) \
                                       .transpose(0, 1) \
                                       .reshape(-1, topk_all.shape[-1])
                    topk_weights, topk_ids, global_pertoken_scale = torch.split(topk_all, [topk_weights.shape[-1], topk_ids.shape[-1], 1], dim=-1)
                    topk_ids = torch.round(topk_ids).to(torch.int32)
                    global_pertoken_scale = global_pertoken_scale.squeeze(-1)

        final_hidden_states_list = self.experts(
            hidden_states=global_hidden_states,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            pertoken_scale=global_pertoken_scale,
            attn_metadata=attn_metadata
        )

        if is_prefill and model_extra_config.operator_opt_config.prefill_dispatch_combine:
            if len(final_hidden_states_list) != 4:
                raise RuntimeError("len(final_hidden_states_list) != 4")
            final_hidden_states = final_hidden_states_list[0]
            gathered_tokens = final_hidden_states_list[1]
            expanded_row_idx = final_hidden_states_list[3]
        else:
            final_hidden_states = final_hidden_states_list


        # skip when use dispatch&combine
        if (is_prefill and
            not model_extra_config.operator_opt_config.prefill_dispatch_combine) or (
            not is_prefill and
            not model_extra_config.operator_opt_config.moe_dispatch_combine):
            final_hidden_states = reduce_scatter_two_stage(final_hidden_states, idx=0)

        if shared_output is not None:
            if is_prefill and model_extra_config.operator_opt_config.prefill_dispatch_combine:
                final_hidden_states = torch_npu.npu_moe_finalize_routing(
                	gathered_tokens,
                	skip1=shared_output,
					skip2=None,
                	bias=None,
                	scales=topk_weights.to(gathered_tokens.dtype),
                	expanded_src_to_dst_row=expanded_row_idx,
                	export_for_source_row=None,
                	drop_pad_mode=2
                )
            else:
                final_hidden_states = final_hidden_states + shared_output

        return final_hidden_states, residual


def yarn_get_mscale(scale: float = 1, mscale: float = 1) -> float:
    import math
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


class AscendDeepseekAttention_MLA(nn.Module):

    def __init__(
            self,
            config: PretrainedConfig,
            hidden_size: int,
            num_heads: int,
            qk_nope_head_dim: int,
            qk_rope_head_dim: int,
            v_head_dim: int,
            q_lora_rank: int,
            kv_lora_rank: int,
            rope_theta: float = 10000,
            rope_scaling: Optional[Dict[str, Any]] = None,
            max_position_embeddings: int = 8192,
            cache_config: Optional[CacheConfig] = None, # type: ignore
            quant_config: Optional[QuantizationConfig] = None,
            prefix: str = "",
    ) -> None:
        super().__init__()
        self.prefix = prefix
        self.hidden_size = hidden_size
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.num_heads = num_heads
        tp_size = get_tensor_model_parallel_world_size()
        if num_heads % tp_size != 0:
            raise RuntimeError("num_heads % tp_size != 0")
        self.num_local_heads = num_heads // tp_size
        self.scaling = self.qk_head_dim ** -0.5
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings
        self.kv_scale = None
        # FA is fully quantized, KVCache is not quantized, and the function is not enabled.
        self.use_faquant = False

        self.merge_qkv = model_extra_config.operator_opt_config.merge_qkv
        if self.q_lora_rank is not None:
            if self.merge_qkv:
                self.qkv_a_proj = MergedReplicatedLinear(self.hidden_size,
                                                         [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim],
                                                         bias=False,
                                                         quant_config=quant_config,
                                                         prefix=f"{prefix}.qkv_a_proj")
            else:
                self.q_a_proj = ReplicatedLinear(self.hidden_size,
                                                 self.q_lora_rank,
                                                 bias=False,
                                                 quant_config=quant_config,
                                                 prefix=f"{prefix}.q_a_proj")
                self.kv_a_proj_with_mqa = ReplicatedLinear(
                    self.hidden_size,
                    self.kv_lora_rank + self.qk_rope_head_dim,
                    bias=False,
                    quant_config=quant_config,
                    prefix=f"{prefix}.kv_a_proj_with_mqa")
            self.q_a_layernorm = RMSNorm(self.q_lora_rank,
                                         eps=config.rms_norm_eps)

            self.q_b_proj = ColumnParallelLinear(q_lora_rank,
                                                 self.num_heads *
                                                 self.qk_head_dim,
                                                 bias=False,
                                                 quant_config=quant_config,
                                                 prefix=f"{prefix}.q_b_proj")
        else:
            self.q_proj = ColumnParallelLinear(self.hidden_size,
                                               self.num_heads *
                                               self.qk_head_dim,
                                               bias=False,
                                               quant_config=quant_config,
                                               prefix=f"{prefix}.q_proj")
            self.kv_a_proj_with_mqa = ReplicatedLinear(
                self.hidden_size,
                self.kv_lora_rank + self.qk_rope_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.kv_a_proj_with_mqa")

        self.kv_a_layernorm = RMSNorm(self.kv_lora_rank,
                                      eps=config.rms_norm_eps)
        self.kv_b_proj = ColumnParallelLinear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.kv_b_proj")
        # O projection.
        if model_extra_config.operator_opt_config.prefill_enable_mla_alltoall:
            self.o_proj = ReplicatedLinear(self.num_heads * self.v_head_dim,
                                            hidden_size,
                                            bias=False,
                                            quant_config=quant_config,
                                            prefix=f"{prefix}.o_proj")
        else:
            self.o_proj = RowParallelLinearWithReduceScatter(self.num_heads * self.v_head_dim,
                                            self.hidden_size,
                                            bias=False,
                                            quant_config=quant_config,
                                            prefix=f"{prefix}.o_proj")

        rope_scaling["rope_type"] = 'deepseek_yarn'
        self.rotary_emb = get_rope(qk_rope_head_dim,
                                   rotary_dim=qk_rope_head_dim,
                                   max_position=max_position_embeddings,
                                   base=rope_theta,
                                   rope_scaling=rope_scaling,
                                   is_neox_style=False)

        if rope_scaling:
            mscale_all_dim = rope_scaling.get("mscale_all_dim", False)
            scaling_factor = rope_scaling["factor"]
            mscale = yarn_get_mscale(scaling_factor, float(mscale_all_dim))
            self.scaling = self.scaling * mscale * mscale

        self.attn_mla = Attention(
            num_heads=self.num_local_heads,
            head_size=self.qk_head_dim,
            scale=self.scaling,
            use_mla=True,
            num_kv_heads=self.num_local_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
            # MLA Args
            q_lora_rank=self.q_lora_rank,
            kv_lora_rank=self.kv_lora_rank,
            qk_nope_head_dim=self.qk_nope_head_dim,
            qk_rope_head_dim=self.qk_rope_head_dim,
            qk_head_dim=self.qk_head_dim,
            v_head_dim=self.v_head_dim,
            rotary_emb=self.rotary_emb,
            q_proj=self.q_proj if self.q_lora_rank is None else self.q_b_proj,
            kv_a_proj_with_mqa=self.kv_a_proj_with_mqa if hasattr(self, 'kv_a_proj_with_mqa') else None,
            kv_a_layernorm=self.kv_a_layernorm,
            kv_b_proj=self.kv_b_proj,
            o_proj=self.o_proj,
            qkv_a_proj=self.qkv_a_proj if hasattr(self, 'qkv_a_proj') else None,
            q_a_layernorm=self.q_a_layernorm if hasattr(self, 'q_a_layernorm') else None,
            q_b_proj=self.q_b_proj if hasattr(self, 'q_b_proj') else None,
            q_a_proj=self.q_a_proj if hasattr(self, 'q_a_proj') else None
        )


    def forward(
            self,
            positions: torch.Tensor,
            hidden_states: torch.Tensor,
            kv_cache: torch.Tensor,
            attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:

        return self.attn_mla.impl.forward(
            positions=positions,
            hidden_states=hidden_states,
            kv_cache=kv_cache,
            attn_metadata=attn_metadata)


class DeepseekDecoderLayer(nn.Module):

    def __init__(
            self,
            config: PretrainedConfig,
            prefix: str,
            cache_config: Optional[CacheConfig] = None,
            quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        super().__init__()
        self.prefix = prefix
        self.hidden_size = config.hidden_size
        rope_theta = getattr(config, "rope_theta", 10000)
        rope_scaling = getattr(config, "rope_scaling", None)
        max_position_embeddings = getattr(config, "max_position_embeddings",
                                          8192)
        # DecoderLayers are created with `make_layers` which passes the prefix
        # with the layer's index.
        layer_idx = int(prefix.split(sep='.')[-1])
        self.self_attn = AscendDeepseekAttention_MLA(
            config=config,
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            qk_nope_head_dim=config.qk_nope_head_dim,
            qk_rope_head_dim=config.qk_rope_head_dim,
            v_head_dim=config.v_head_dim,
            q_lora_rank=config.q_lora_rank
            if hasattr(config, "q_lora_rank") else None,
            kv_lora_rank=config.kv_lora_rank,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            max_position_embeddings=max_position_embeddings,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
        )
        if (config.n_routed_experts is not None
                and layer_idx >= config.first_k_dense_replace
                and layer_idx % config.moe_layer_freq == 0):
            self.mlp = DeepseekMoE(
                config=config,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )
            self.is_moe = True
        else:
            self.mlp = ParallelDeepseekMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )
            self.is_moe = False
        self.input_layernorm = RMSNorm(config.hidden_size,
                                       eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size,
                                                eps=config.rms_norm_eps)

        self.enable_dp_attention = get_data_parallel_world_size() > 1
        if self.enable_dp_attention:
            self.dp_rank = get_data_parallel_rank()
            self.dp_size = get_data_parallel_world_size()
            self.dp_group = get_dp_group().device_group
        else:
            self.dp_rank = None
            self.dp_size = None
            self.dp_group = None

    def forward(
            self,
            positions: torch.Tensor,
            hidden_states: torch.Tensor,
            kv_cache: torch.Tensor,
            attn_metadata: AttentionMetadata,
            residual: Optional[torch.Tensor],
            layer_id: Optional[int] = None
    ) -> torch.Tensor:
        # Self Attention
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            # Adapt: adapt for w8a8 dynamic, do quant
            # Combines residual add and rmsnorm
            hidden_states, residual = self.input_layernorm(
                hidden_states, residual, quant_symbol=(not model_extra_config.operator_opt_config.use_mlaprolog))
            # Adapt end.
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            kv_cache=kv_cache,
            attn_metadata=attn_metadata,
        )

        is_prefill = attn_metadata is None or attn_metadata.prefill is not None

        # TODO 调试稳定后再加，只能在图模式下使用
        if self.is_moe == True and not is_prefill:
            torch_npu.npu_prefetch(self.mlp.experts.w13_weight, hidden_states, FFN1_PREFETCH_SIZE, 0)

        if self.is_moe == True and not is_prefill and model_extra_config.operator_opt_config.use_super_kernel:
            with tng.scope.super_kernel(self.mlp.prefix, 'stream-fusion=1'):
                hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        else:
            hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        # hidden : tokens * 7168

        # Perform full hidden splitting to avoid OOM
        if model_extra_config.parall_config.dp_size > 1 and attn_metadata is not None and attn_metadata.prefill is not None:
            reduce_length = torch.tensor(hidden_states.shape[0], dtype=torch.int64, device=current_platform.device_type)
            local_length = hidden_states.shape[0]
            # global_max_length = torch.tensor(0, dtype=torch.int64)
            dist.all_reduce(reduce_length, op=dist.ReduceOp.MAX, async_op=False)
            global_max_length = reduce_length.item()
            pad_size = global_max_length - hidden_states.shape[0]
            hidden_states = torch.nn.functional.pad(
                hidden_states, (0, 0, 0, pad_size)
            )
            residual = torch.nn.functional.pad(
                residual, (0, 0, 0, pad_size)
            )
            hidden_states_list = hidden_states.split(SEQ_SPLIT_LENGTH_BEFORE_ALL_GATHER)
            residual_list = residual.split(SEQ_SPLIT_LENGTH_BEFORE_ALL_GATHER)
            hidden_state_out = []
            residual_out = []
            for i in range(len(hidden_states_list)):
                hidden_states, residual = self.mlp(hidden_states_list[i], residual_list[i], attn_metadata, layer_id)
                hidden_state_out.append(hidden_states)
                residual_out.append(residual)
            hidden_states = torch.cat(hidden_state_out)[:local_length]
            residual = torch.cat(residual_out)[:local_length]
        else:
            if self.is_moe == True:
                if not is_prefill and model_extra_config.operator_opt_config.use_super_kernel:
                    with tng.scope.super_kernel(self.mlp.prefix, 'stream-fusion=1'):
                        hidden_states, residual = self.mlp(hidden_states, residual, attn_metadata, layer_id)
                else:
                    hidden_states, residual = self.mlp(hidden_states, residual, attn_metadata, layer_id)
                if isinstance(hidden_states, (tuple, list)):
                    assert len(hidden_states) == 2
                    # 0 is the shared expert hidden_states, 1 is the routing expert hidden_states, add operation cannot be placed in the super kernel
                    hidden_states = hidden_states[0] + hidden_states[1]
            else:
                hidden_states, residual = self.mlp(hidden_states, residual, attn_metadata)

        return hidden_states, residual

    CACHED_GATHERED_BUFFER = None

    def get_cached_gathered_buffer(self, token_nums, dtype):
        if DeepseekDecoderLayer.CACHED_GATHERED_BUFFER is None \
                or DeepseekDecoderLayer.CACHED_GATHERED_BUFFER.shape[0] != token_nums \
                or DeepseekDecoderLayer.CACHED_GATHERED_BUFFER.dtype != dtype:
            DeepseekDecoderLayer.CACHED_GATHERED_BUFFER = torch.zeros((token_nums, self.hidden_size), dtype=dtype,
                                                                      device=current_platform.device_type)
        return DeepseekDecoderLayer.CACHED_GATHERED_BUFFER

    def data_parallel_all_gather(
            self, input_tensor: torch.Tensor, attn_metadata):
        if self.dp_size == 1:
            return input_tensor

        global_num_tokens = attn_metadata.global_num_tokens.tolist()
        all_lens = global_num_tokens
        max_len = max(global_num_tokens)
        need_pad = min(global_num_tokens) != max_len

        gathered_buffer = self.get_cached_gathered_buffer(max_len * self.dp_size, input_tensor.dtype)

        if need_pad:
            padded_tensor = torch.nn.functional.pad(
                input_tensor, (0, 0, 0, max_len - input_tensor.shape[0])
            )
        else:
            padded_tensor = input_tensor

        torch.distributed.all_gather_into_tensor(
            gathered_buffer, padded_tensor, group=self.dp_group
        )

        if need_pad:
            select_index = list(
                itertools.chain(*(range(i * max_len, i * max_len + all_lens[i]) for i in range(self.dp_size))))
            gathered_tensors = gathered_buffer[select_index, :]
        else:
            gathered_tensors = gathered_buffer

        start_index = 0 if self.dp_rank == 0 else sum(all_lens[:self.dp_rank])
        end_index = start_index + all_lens[self.dp_rank]
        return gathered_tensors, start_index, end_index


# @support_torch_compile
class DeepseekV3Model(nn.Module):
    fall_back_to_pt_during_load = False

    def __init__(self,
                 config: PretrainedConfig,
                 cache_config: Optional[CacheConfig] = None,
                 quant_config: Optional[QuantizationConfig] = None,
                 prefix: str = "",
                 ):
        super().__init__()

        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        if get_pp_group().is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
            )
        else:
            self.embed_tokens = PPMissingLayer()

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: DeepseekDecoderLayer(
                config,
                prefix,
                cache_config=cache_config,
                quant_config=quant_config,
            ),
            prefix=f"{prefix}.layers")

        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()
        self.make_empty_intermediate_tensors = (
            make_empty_intermediate_tensors_factory(
                ["hidden_states", "residual"], config.hidden_size))

        self.enable_dp_attention = get_data_parallel_world_size() > 1
        if self.enable_dp_attention:
            self.dp_rank = get_data_parallel_rank()
            self.dp_size = get_data_parallel_world_size()
            self.dp_group = get_dp_group().device_group

    CACHED_LOCAL_NUM_TOKENS = None
    CACHED_GLOBAL_NUM_TOKENS = None
 

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids, reduce=1)

    def forward(
            self,
            input_ids: torch.Tensor,
            positions: torch.Tensor,
            kv_caches: List[torch.Tensor],
            attn_metadata: AttentionMetadata,
            intermediate_tensors: Optional[IntermediateTensors],
    ) -> Union[torch.Tensor, IntermediateTensors]:
        # adapt dp allgather batchinfo

        if get_pp_group().is_first_rank:
            hidden_states = self.get_input_embeddings(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        for i in range(self.start_layer, self.end_layer):
            layer = self.layers[i]
            layer_id = i - 3
            hidden_states, residual = layer(positions,
                                            hidden_states,
                                            kv_caches[i - self.start_layer] if kv_caches is not None else None,
                                            attn_metadata, 
                                            residual,
                                            layer_id)

        if not get_pp_group().is_last_rank:
            return IntermediateTensors({
                "hidden_states": hidden_states,
                "residual": residual
            })

        hidden_states, _ = self.norm(hidden_states, residual)

        hidden_states = tensor_model_parallel_all_gather(hidden_states, dim=0)

        return hidden_states


class DeepseekV3ForCausalLM(nn.Module, GraphCompileConfiguration):

    packed_modules_mapping = {
        "gate_up_proj": ["gate_proj", "up_proj"],
        "experts":
        ["experts.0.gate_proj", "experts.0.up_proj", "experts.0.down_proj"]
    }
    
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        self.config = vllm_config.model_config.hf_config
        self.quant_config = vllm_config.quant_config
        self.model = DeepseekV3Model(self.config,
                                     vllm_config.cache_config,
                                     self.quant_config,
                                     prefix="model")
        self.lm_head = ParallelLMHead(self.config.vocab_size,
                                      self.config.hidden_size,
                                      quant_config=self.quant_config,
									  parallel_lmhead=(model_extra_config.parall_config.dp_size > 1))
        self.logits_processor = LogitsProcessor(self.config.vocab_size,
                                                logits_as_input=True)
        self.sampler = Sampler()
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors)

        self.return_hidden_states = True
        self.input_marked=False

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embeddings(input_ids)

    def forward(
            self,
            input_ids: torch.Tensor,
            positions: torch.Tensor,
            kv_caches: List[torch.Tensor] = None,
            attn_metadata: AttentionMetadata = None,
            prefill_padding_or_selected_indices: Optional[torch.Tensor] = None,
            intermediate_tensors: Optional[IntermediateTensors] = None,
            inputs_embeds = None
    ) -> Optional[torch.Tensor]:
        hidden_states = self.model(input_ids, positions, kv_caches,
                                   attn_metadata, intermediate_tensors)
        if attn_metadata is None:
            logits = self.logits_processor._get_logits(hidden_states[-1:, ...], self.lm_head, None)
        else:
            logits = self.compute_lmhead(self.lm_head, hidden_states, prefill_padding_or_selected_indices)

        if self.return_hidden_states:
            return hidden_states, logits
        else:
            return logits

    def compute_lmhead(
            self,
            lm_head: VocabParallelEmbedding,
            hidden_states: torch.Tensor,
            prefill_padding_or_selected_indices: Optional[torch.Tensor] = None,
            embedding_bias: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        if prefill_padding_or_selected_indices is not None:
            hidden_states = _prune_hidden_states(hidden_states, prefill_padding_or_selected_indices)

        # Get the logits for the next tokens.
        if model_extra_config.parall_config.dp_size > 1:
            logits = self.logits_processor._get_logits_decode(hidden_states, lm_head, embedding_bias)
        else:
            logits = self.logits_processor._get_logits(hidden_states, lm_head, embedding_bias)
        return logits

    def compute_logits(
            self,
            hidden_states: torch.Tensor,
            sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        logits = self.logits_processor(self.lm_head, hidden_states,
                                       sampling_metadata)

        return logits

    def sample(
            self,
            logits: Optional[torch.Tensor],
            sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens

    def make_empty_intermediate_tensors(
            self, batch_size: int, dtype: torch.dtype,
            device: torch.device) -> IntermediateTensors:
        return IntermediateTensors({
            "hidden_states":
                torch.zeros((batch_size, self.config.hidden_size),
                            dtype=dtype,
                            device=device),
            "residual":
                torch.zeros((batch_size, self.config.hidden_size),
                            dtype=dtype,
                            device=device),
        })

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> Set[str]:
        if model_extra_config.operator_opt_config.merge_qkv:
            stacked_params_mapping = [
                # (param_name, shard_name, shard_id)
                ("gate_up_proj", "gate_proj", 0),
                ("gate_up_proj", "up_proj", 1),
                ("qkv_a_proj", "q_a_proj", 0),
                ("qkv_a_proj", "kv_a_proj_with_mqa", 1),
            ]
        else:
            stacked_params_mapping = [
                # (param_name, shard_name, shard_id)
                ("gate_up_proj", "gate_proj", 0),
                ("gate_up_proj", "up_proj", 1),
            ]

        # Params for weights, fp8 weight scales, fp8 activation scales
        # (param_name, weight_name, expert_id, shard_id)
        expert_params_mapping = FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.n_routed_experts)

        params_dict = dict(self.named_parameters())
        loaded_params: Set[str] = set()
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue

            if self.config.architectures[0] == 'DeepseekV3ForCausalLM' and self.config.num_nextn_predict_layers > 0:
                assert self.config.num_nextn_predict_layers == 1
                layer_idx = self.config.num_hidden_layers
                if name.startswith(f"model.layers.{layer_idx}"):
                    continue

            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                # Skip non-stacked layers and experts (experts handled below).
                if weight_name not in name:
                    continue
                # We have mlp.experts[0].gate_proj in the checkpoint.
                # Since we handle the experts below in expert_params_mapping,
                # we need to skip here BEFORE we update the name, otherwise
                # name will be updated to mlp.experts[0].gate_up_proj, which
                # will then be updated below in expert_params_mapping
                # for mlp.experts[0].gate_gate_up_proj, which breaks load.
                if (("mlp.experts." in name) and name not in params_dict):
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue

                if is_pp_missing_parameter(name, self):
                    continue
                if name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping
                    if weight_name not in name:
                        continue
                    name = name.replace(weight_name, param_name)

                    if is_pp_missing_parameter(name, self):
                        continue
                    if name not in params_dict:
                        continue
                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    weight_loader(param,
                                  loaded_weight,
                                  name,
                                  shard_id=shard_id,
                                  expert_id=expert_id)
                    break
                else:
                    # Skip loading extra bias for GPTQ models.
                    if name.endswith(".bias") and name not in params_dict:
                        continue

                    if is_pp_missing_parameter(name, self):
                        continue
                    if name not in params_dict:
                        continue
                    if name not in params_dict:
                        continue
                    param = params_dict[name]
                    weight_loader = getattr(param, "weight_loader",
                                            default_weight_loader)
                    weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


    def mark_static_for_graph(self, input_ids, positions, attn_metadata, kv_caches):
        if not self.input_marked:
            torch._dynamo.mark_static(input_ids)
            torch._dynamo.mark_static(positions)
            if attn_metadata.decode.cos is not None:
                torch._dynamo.mark_static(attn_metadata.decode.cos)
            if attn_metadata.decode.sin is not None:
                torch._dynamo.mark_static(attn_metadata.decode.sin)
            if attn_metadata.decode.mc2_mask is not None:
                torch._dynamo.mark_static(attn_metadata.decode.mc2_mask)
            if attn_metadata.decode.best_topk is not None:
                torch._dynamo.mark_static(attn_metadata.decode.best_topk)
            if attn_metadata.decode.block_table is not None:
                torch._dynamo.mark_static(attn_metadata.decode.block_table)
            if attn_metadata.decode.seq_lens is not None:
                torch._dynamo.mark_static(attn_metadata.decode.seq_lens)
            if attn_metadata.slot_mapping is not None:
                torch._dynamo.mark_static(attn_metadata.slot_mapping)
            for i in range(len(kv_caches)):
                if kv_caches[i][0] is not None:
                    torch._dynamo.mark_static(kv_caches[i][0])
                if kv_caches[i][1] is not None:
                    torch._dynamo.mark_static(kv_caches[i][1])
            self.input_marked = True
        # adapt 1. baichuan2-13b do not use rope, so att do not have rotary_emb attribute
        # adapt 2. DeepseekV2 rotary_emb do not have sin, cos attribute
        # if hasattr(self.model.layers[0].self_attn, "rotary_emb") and self.config.architectures[0] not in  ['DeepseekV2ForCausalLM', 'DeepseekV3ForCausalLM']:
        #     torch._dynamo.mark_static(self.model.layers[0].self_attn.rotary_emb.sin)
        #     torch._dynamo.mark_static(self.model.layers[0].self_attn.rotary_emb.cos)