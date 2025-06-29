# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

from typing import Iterable, List, Optional, Tuple, Set

import os
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
 
from vllm.config import CacheConfig, QuantizationConfig, VllmConfig
from vllm.attention.backends.abstract import AttentionMetadata
from vllm.distributed.communication_op import tensor_model_parallel_all_gather
 
from vllm.model_executor.models.utils import (
    PPMissingLayer, 
    is_pp_missing_parameter, 
    make_layers, 
    make_empty_intermediate_tensors_factory,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.sequence import IntermediateTensors
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.model_executor.layers.sampler import Sampler
from .deepseek_v3 import DeepseekDecoderLayer
 
from vllm.distributed.parallel_state import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
 
 
 
from transformers import PretrainedConfig
 
from omni.models.common.layers.layernorm import RMSNorm #zxp: not use
from omni.models.common.layers.vocab_parallel_embedding import (
    ParallelLMHead, 
    VocabParallelEmbedding
)
from omni.models.common.layers.fused_moe.layer import FusedMoE
from omni.models.common.layers.logits_processor import _prune_hidden_states
from omni.models.common.layers.logits_processor import LogitsProcessor
from omni.models.common.config.model_config import model_extra_config
from omni.adaptors.vllm.worker.npu_model_runner import GraphCompileConfiguration 
 
class DeepseekV3MTP(nn.Module, GraphCompileConfiguration):
    def __init__(self,
                 config: PretrainedConfig,
                 cache_config: Optional[CacheConfig] = None,
                 quant_config: Optional[QuantizationConfig] = None,
                 prefix: str = "",
                 layer_index: int = 61,
                 ):
        super().__init__()
        self.config = config
        prefix = "model"
        self.cache_config = cache_config
        self.quant_config = quant_config
        self.ignore_share_weight = True
        if self.ignore_share_weight:
            self.embed_tokens = None
            self.shared_head = nn.ModuleDict({
                "head": None,
                "norm": RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            })
        else:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
            )
            self.shared_head = nn.ModuleDict({
                "head": ParallelLMHead(config.vocab_size, config.hidden_size, quant_config=self.quant_config),
                "norm": RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            })
 
        self.enorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hnorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
 
        self.eh_proj = nn.Linear(2 * config.hidden_size, config.hidden_size, bias=False)
        self.decoderlayer = DeepseekDecoderLayer(config,
                                                 f"{prefix}.layers.{layer_index}",
                                                 quant_config=self.quant_config,
                                                 cache_config=self.cache_config)
 
        self.logits_processor = LogitsProcessor(config.vocab_size, logits_as_input=True)
        self.greedy_sampler = Sampler()
 
    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids, reduce=1)
 
    def forward(
            self,
            input_ids: torch.Tensor,
            positions: torch.Tensor,
            kv_caches: List[torch.Tensor],
            attn_metadata: AttentionMetadata,
            previous_hidden_states: torch.Tensor,
            prefill_padding_or_selected_indices: Optional[torch.Tensor] = None,
            intermediate_tensors: Optional[IntermediateTensors] = None,
            require_hidden_states: Optional[bool] = False,
            inputs_embeds = None
    ) -> torch.Tensor:
        tok_embeds = self.enorm(self.get_input_embeddings(input_ids))
        if len(tok_embeds.shape) > 2:
            tok_embeds = tok_embeds.view(-1, self.config.hidden_size)
 
        tp_size = get_tensor_model_parallel_world_size()  # cloud: get_tp_group().world_size
        rank_in_group = get_tensor_model_parallel_rank()
 
        if tp_size > 1:
            token_num = previous_hidden_states.shape[0]
            start_range = rank_in_group * (token_num // tp_size)
            end_range = (1 + rank_in_group) * (token_num // tp_size)
            previous_hidden_states = previous_hidden_states[start_range: end_range, :]
 
        previous = self.hnorm(previous_hidden_states)  
        cat_hidden_states = torch.cat([tok_embeds, previous], dim=-1)
        inputs_embeds = self.eh_proj.forward(cat_hidden_states)
 
        encoded_states, residual = self.decoderlayer(
            positions=positions,
            kv_cache=kv_caches[0] if kv_caches is not None else None,
            hidden_states=inputs_embeds,
            attn_metadata=attn_metadata,
            residual=None
        )
 
        hidden_states, _ = self.shared_head["norm"](encoded_states, residual)
 
        hidden_states = tensor_model_parallel_all_gather(hidden_states, dim=0)
 
        if attn_metadata is None:
            logits = self.logits_processor._get_logits(hidden_states[-1:, ...], self.shared_head['head'], None)
        else:
            logits = self.compute_lmhead(self.shared_head['head'], hidden_states, prefill_padding_or_selected_indices)
 
        if require_hidden_states:
            return logits, hidden_states
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
 
    @property
    def sampler(self):
        return self.greedy_sampler
 
    def compute_logits(
            self,
            hidden_states: torch.Tensor,
            sampling_metadata: SamplingMetadata
    ) -> torch.Tensor:
        logits = self.logits_processor(self.shared_head["head"], hidden_states, sampling_metadata)
        return logits
 
    def load_weights(self, weights: Iterable[Tuple[str,
                                                   torch.Tensor]], layer_idx: int = 61) -> Set[str]:
 
        stacked_params_mapping = [
            # 字段说明: (param_name, shard_name, shard_id)
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
 
        # Params for weights, fp8 weight scales, fp8 activation scales
        # 字段说明: (param_name, weight_name, expert_id, shard_id)
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
            if self.ignore_share_weight and any(
                    substring in name for substring in ["embed_tokens.weight", "shared_head.head"]):
                continue
            if name.startswith(f"model.layers.{layer_idx}"):
                name = name.replace(f"model.layers.{layer_idx}.", "")
                if (name.startswith("input_layernorm") or
                        name.startswith("post_attention_layernorm") or
                        name.startswith("mlp") or
                        name.startswith("self_attn")):
                    name = "decoderlayer." + name
            else:
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
                    breakpoint()
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
                        breakpoint()
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
                        breakpoint()
                    param = params_dict[name]
                    weight_loader = getattr(param, "weight_loader",
                                            default_weight_loader)
                    weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params