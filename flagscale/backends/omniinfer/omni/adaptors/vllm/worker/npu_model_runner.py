#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
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
# This file is a part of the vllm-ascend project.
# Adapted from vllm-project/vllm/vllm/worker/gpu_model_runner.py
#

import copy
import gc
import os
import time
import weakref
from typing import TYPE_CHECKING, Dict, List, Optional, Union

import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
import torch.distributed as dist
from vllm.attention import AttentionType, get_attn_backend
from vllm.attention.layer import Attention
from vllm.config import CompilationLevel, VllmConfig
from vllm.distributed.parallel_state import get_pp_group, get_tensor_model_parallel_world_size
from vllm.forward_context import set_forward_context, get_forward_context
from vllm.inputs import INPUT_REGISTRY
from vllm.logger import logger
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.model_loader import get_model
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalKwargs
from vllm.sampling_params import SamplingType
from vllm.sequence import IntermediateTensors
from vllm.utils import (STR_DTYPE_TO_TORCH_DTYPE, DeviceMemoryProfiler,
                        LayerBlockType, LazyLoader, cdiv)
from vllm.v1.core.encoder_cache_manager import compute_encoder_budget
from vllm.v1.kv_cache_interface import (AttentionSpec, FullAttentionSpec,
                                        KVCacheConfig, KVCacheSpec,
                                        SlidingWindowSpec)
from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT, ModelRunnerOutput
from vllm.v1.sample.sampler import Sampler
from vllm.v1.utils import bind_kv_cache
from vllm.v1.worker.gpu_input_batch import CachedRequestState, InputBatch

from omni.models.common.layers.attention.attention import AttentionMaskBuilder
from omni.models.common.layers.attention.attention import AscendAttentionState
from omni.models.common.layers.sampler import SimpleSampler
from omni.adaptors.vllm.platform import NPUPlatform
from vllm.distributed.parallel_state import get_dp_group
from vllm.distributed.kv_transfer import (get_kv_transfer_group,
                                          has_kv_transfer_group)
from vllm.distributed.kv_transfer.kv_connector.v1 import KVConnectorBase_V1

from vllm.attention.backends.abstract import (AttentionBackend,
                                              AttentionMetadataBuilder)

from vllm.v1.worker.gpu_model_runner import GPUModelRunner
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata
import vllm.envs as envs

from abc import abstractmethod, ABCMeta

omni_use_dsv3 = int(os.getenv("OMNI_USE_DSV3", "0"))

if TYPE_CHECKING:
    import xgrammar as xgr  # type: ignore[import-untyped]
    from vllm.v1.core.sched.output import SchedulerOutput
else:
    xgr = LazyLoader("xgr", globals(), "xgrammar")

from omni.models.common.config.model_config import model_extra_config
if model_extra_config.operator_opt_config.use_omni_placement:
    from omni_planner import OmniPlanner
    _GLOBAL_STEP = 0

MAX_GEAR_NUM = 6
def _get_pad_size(num_seqs):
    tp_size = get_tensor_model_parallel_world_size()
    return (tp_size - num_seqs % tp_size) % tp_size

class GraphCompileConfiguration:
    """
    When the graph mode is turned on
    you can set the gear or clarify the static shape by inheriting this class to speed up the model running
    """

    def set_dynamic_gears(self, *args, **kwargs):
        pass


    def mark_static_for_graph(self, *args, **kwargs):
        torch._dynamo.mark_static(args[0])
        torch._dynamo.mark_static(args[1])

class DummyAttentionMetadataBuilder(metaclass=ABCMeta):
    """
    When Model DP is turned on, the idle DP needs to build fake data to run with it.
    At this time, the attention metadata builder needs to inherit this interface to implement build_dummy method.
    """

    @abstractmethod
    def build_dummy(self, *args, **kwargs):
        pass

class NPUModelRunner(GPUModelRunner):
    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        super().__init__(vllm_config, device)
        self.head_size = self.model_config.get_head_size()
        self.block_size = vllm_config.cache_config.block_size
        self.attn_backend = get_attn_backend(
            self.head_size,
            self.dtype,
            self.kv_cache_dtype,
            self.block_size,
            self.model_config.is_attention_free,
            use_mla=self.model_config.use_mla,
        )
        if self.attn_backend is None:
            error_msg = (
                f"Error with get_att_backend: {self.head_size=}, "
                f"{self.dtype=}, {self.kv_cache_dtype=}, {self.block_size=}, "
                f"{self.model_config.is_attention_free=}, "
                f"{self.model_config.use_mla=}")
            logger.error(error_msg)
            raise NotImplementedError(
                "Non-Attention backend is not supported by V1 NPUModelRunner.")

        self.attn_metadata_builder = self.attn_backend.get_builder_cls()(
            weakref.proxy(self))

        self.num_attn_layers = self.model_config.get_num_layers_by_block_type(
            vllm_config.parallel_config, LayerBlockType.attention)
        self.scheduler_config = vllm_config.scheduler_config
        self.speculative_config = vllm_config.speculative_config
        self.max_num_reqs = self.scheduler_config.max_num_seqs
        if self.use_spec_decode:
            self.rejection_sampler = SimpleSampler(self.sampler)

        additional_config = vllm_config.additional_config
        self._init_graph_options(additional_config)

        self.slot_mapping_cpu = torch.zeros(self.max_num_tokens,
                                            dtype=torch.int64,
                                            device="cpu",
                                            pin_memory=True)
        self.slot_mapping_np = self.slot_mapping_cpu.numpy()
        self.input_ids = torch.zeros(self.max_num_tokens,
                                     dtype=torch.int64,
                                     device=self.device)
        self.input_ids_cpu = torch.zeros(self.max_num_tokens,
                                         dtype=torch.int64,
                                         device="cpu",
                                         pin_memory=self.pin_memory)
        self.seq_lens = torch.zeros(self.max_num_reqs,
                                    dtype=torch.int64,
                                    device=self.device)
        self.seq_lens_cpu = torch.zeros(self.max_num_reqs,
                                        dtype=torch.int64,
                                        device="cpu",
                                        pin_memory=self.pin_memory)
        self.seq_lens_np = self.seq_lens_cpu.numpy()
        # TODO: support arbitrary spec tokens
        self.graph_block_tables = np.zeros(
            (self.max_num_reqs if not self.use_spec_decode else self.max_num_reqs * 2,
             (self.model_config.max_model_len + self.block_size - 1) //
             self.block_size),
            dtype=np.int32)
        self.attn_mask = None
        self.attn_state = None
        self.max_num_blocks_per_req = cdiv(self.model_config.max_model_len,
                                           self.block_size)

        mask_len = os.getenv("PAGED_ATTENTION_MASK_LEN", 10000)
        self.attn_mask = None
        self.attn_state = None
        self.attn_mask_len = min(self.model_config.max_model_len,
                                 int(mask_len))
        self.attn_mask_builder = AttentionMaskBuilder.initialize_from_len(
            self.attn_mask_len, self.dtype)
        
        self.drafter_mark_static = False
        self.dummy_drafter_mark_static = False

    def _init_graph_options(self, additional_config):
        self.enable_torchair_graph_mode = False
        self.use_cached_npu_graph = False
        self.decode_gear_list = []
        self.decode_gear_list_ori = []
        self.max_batch_size = self.max_num_reqs if not self.use_spec_decode else self.max_num_reqs * 2
        
        if additional_config:
            self.enable_torchair_graph_mode = additional_config.get(
                "enable_graph_mode",
                False) and self.vllm_config.model_config.use_mla
            self.use_cached_npu_graph = additional_config.get(
                "use_cached_npu_graph", False)
            self.decode_gear_list_ori = additional_config.get(
                "decode_gear_list", [])
            if not isinstance(self.decode_gear_list_ori, list):
                raise TypeError("decode_gear_list must be list[int]")
            if len(self.decode_gear_list) > MAX_GEAR_NUM:
                raise ValueError(f"Max gear num supported is {MAX_GEAR_NUM} now.")
        
            if self.decode_gear_list and self.max_batch_size != max(self.decode_gear_list):
                self.decode_gear_list = [gear for gear in self.decode_gear_list_ori if gear < self.max_batch_size] + [self.max_batch_size]
                logger.warning(f"PTA_TORCHAIR_DECODE_GEAR_LIST({self.decode_gear_list_ori}) becomes ({self.decode_gear_list}) due to max_batch_size({self.max_batch_size})")
            else:
                self.decode_gear_list = self.decode_gear_list_ori # List of categories

        if len(self.decode_gear_list) == 0:
            self.decode_gear_list = [
                self.max_batch_size
            ]

    def _may_reorder_batch(self, scheduler_output: "SchedulerOutput") -> bool:
        """
        Update the order of requests in the batch based on the attention
        backend's needs. For example, some attention backends (namely MLA) may 
        want to separate requests based on if the attention computation will be
        compute-bound or memory-bound.

        Args:
            scheduler_output: The scheduler output.

        Returns:
            True if the batch was reordered, False otherwise.
        """
        if not self.attn_metadata_builders:
            return False
        
        batch_reordered = self.attn_metadata_builders[0].reorder_batch(
            self.input_batch, scheduler_output)

        # For models with multiple KV cache groups, the groups should agree on
        # the same order of requests. We ensure this by only allowing the first
        # group to reorder the batch and asserting that all other groups do not
        # reorder the batch.
        for i in range(1, len(self.kv_cache_config.kv_cache_groups)):
            if self.attn_metadata_builders[i].reorder_batch(self.input_batch, scheduler_output):
                raise RuntimeError("reorder_batch returned True, which is not expected")
        return batch_reordered

    def _make_attention_mask(self, seq_lens, query_lens, position,
                             attn_state) -> torch.Tensor:
        # Chunk Prefill situation.
        if attn_state == AscendAttentionState.ChunkedPrefill:
            return self.attn_mask_builder.get_splitfuse_attn_mask(
                seq_lens, query_lens, position, self.dtype, self.device)
        # Prefill without cache situation.
        elif attn_state == AscendAttentionState.PrefillNoCache:
            max_seq_len = max(seq_lens, default=0)
            return self.attn_mask_builder.get_attn_mask(
                max_seq_len, self.dtype, self.device)
        # Prefill with cache hit.
        elif attn_state == AscendAttentionState.PrefillCacheHit:
            return self.attn_mask_builder.get_attn_mask(
                128, self.dtype, self.device)
        # Decode-only situation.
        else:
            return None

    def _prepare_inputs(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> torch.Tensor:
        # Check input valid
        total_num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
        if total_num_scheduled_tokens <= 0:
            raise RuntimeError("total_num_scheduled_tokens must be greater than 0")
        num_reqs = self.input_batch.num_reqs
        if num_reqs <= 0:
            raise RuntimeError("num_reqs must be greater than 0")
        num_input_tokens = total_num_scheduled_tokens
        logger.warning(f"current num reqs = {num_reqs}, num_input_tokens = {num_input_tokens}")
        modified_batch = self.attn_metadata_builder.reorder_batch(
            self.input_batch, scheduler_output)
        if modified_batch:
            self.input_batch.refresh_sampling_metadata()

        # OPTIMIZATION: Start copying the block table first.
        # This way, we can overlap the copy with the following CPU operations.
        self.input_batch.block_table.commit(num_reqs)

        # Get the number of scheduled tokens for each request.
        num_scheduled_tokens = np.empty(num_reqs, dtype=np.int32)
        num_scheduled_spec_decode_tokens = len(scheduler_output.scheduled_spec_decode_tokens)
        max_num_scheduled_tokens = 0
        for i, req_id in enumerate(self.input_batch.req_ids):
            num_tokens = scheduler_output.num_scheduled_tokens[req_id]
            num_scheduled_tokens[i] = num_tokens
            max_num_scheduled_tokens = max(max_num_scheduled_tokens,
                                           num_tokens)

        # Prepare positions
        req_indices = np.repeat(self.arange_np[:num_reqs],
                                num_scheduled_tokens)
        cu_num_tokens = np.cumsum(num_scheduled_tokens)
        cumsums_offsets = np.repeat(cu_num_tokens - num_scheduled_tokens,
                                    num_scheduled_tokens)
        sample_indices = cu_num_tokens - 1
        sample_indices = torch.from_numpy(sample_indices).to(self.device,
                                                             non_blocking=True)
        arange = self.arange_np[:total_num_scheduled_tokens] - cumsums_offsets

        positions_np = self.positions_np[:total_num_scheduled_tokens]
        np.add(self.input_batch.num_computed_tokens_cpu[req_indices],
               arange,
               out=positions_np)

        self.positions[:total_num_scheduled_tokens].copy_(
            self.positions_cpu[:total_num_scheduled_tokens], non_blocking=True)
        positions = self.positions[:num_input_tokens]
        self.query_lens = torch.from_numpy(num_scheduled_tokens)

        self.seq_lens_np[:num_reqs] = (
            self.input_batch.num_computed_tokens_cpu[:num_reqs] +
            num_scheduled_tokens)
        seq_lens = self.seq_lens_cpu[:num_reqs]

        block_table_indices = (req_indices * self.max_num_blocks_per_req +
                               positions_np // self.block_size)
        block_table_cpu = self.input_batch.block_table[0].get_cpu_tensor()
        block_numbers = block_table_cpu.flatten()[block_table_indices].numpy()
        block_offsets = positions_np % self.block_size
        np.add(block_numbers * self.block_size,
               block_offsets,
               out=self.slot_mapping_np[:total_num_scheduled_tokens])

        if np.array_equal(self.seq_lens_np[:num_reqs], num_scheduled_tokens):
            attn_state = AscendAttentionState.PrefillNoCache
        # We assume it is the decode stage, where prefill occurs but only one token is not hit in cache.
        elif np.all(num_scheduled_tokens == 1) or num_scheduled_spec_decode_tokens == num_reqs:
            attn_state = AscendAttentionState.DecodeOnly
        # splitfuse
        else:
            attn_state = AscendAttentionState.ChunkedPrefill

        self.attn_state = attn_state
		# deepseek v3 requires padding
        if attn_state == AscendAttentionState.DecodeOnly:
            if num_reqs > self.max_batch_size:
                raise RuntimeError("num_reqs is bigger than max_batch_size")
            graph_pad_size = self.max_batch_size - num_reqs
            if self.use_spec_decode:
                graph_pad_size = self.max_batch_size - num_reqs * 2 # TODO 根据投机config设置
        else:    
            # The reduce_scatter in the TP communication domain after embedding, P goes through this
            graph_pad_size = _get_pad_size(num_input_tokens)
   
        if not (omni_use_dsv3 or (attn_state == AscendAttentionState.DecodeOnly and self.enable_torchair_graph_mode)):
            graph_pad_size = 0

        if graph_pad_size >= 0:
            padding_positions = torch.zeros(graph_pad_size,
                                            dtype=positions.dtype,
                                            device=positions.device)
            positions = torch.cat([positions, padding_positions])

        extra_builder_kwargs = {}
        extra_builder_kwargs['graph_pad_size'] = graph_pad_size

        attn_metadata = self.attn_metadata_builder.build(  # type: ignore
            num_reqs=num_reqs,
            num_actual_tokens=total_num_scheduled_tokens,
            max_query_len=max_num_scheduled_tokens,
            common_prefix_len=None,
            **extra_builder_kwargs,
        )

        # Prepare input_ids
        token_indices = (positions_np +
                         req_indices * self.input_batch.token_ids_cpu.shape[1])
        torch.index_select(self.input_batch.token_ids_cpu_tensor.flatten(),
                           0,
                           torch.from_numpy(token_indices),
                           out=self.input_ids_cpu[:total_num_scheduled_tokens])
        self.query_start_loc_np[0] = 0
        self.query_start_loc_np[1:num_reqs + 1] = cu_num_tokens
        # Copy the tensors to the NPU.
        self.input_ids[:total_num_scheduled_tokens].copy_(
            self.input_ids_cpu[:total_num_scheduled_tokens], non_blocking=True)

        has_spec_tokens = len(
            scheduler_output.scheduled_spec_decode_tokens) > 0
 
        if has_spec_tokens:
            # 当前仅在DecodeOnly时才可能到此逻辑
            # TODO 复用GPU ModelRunner中的_calc_spec_decode_metadata及SpecDecodeMetadata
            # Get the number of draft tokens for each request.
            # Iterate over the dictionary rather than all requests since not all
            # requests have draft tokens.

            sample_indices = torch.arange(total_num_scheduled_tokens, dtype = sample_indices.dtype, device=sample_indices.device)

        return attn_metadata, graph_pad_size, sample_indices, positions, has_spec_tokens

    def _execute_model(
        self,
        scheduler_output,
        attn_metadata,
        graph_pad_size,
        sample_indices,
        positions,
        intermediate_tensors: Optional[IntermediateTensors] = None,
    ) -> Union[ModelRunnerOutput, IntermediateTensors]:
        start_before_f = time.time()
        num_input_tokens = scheduler_output.total_num_scheduled_tokens
        input_ids = self.input_ids[:num_input_tokens]
        model_kwargs = {}
        raw_hidden_states = None
        if attn_metadata.attn_state == AscendAttentionState.DecodeOnly:
            logger.debug(f">>>>> num_input_tokens = {num_input_tokens}, max_batch_size = {self.max_batch_size}, graph_pad_size = {graph_pad_size}")
            if graph_pad_size >= 0:
                padding = torch.zeros(graph_pad_size,
                                      dtype=input_ids.dtype,
                                      device=input_ids.device)
                input_ids = torch.cat([input_ids, padding]) 
        else:
            if graph_pad_size >= 0:
                vocab_size = self.model_config.get_vocab_size()
                padding = torch.randint(1, vocab_size, (graph_pad_size, ),
                                        dtype=input_ids.dtype,
                                        device=input_ids.device)
                input_ids = torch.cat([input_ids, padding])
            model_kwargs["prefill_padding_or_selected_indices"] = sample_indices

        start_fc = time.time()
        start_fc_exit = 0
        # Run forward pass
        with set_forward_context(attn_metadata,
                                 self.vllm_config,
                                 num_tokens=num_input_tokens):
            start_setup_connector = time.time()
            self.maybe_setup_kv_connector(scheduler_output)
            model_kwargs["kv_caches"] = self.kv_caches
            model_kwargs["attn_metadata"] = attn_metadata
            start_f = time.time()

            if model_extra_config.operator_opt_config.use_omni_placement:
                is_prompt = False if attn_metadata.attn_state == AscendAttentionState.DecodeOnly else True
                planner = OmniPlanner(config_file=model_extra_config.operator_opt_config.omni_placement_config_path)
                global _GLOBAL_STEP
                planner.dump(0 if is_prompt else _GLOBAL_STEP)
                if attn_metadata.attn_state == AscendAttentionState.DecodeOnly :
                    _GLOBAL_STEP += 1
                else :
                    _GLOBAL_STEP = 0

            if self.enable_torchair_graph_mode and attn_metadata.attn_state == AscendAttentionState.DecodeOnly:
                start_debug = time.time()
                logger.info("Start running compiled model.")
                if isinstance(self.model, GraphCompileConfiguration):
                    self.model.mark_static_for_graph(input_ids, positions, attn_metadata, self.kv_caches)
                start_os_env = time.time()
                if os.environ.get('PROFILING_FORWARD', "0") == '1':
                    start_time = time.time()
                    import torch_npu
                    prof_save_path = os.environ.get("PROFILING_SAVE_PATH", "./")
                    experimental_config = torch_npu.profiler._ExperimentalConfig(
                        profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
                        aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization)
                    with torch_npu.profiler.profile(
                            activities=[
                                torch_npu.profiler.ProfilerActivity.NPU,
                                torch_npu.profiler.ProfilerActivity.CPU],
                            with_stack=False,
                            record_shapes=False,
                            profile_memory=False,
                            experimental_config=experimental_config,
                            schedule=torch_npu.profiler.schedule(wait=0, warmup=0, active=4, repeat=1, skip_first=1),
                            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(
                                prof_save_path + "_generate")) as prof:
                        for _ in range(6):
                            torch.npu.synchronize()
                            hidden_states = self.compile_model(
                                input_ids=input_ids,
                                positions=positions,
                                intermediate_tensors=intermediate_tensors,
                                inputs_embeds=None,
                                **model_kwargs,
                            )
                            torch.npu.synchronize()
                            prof.step()
                else:
                    start_time = time.time()
                    forward_results = self.compile_model(
                                input_ids=input_ids,
                                positions=positions,
                                intermediate_tensors=intermediate_tensors,
                                inputs_embeds=None,
                                **model_kwargs,
                            )
                    if not omni_use_dsv3:
                        hidden_states = forward_results                       
                    else:
                        raw_hidden_states, hidden_states = forward_results
                    end_model = time.time()
                    cost_model = end_model - start_time
                    cost_os_env = start_time - start_os_env
                    cost_debug = start_debug - start_os_env
                    logger.warning(f" ***** model forward: {cost_model:.6f}, os env: {cost_os_env:.6f}, debug: {cost_debug:.6f}")
            else:
                if self.model is None:
                    raise RuntimeError("self.model must not be None")
                logger.info("Start running eager model.")
                if os.environ.get('PROFILING_FORWARD', "0") == '1' and num_input_tokens > 20000:
                    import torch_npu
                    prof_save_path = os.environ.get("PROFILING_SAVE_PATH", "./")
                    experimental_config = torch_npu.profiler._ExperimentalConfig(
                        profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
                        aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization)
                    with torch_npu.profiler.profile(
                            activities=[
                                torch_npu.profiler.ProfilerActivity.NPU,
                                torch_npu.profiler.ProfilerActivity.CPU],
                            with_stack=False,
                            record_shapes=False,
                            profile_memory=False,
                            experimental_config=experimental_config,
                            schedule=torch_npu.profiler.schedule(wait=0, warmup=0, active=4, repeat=1, skip_first=1),
                            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(
                                prof_save_path + "_generate")) as prof:
                        for _ in range(6):
                            torch.npu.synchronize()
                            if not omni_use_dsv3:
                                    hidden_states = self.model(
                                        input_ids=input_ids,
                                        positions=positions,
                                        intermediate_tensors=intermediate_tensors,
                                        inputs_embeds=None
                                    )
                            else:
                                raw_hidden_states, hidden_states = self.model(
                                        input_ids=input_ids,
                                        positions=positions,
                                        intermediate_tensors=intermediate_tensors,
                                        inputs_embeds=None,
                                        **model_kwargs,
                                    )
                            torch.npu.synchronize()
                            prof.step()
                else:
                    if not omni_use_dsv3:
                            hidden_states = self.model(
                                input_ids=input_ids,
                                positions=positions,
                                intermediate_tensors=intermediate_tensors,
                                inputs_embeds=None
                            )
                    else:
                        raw_hidden_states, hidden_states = self.model(
                                input_ids=input_ids,
                                positions=positions,
                                intermediate_tensors=intermediate_tensors,
                                inputs_embeds=None,
                                **model_kwargs,
                            )
            self.maybe_wait_for_kv_save()
            finished_sending, finished_recving = (
            self.get_finished_kv_transfers(scheduler_output))
            start_fc_exit = time.time()
        start_ret = time.time()
        cost_before_fc = start_fc - start_before_f
        cost_fc = start_ret - start_fc
        cost_setup_connector = start_f - start_setup_connector
        cost_fc_exit = start_ret - start_fc_exit
        logger.info(f" ***** before fc {cost_before_fc:.6f}, fc {cost_fc:.6f}={cost_setup_connector:.6f}+{cost_fc_exit:.6f}")
        return hidden_states, raw_hidden_states, input_ids, finished_sending, finished_recving

    @torch.inference_mode()
    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
        intermediate_tensors: Optional[IntermediateTensors] = None,
    ) -> Union[ModelRunnerOutput, IntermediateTensors]:
        start = time.time()
        # Update KVConnector with the KVConnector metadata forward().
        self._update_states(scheduler_output)
        start_1 = time.time()
        if not scheduler_output.total_num_scheduled_tokens:
            if not has_kv_transfer_group():
                # Return empty ModelRunnerOuptut if there's no work to do.
                return EMPTY_MODEL_RUNNER_OUTPUT
            return self.kv_connector_no_forward(scheduler_output)
        attn_metadata, graph_pad_size, sample_indices, positions, has_spec_tokens = self._prepare_inputs(scheduler_output)
        hidden_states, raw_hidden_states, input_ids, finished_sending, finished_recving = self._execute_model(scheduler_output,
                                           attn_metadata, graph_pad_size, sample_indices, positions, intermediate_tensors)        
        start_2 = time.time()
        logits = self.model.compute_logits(hidden_states[sample_indices], None)
        start_3 = time.time()
        # Apply structured output bitmasks if present
        if scheduler_output.grammar_bitmask is not None:
            logits = self.apply_grammar_bitmask(scheduler_output, logits)
        start_4 = time.time()

        # Sample the next token and get logprobs if needed.
        sampling_metadata = self.input_batch.sampling_metadata 
        if not self.use_spec_decode:
            sampler_output = self.sampler(
                    logits=logits,
                    sampling_metadata=sampling_metadata,
                )
        else:
            sampler_output, mtp_input_tokens, last_accepted_index = self.rejection_sampler(input_ids=input_ids,
                                                                                             logits=logits,
                                                                                             logits_indices=sample_indices,
                                                                                             sampling_metadata=sampling_metadata,
                                                                                             num_decodes=attn_metadata.num_decodes,
                                                                                             num_prefills=attn_metadata.num_prefills
                                                                                            )
        start_5 = time.time()

        discard_sampled_tokens_req_indices = []
        for i, req_id in enumerate(self.input_batch.req_ids):
            req_state = self.requests[req_id]
            seq_len = (req_state.num_computed_tokens +
                       scheduler_output.num_scheduled_tokens[req_id])
            if seq_len < req_state.num_tokens:
                # Ignore the sampled token.
                # Rewind the generator state as if the token was not sampled.
                generator = self.input_batch.generators.get(i)
                if generator is not None:
                    generator.set_offset(generator.get_offset() - 4)
                # Record the index of the request that should not be sampled,
                # so that we could clear the sampled tokens before returning.
                discard_sampled_tokens_req_indices.append(i)
        start_6 = time.time()

        if not self.use_spec_decode:
            # Speculative decoding is not enabled.
            spec_tokens_tensor = None
        elif self.speculative_config.method == 'mtp':
            spec_tokens_tensor = self.run_mtp(
                attn_metadata, scheduler_output, input_ids, raw_hidden_states, mtp_input_tokens, positions, sample_indices, last_accepted_index
            )
        else:
            raise ValueError(f"Speculative method {self.speculative_config.method} is not supported in this version.")

        # NOTE: NPU -> CPU Sync happens here.
        # Move as many CPU operations as possible before this sync point.
        logprobs_tensors = sampler_output.logprobs_tensors
        logprobs_lists = logprobs_tensors.tolists() \
            if logprobs_tensors is not None else None

        # Get the valid generated tokens.
        sampled_token_ids = sampler_output.sampled_token_ids
        max_gen_len = sampled_token_ids.shape[-1]
        if max_gen_len == 1:
            # No spec decode tokens.
            valid_sampled_token_ids = sampled_token_ids.tolist()
        else:
            # Includes spec decode tokens.
            # [[bonus,b_forward], [forward], [bonus,b_forward], [bonus,b_forward],..]
            valid_sampled_token_ids = self.rejection_sampler.parse_output(
                sampled_token_ids,
                self.input_batch.vocab_size,
            )
 
        spec_token_ids = None if spec_tokens_tensor is None else spec_tokens_tensor.tolist()

        # Mask out the sampled tokens that should not be sampled.
        for i in discard_sampled_tokens_req_indices:
            valid_sampled_token_ids[i].clear()
        # Clear KVConnector state after all KVs are generated.
        if has_kv_transfer_group():
            get_kv_transfer_group().clear_connector_metadata()
        model_runner_output = ModelRunnerOutput(
            req_ids=self.input_batch.req_ids,
            req_id_to_index=self.input_batch.req_id_to_index,
            sampled_token_ids=valid_sampled_token_ids,
            spec_token_ids=spec_token_ids,
            logprobs=logprobs_lists,
            prompt_logprobs_dict={},
            finished_sending=finished_sending,
            finished_recving=finished_recving,
        )
        cost_upd_states = start_1 - start
        cost_proc_reqs = start_2 - start_1
        cost_logits = start_3 - start_2
        cost_bitmask = start_4 - start_3
        cost_sampler = start_5 - start_4
        cost_disc = start_6 - start_5
        cost_output = time.time() - start_6
        cost = cost_upd_states + cost_proc_reqs + cost_logits + cost_bitmask + cost_sampler + cost_disc + cost_output
        logger.info(f" ***** execute model cost:{cost:.6f}={cost_upd_states:.6f}+{cost_proc_reqs:.6f}+{cost_logits:.6f}+{cost_bitmask:.6f}+{cost_sampler:.6f}+{cost_disc:.6f}+{cost_output:.6f}")
        return model_runner_output

    @torch.inference_mode()
    def run_mtp(self, attn_metadata, scheduler_output, input_ids, raw_hidden_states, mtp_input_tokens, positions, sample_indices, last_accepted_index):
        if self.enable_torchair_graph_mode and attn_metadata.attn_state == AscendAttentionState.DecodeOnly:
            with set_forward_context(attn_metadata,
                                    self.vllm_config,
                                    num_tokens=scheduler_output.total_num_scheduled_tokens):
                torch._dynamo.mark_static(input_ids)
                torch._dynamo.mark_static(raw_hidden_states)
                mtp_hidden_states = self.compile_drafter(
                    input_ids=mtp_input_tokens.to(torch.long),
                    positions=positions,
                    kv_caches=self.kv_caches[-1:],
                    attn_metadata=attn_metadata,
                    previous_hidden_states=raw_hidden_states,
                    intermediate_tensors=None,
                    inputs_embeds=None
                )
        else:
            # prefill or nograph
            with set_forward_context(attn_metadata,
                                    self.vllm_config,
                                    num_tokens=scheduler_output.total_num_scheduled_tokens):
                mtp_hidden_states = self.drafter(
                    input_ids=mtp_input_tokens.to(torch.long),
                    positions=positions,
                    kv_caches=self.kv_caches[-1:],
                    attn_metadata=attn_metadata,
                    previous_hidden_states=raw_hidden_states,
                    prefill_padding_or_selected_indices=sample_indices,
                    intermediate_tensors=None,
                    inputs_embeds=None
                )
    
        mtp_logits =self.drafter.compute_logits(mtp_hidden_states[last_accepted_index], None)
        return mtp_logits.argmax(dim=-1, keepdim=True)
        

    @torch.inference_mode()
    def _dummy_run(self, num_tokens: int) -> torch.Tensor:
        if self.is_multimodal_model:
            input_ids = None
            inputs_embeds = self.inputs_embeds[:num_tokens]
        else:
            input_ids = self.input_ids[:num_tokens]
            inputs_embeds = None

        if get_pp_group().is_first_rank:
            intermediate_tensors = None
        else:
            if self.intermediate_tensors is None:
                self.intermediate_tensors = (
                    self.model.make_empty_intermediate_tensors(
                        batch_size=num_tokens,
                        dtype=self.dtype,
                        device=self.device))
            intermediate_tensors = IntermediateTensors({
                k: v[:num_tokens]
                for k, v in self.intermediate_tensors.items()
            })
        positions = self.mrope_positions[:, :num_tokens] if self.uses_mrope else self.positions[:num_tokens]

        attn_metadata = None
        raw_hidden_states = None

        if not self.kv_caches:
            # profile run
            with set_forward_context(None, self.vllm_config, num_tokens=num_tokens):            
                forward_results = self.model(
                                    input_ids=input_ids,
                                    positions=positions,
                                    intermediate_tensors=intermediate_tensors,
                                    inputs_embeds=inputs_embeds,
                                )
                if not omni_use_dsv3:
                    hidden_states = forward_results
                else:
                    raw_hidden_states, hidden_states = forward_results
                if self.use_spec_decode and self.speculative_config.method in ('mtp'):
                    self.drafter(
                        input_ids=input_ids,
                        positions=positions,
                        kv_caches=None,
                        attn_metadata=None,
                        previous_hidden_states=raw_hidden_states,
                        intermediate_tensors=None,
                        inputs_embeds=None
                    )
        else:
            fake_input = torch.zeros(self.max_batch_size,
                                     dtype=input_ids.dtype,
                                     device=input_ids.device)
            input_ids = fake_input
            positions = fake_input
            self.attn_mask = None
            self.attn_state = AscendAttentionState.DecodeOnly

            if not isinstance(self.attn_metadata_builder, DummyAttentionMetadataBuilder):
                raise ValueError("attn_metadata_builder does not implement DummyAttentionMetadataBuilder")
            attn_metadata = self.attn_metadata_builder.build_dummy(num_tokens, self.max_batch_size)
            with set_forward_context(attn_metadata, self.vllm_config, num_tokens=num_tokens):
                is_pd_seperate_d = self.vllm_config.kv_transfer_config is not None and self.vllm_config.kv_transfer_config.kv_role == "kv_consumer"
                if self.enable_torchair_graph_mode and is_pd_seperate_d:
                    logger.debug("Start running dummy compiled model.")
                    model_kwargs = {}
                    model_kwargs["kv_caches"] = self.kv_caches
                    model_kwargs["attn_metadata"] = attn_metadata
                    if isinstance(self.model, GraphCompileConfiguration):
                        self.model.mark_static_for_graph(input_ids, positions, attn_metadata, self.kv_caches)
                    forward_results = self.compile_model(
                        input_ids=input_ids,
                        positions=positions,
                        intermediate_tensors=intermediate_tensors,
                        inputs_embeds=None,
                        **model_kwargs,
                    )
                    if not omni_use_dsv3:
                        hidden_states = forward_results                       
                    else:
                        raw_hidden_states, hidden_states = forward_results
                    if self.use_spec_decode and self.speculative_config.method in ('mtp'):
                        if not self.dummy_drafter_mark_static:
                            torch._dynamo.mark_static(input_ids)
                            torch._dynamo.mark_static(raw_hidden_states)
                            self.dummy_drafter_mark_static = True
                        self.compile_drafter(
                            input_ids=input_ids,
                            positions=positions,
                            kv_caches=self.kv_caches[-1:] if self.kv_caches else None,
                            attn_metadata=attn_metadata,
                            previous_hidden_states=raw_hidden_states,
                            intermediate_tensors=None,
                            inputs_embeds=None
                        )
                else:
                    logger.debug("Start running dummy eager model.")
                    if not omni_use_dsv3:
                        hidden_states = self.model(input_ids=input_ids,
                                            positions=positions,
                                            intermediate_tensors=intermediate_tensors,
                                            inputs_embeds=inputs_embeds)
                    else:
                        raw_hidden_states, hidden_states = self.model(input_ids=input_ids,
                                            positions=positions,
                                            intermediate_tensors=intermediate_tensors,
                                            inputs_embeds=inputs_embeds,
                                            kv_caches=self.kv_caches,
                                            attn_metadata=attn_metadata)
                    if self.use_spec_decode and self.speculative_config.method in ('mtp'):
                        self.drafter(
                            input_ids=input_ids,
                            positions=positions,
                            kv_caches=self.kv_caches[-1:] if self.kv_caches else None,
                            attn_metadata=attn_metadata,
                            previous_hidden_states=raw_hidden_states,
                            intermediate_tensors=None,
                            inputs_embeds=None
                        )
        return hidden_states


    def profile_run(self) -> None:
        hidden_states = self._dummy_run(self.max_num_tokens)

        NPUPlatform.synchronize()
        del hidden_states
        self.encoder_cache.clear()
        gc.collect()

    def load_model(self) -> None:
        logger.info("Starting to load model %s...", self.model_config.model)

        with DeviceMemoryProfiler() as m:  # noqa: SIM117
            self.model = get_model(vllm_config=self.vllm_config)
            if self.lora_config:
                raise ValueError("LoRA model is not supported on NPU now.")
            if hasattr(self, "drafter"):            
                logger.info("Loading mtp model...")
                original_arch = self.model_config.hf_config.architectures # ['DeepseekV3ForCausalLM']
                original_type = self.model_config.hf_config.model_type    # 'deepseek_v3'
                                                      
                self.model_config.hf_config.architectures = ["DeepSeekMTPModel"]
                self.model_config.hf_config.model_type = "deepseek_mtp"
                self.drafter = get_model(vllm_config=self.vllm_config)
                self.drafter.embed_tokens = self.model.model.embed_tokens
                self.drafter.shared_head['head'] = self.model.lm_head
                self.model_config.hf_config.architectures = original_arch
                self.model_config.hf_config.model_type = original_type
                # zxp TODO: check if fusion_spec.py from line 90 needed?
        logger.info("Loading model weights took %.4f GB",
                    m.consumed_memory / float(2**30))

        # adapter torch compile with npu_backend
        if self.enable_torchair_graph_mode:
            import torchair  # type: ignore
            from torchair import patch_for_hcom  # type: ignore

            patch_for_hcom()
            config = torchair.CompilerConfig()
            # Set the export image structure file format
            # config.debug.graph_dump.type = "py"
            config.experimental_config.frozen_parameter = True
            config.experimental_config.tiling_schedule_optimize = True
            torch.npu.set_compile_mode(jit_compile=False)
            if not self.use_cached_npu_graph:
                logger.debug(f"[not use cache npu graph], VLLM_TEST_DYNAMO_FULLGRAPH_CAPTURE = {envs.VLLM_TEST_DYNAMO_FULLGRAPH_CAPTURE}")
                npu_backend = torchair.get_npu_backend(compiler_config=config)
                self.compile_model = torch.compile(
                    self.model,
                    dynamic=True,
                    fullgraph=envs.VLLM_TEST_DYNAMO_FULLGRAPH_CAPTURE,
                    backend=npu_backend)
                if hasattr(self, "drafter"):
                    self.compile_drafter = torch.compile(
                        self.drafter,
                        dynamic=True,
                        fullgraph=envs.VLLM_TEST_DYNAMO_FULLGRAPH_CAPTURE,
                        backend=npu_backend)
            else:
                logger.debug("[use cache npu graph]")
                self.compile_model = WrapModel(self.model, self.decode_gear_list)
                if hasattr(self, "drafter"):
                    self.compile_drafter = WrapDrafter(self.drafter, self.decode_gear_list)

    def initialize_kv_cache(self, kv_cache_config: KVCacheConfig) -> None:
        """
        Initialize KV cache based on `kv_cache_config`.
        Args:
            kv_cache_config: Configuration for the KV cache, including the KV
            cache size of each layer
        """
        import torch_npu
        kv_caches: Dict[str, torch.Tensor] = {}
        self.kv_cache_config = kv_cache_config
        self.input_batch = InputBatch(
            max_num_reqs=self.max_num_reqs,
            max_model_len=self.model_config.max_model_len,
            max_num_batched_tokens=self.max_num_tokens,
            device=self.device,
            pin_memory=True,
            vocab_size=self.model_config.get_vocab_size(),
            kv_cache_config=kv_cache_config,
        )
        self.input_batch.token_ids_cpu_tensor = torch.zeros(
            (self.max_num_reqs, self.model_config.max_model_len),
            device="cpu",
            dtype=torch.int64,
            pin_memory=False,
        )
        self.input_batch.token_ids_cpu = self.input_batch.token_ids_cpu_tensor.numpy()

        for kv_cache_group in kv_cache_config.kv_cache_groups:
            kv_cache_spec = kv_cache_group.kv_cache_spec
            for layer_name in kv_cache_group.layer_names:
                tensor_config = kv_cache_config.tensors[layer_name]
                if tensor_config.size % kv_cache_spec.page_size_bytes != 0:
                    raise RuntimeError("tensor_config.size must be divisible by kv_cache_spec.page_size_bytes")
                num_blocks = tensor_config.size // kv_cache_spec.page_size_bytes
                # `num_blocks` is the number of blocks the model runner can use.
                # `kv_cache_config.num_blocks` is the number of blocks that
                # KVCacheManager may allocate.
                # Since different GPUs may have different number of layers and
                # different memory capacities, `num_blocks` can be different on
                # different GPUs, and `kv_cache_config.num_blocks` is set to
                # the min of all `num_blocks`. Verify it here.
                if num_blocks < kv_cache_config.num_blocks:
                    raise RuntimeError("num_blocks must be greater than or equal to kv_cache_config.num_blocks")
                if isinstance(kv_cache_spec, FullAttentionSpec):
                    kv_cache_shape = self.attn_backend.get_kv_cache_shape(
                        num_blocks, kv_cache_spec.block_size,
                        kv_cache_spec.num_kv_heads, kv_cache_spec.head_size)
                    dtype = kv_cache_spec.dtype
                    kv_caches[layer_name] = self.attn_backend.init_kv_cache_each_layer(kv_cache_shape, self.dtype, self.device, self.model_config, self.enable_torchair_graph_mode)
                else:
                    raise ValueError("Unknown KV cache spec type.")

        bind_kv_cache(
            kv_caches,
            self.vllm_config.compilation_config.static_forward_context,
            self.kv_caches)

        if has_kv_transfer_group():
            get_kv_transfer_group().register_kv_caches(kv_caches)

    def capture_model(self) -> None:
        start_time = time.perf_counter()
        start_free_npu_memory = torch.npu.mem_get_info()[0]
        if self.enable_torchair_graph_mode:
            decode_gear_list = self.decode_gear_list
            graph_num = len(decode_gear_list)
            logger.info(
                "Capturing torchair graph, this usually takes %.1f~%.1f mins.",
                0.5 * graph_num, 1.5 * graph_num)
            # Trigger torchair graph capture for specific shapes.
            # Capture the large shapes first so that the smaller shapes
            # can reuse the memory pool allocated for the large shapes.
            for idx, num_tokens in enumerate(
                    reversed(decode_gear_list)):
                self._dummy_run(num_tokens)
                logger.info("Batchsize %d is compiled successfully: %d/%d.",
                            num_tokens, idx + 1, graph_num)
        else:
            logger.warning(
                "Skipping NPU graph capture. Please add "
                "-O %s to use NPU graphs.", CompilationLevel.PIECEWISE)
            return

    def _get_closest_gear(self, max_num_token):
        for gear in self.decode_gear_list:
            if gear >= max_num_token:
                return gear
        raise ValueError(f"decode input batch size {max_num_token} exceeds maximum gear {max(self.decode_gear_list)}.")


class WrapModel(nn.Module):
    def __init__(self, model, decode_gear_list) -> None:
        super().__init__()
        self.model = model
        self.decode_gear_list = decode_gear_list
        from torchair.configs.compiler_config import CompilerConfig
        import torchair.ge_concrete_graph.ge_converter.experimental.patch_for_hcom_allreduce
        torch._dynamo.reset()
        config = CompilerConfig()
        config.experimental_config.keep_inference_input_mutations = True
        config.experimental_config.tiling_schedule_optimize = True
        torch.npu.set_compile_mode(jit_compile=False)
        self.cached_decode_dict = {}
        for i, gear in enumerate(self.decode_gear_list):
            self.cached_decode_dict[gear] = torchair.inference.cache_compile(getattr(self, f"decode_batch_{i}"), config=config, dynamic=True, ge_cache=False, fullgraph=envs.VLLM_TEST_DYNAMO_FULLGRAPH_CAPTURE,)

    def decode_batch_0(self, *args, **kwargs):
        return self._forward(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self._forward(*args, **kwargs)

    def forward(
            self,
            input_ids: torch.Tensor,
            positions: torch.Tensor,
            intermediate_tensors: Optional[IntermediateTensors] = None,
            **kwargs,
    ) -> torch.Tensor:
        gear = input_ids.shape[0]
        return self.cached_decode_dict[gear](input_ids, positions, intermediate_tensors=intermediate_tensors, **kwargs)

    def _forward(
            self,
            input_ids: torch.Tensor,
            positions: torch.Tensor,
            intermediate_tensors: Optional[IntermediateTensors] = None,
            **kwargs,
    ) -> torch.Tensor:
        # adapt Inconsistent attribute names in model cls
        hidden_states = self.model.forward(input_ids, positions, intermediate_tensors=intermediate_tensors, **kwargs)
        return hidden_states

class WrapDrafter(nn.Module):
    def __init__(self, model, decode_gear_list) -> None:
        super().__init__()
        self.model = model
        self.decode_gear_list = decode_gear_list
        from torchair.configs.compiler_config import CompilerConfig
        import torchair.ge_concrete_graph.ge_converter.experimental.patch_for_hcom_allreduce
        torch._dynamo.reset()
        config = CompilerConfig()
        config.experimental_config.keep_inference_input_mutations = True
        config.experimental_config.tiling_schedule_optimize = True
        torch.npu.set_compile_mode(jit_compile=False)
        self.cached_decode_dict = {}
        for i, gear in enumerate(self.decode_gear_list):
            self.cached_decode_dict[gear] = torchair.inference.cache_compile(getattr(self, f"decode_batch_{i}"), config=config, dynamic=True, ge_cache=False, fullgraph=envs.VLLM_TEST_DYNAMO_FULLGRAPH_CAPTURE,)

    def decode_batch_0(self, *args, **kwargs):
        return self._forward(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self._forward(*args, **kwargs)

    def forward(
            self,
            input_ids: torch.Tensor,
            positions: torch.Tensor,
            kv_caches: List[torch.Tensor],
            attn_metadata,
            previous_hidden_states: torch.Tensor,
            intermediate_tensors: Optional[IntermediateTensors],
            **kwargs,
    ) -> torch.Tensor:
        gear = input_ids.shape[0]
        return self.cached_decode_dict[gear](input_ids,
                                             positions,
                                             kv_caches,
                                             attn_metadata,
                                             previous_hidden_states,
                                             intermediate_tensors=intermediate_tensors,
                                             **kwargs)

    def _forward(
            self,
            input_ids: torch.Tensor,
            positions: torch.Tensor,
            kv_caches: List[torch.Tensor],
            attn_metadata, 
            previous_hidden_states: torch.Tensor,
            intermediate_tensors: Optional[IntermediateTensors],
            **kwargs,
    ) -> torch.Tensor:
        # adapt Inconsistent attribute names in model cls
        hidden_states = self.model.forward(input_ids,
                                           positions,
                                           kv_caches,
                                           attn_metadata,
                                           previous_hidden_states,
                                           intermediate_tensors=intermediate_tensors,
                                           **kwargs)
        return hidden_states
