# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

from typing import List, Tuple, Union, Optional, Dict, Type
from typing_extensions import override
from dataclasses import dataclass
import itertools
import ast

import torch
import torch.distributed
import torch_npu
import torchair as tng

from vllm.logger import init_logger
from vllm.utils import is_pin_memory_available, make_tensor_with_pad, async_tensor_h2d
from vllm.config import CacheConfig, DeviceConfig, ModelConfig, ParallelConfig
from vllm.core.interfaces import AllocStatus
from vllm.core.block_manager import SelfAttnBlockSpaceManager
from vllm.core.block.cpu_gpu_block_allocator import CpuGpuBlockAllocator
from vllm.core.block.block_table import BlockTable
from vllm.attention.backends import utils as attn_utils
from vllm.sequence import SequenceStatus
from vllm.utils import Device
from vllm.distributed import tensor_model_parallel_all_gather, get_world_group
from vllm.platforms import current_platform

from vllm_npu.common.attention.backends.fx_base_attn import (
    FXAttentionMetadata,
    FXBaseAttentionBackend,
    FxMetadataBuilder,
)
from vllm_npu import ENV
from vllm_npu.fx.worker.fx_cache_engine import AscendCacheEngine
from vllm_npu.fx.worker.fx_model_runner import ModelInputForNPUBuilder
from vllm_npu.fx.model_executor.models.deepseek_v3 import AscendDeepseekAttention_MLA
from cann_ops import reshape_and_cache, splitV


logger = init_logger('vllm.worker.omni')


# These are initialized when the module is imported.
SINK = ENV.omni_attention_sink
RECENT = ENV.omni_attention_recent
BETA = 2 / (1 + ENV.omni_attention_estimate_avg_inlen/(SINK+RECENT))


def _log_at_rank_0(msg):
    if torch.distributed.get_rank() != 0:
        return
    logger.warning(msg)


def get_block_space_manager_class(version: str):
    if ENV.omni_attention_enabled:
        return OmniBlockSpaceManager

    raise RuntimeError(f"Error. OMNI attention is not enabled, shouldn't call this function.")


class OmniBlockTable(BlockTable):
    def __init__(
        self,
        block_size,
        block_allocator,
        omni_required_blocks,
        _blocks=None,
        max_block_sliding_window=None,
    ):
        super().__init__(
            block_size=block_size,
            block_allocator=block_allocator,
            _blocks=_blocks,
            max_block_sliding_window=None,
        )
        if self._num_full_slots > SINK + RECENT:
            raise RuntimeError("Error. For OMNI block tables, slots used should never"
                               f" exceed {SINK+RECENT=}, but got {self._num_full_slots}.")
        self.omni_required_blocks = omni_required_blocks

    @override
    def allocate(self, token_ids: List[int], device: Device = Device.GPU) -> None:
        """Allocate a constant number of blocks for the given tokens. No matter the length of the input, this function
        uses `(SINK + RECENT) // BLOCK_SIZE` blocks. If the input is shorter than this, some blocks would be empty.
        If the input cannot fit into this, only the first `SINK` and the last `RECENT` tokens are kept.

        This function is supposed to be called during prefilling phase.

        Args:
            token_ids (List[int]): The token ids of the input prompt.
            device (Device, optional): On which device the blocks should be allocated. Defaults to Device.GPU.
        """
        if self._is_allocated:
            raise RuntimeError("Already allocated.")

        if not token_ids:
            raise ValueError("token_ids must not be empty.")

        # make sure token_ids contain no more than s+r tokens
        if len(token_ids) > SINK + RECENT:
            token_ids = token_ids[:SINK] + token_ids[-RECENT:]

        # now blocks length is less than or equal to self.omni_required_blocks
        blocks = self._allocate_blocks_for_token_ids(prev_block=None,
                                                     token_ids=token_ids,
                                                     device=device)

        # make sure blocks contain exactly omni_required_blocks blocks
        cur_num_blocks = len(blocks)
        if cur_num_blocks < self.omni_required_blocks:
            prev_block = blocks[-1]
            for _ in range(self.omni_required_blocks - cur_num_blocks):
                # append empty blocks
                prev_block = self._allocator.allocate_mutable_block(
                    prev_block=prev_block,
                    device=device,
                )
                blocks.append(prev_block)

        self.update(blocks)
        self._num_full_slots = len(token_ids)


class OmniBlockSpaceManager(SelfAttnBlockSpaceManager):
    def __init__(
        self,
        block_size: int,
        num_gpu_blocks: int,
        num_cpu_blocks: int,
        watermark: float = 0.01,
        sliding_window: Optional[int] = None,
        enable_caching: bool = False,
    ):
        if sliding_window is not None:
            _log_at_rank_0("OMNI attention implements its own sliding window already."
                           " Will ignore extra sliding window arguments.")
        if enable_caching:
            _log_at_rank_0("Prefix caching is not supported with OMNI attention yet."
                           " Will use **naive** block allocator instead.")

        # how many blocks are needed for an omni layer
        if SINK % block_size != 0 or RECENT % block_size != 0:
            raise ValueError("Error. For OMNI attention, sink and recent sizes must be divisible by block size."
                             f"However, {SINK=}, {RECENT=}, and {block_size=}.")
        self.omni_required_blocks = (SINK + RECENT) // block_size

        # adapt number of blocks needed for full and omni layers, and use adapted values to create block allocators
        full_num_gpu_blocks, omni_num_gpu_blocks = OmniCacheEngine.adapt_num_blocks(num_gpu_blocks)
        full_num_cpu_blocks, omni_num_cpu_blocks = OmniCacheEngine.adapt_num_blocks(num_cpu_blocks)

        super().__init__(
            block_size=block_size,
            num_gpu_blocks=full_num_gpu_blocks,
            num_cpu_blocks=full_num_cpu_blocks,
            watermark=watermark,
            sliding_window=None,
            enable_caching=False,
        )

        # set omni total gpu and cpu blocks
        self.omni_total_gblck = omni_num_gpu_blocks
        self.omni_total_cblck = omni_num_cpu_blocks

        # another block allocator for omni layers
        self.omni_block_allocator = CpuGpuBlockAllocator.create(
            allocator_type='naive',
            num_gpu_blocks=self.omni_total_gblck,
            num_cpu_blocks=self.omni_total_cblck,
            block_size=block_size,
        )

        # another block table for omni layers
        self.omni_block_tables: Dict[int, OmniBlockTable] = {}

        # some logging
        _log_at_rank_0("For OMNI attention, full layer block allocator is created"
                       f" with {self.block_allocator.get_num_total_blocks(Device.GPU)} gpu blocks,"
                       "\nand omni layer block allocator with"
                       f" {self.omni_block_allocator.get_num_total_blocks(Device.GPU)}.")

    @override
    def can_allocate(self, seq_group, num_lookahead_slots=0) -> AllocStatus:
        if num_lookahead_slots != 0:
            raise RuntimeError("Error. Currently OMNI attention is not supported with speculative"
                               "decoding. Please set `--num-lookahead-slots=0`.")
        full_status = super().can_allocate(seq_group, num_lookahead_slots=0)

        if full_status != AllocStatus.OK:
            return full_status

        # full layer says ok, so whatever omni layer says decides
        num_free_gpu_blocks = self.omni_block_allocator.get_num_free_blocks(Device.GPU)
        if num_free_gpu_blocks - self.omni_required_blocks > 0:
            return AllocStatus.OK
        else:
            return AllocStatus.LATER

    @override
    def _allocate_sequence(self, seq) -> Tuple[BlockTable, OmniBlockTable]:
        full_table = super()._allocate_sequence(seq)
        omni_table = OmniBlockTable(
            block_size=self.block_size,
            block_allocator=self.omni_block_allocator,
            omni_required_blocks=self.omni_required_blocks,
        )

        token_ids = seq.get_token_ids()
        if token_ids:
            omni_table.allocate(token_ids)

        return full_table, omni_table

    @override
    def allocate(self, seq_group) -> None:
        # Allocate self-attention block tables for decoder sequences
        waiting_seqs = seq_group.get_seqs(status=SequenceStatus.WAITING)
        if set(seq.seq_id for seq in waiting_seqs) & self.block_tables.keys():
            raise RuntimeError("block table already exists")

        # NOTE: Here we assume that all sequences in the group have the same
        # prompt.
        # NOTE: Change 1: add `omni_block_table`. --- MRZ
        seq = waiting_seqs[0]
        block_table, omni_block_table = self._allocate_sequence(seq)
        self.block_tables[seq.seq_id] = block_table
        self.omni_block_tables[seq.seq_id] = omni_block_table

        # Track seq
        self._computed_blocks_tracker.add_seq(seq.seq_id)
        self._last_access_blocks_tracker.add_seq(seq.seq_id)

        # Assign the block table for each sequence.
        for seq in waiting_seqs[1:]:
            # NOTE: Change 2: fork `omni_block_table`. --- MRZ
            self.block_tables[seq.seq_id] = block_table.fork()
            self.omni_block_tables[seq.seq_id] = omni_block_table.fork()

            # Track seq
            self._computed_blocks_tracker.add_seq(seq.seq_id)
            self._last_access_blocks_tracker.add_seq(seq.seq_id)

        # Change 3: remove cross-attention since decoder-only models are considered here. --- MRZ

    @override
    def free(self, seq) -> None:
        seq_id = seq.seq_id

        if seq_id not in self.block_tables:
            # Already freed or haven't been scheduled yet.
            return

        # Update seq block ids with the latest access time
        self._last_access_blocks_tracker.update_seq_blocks_last_access(
            seq_id, self.block_tables[seq.seq_id].physical_block_ids)

        # Untrack seq
        self._last_access_blocks_tracker.remove_seq(seq_id)
        self._computed_blocks_tracker.remove_seq(seq_id)

        # Free table/blocks
        self.block_tables[seq_id].free()
        del self.block_tables[seq_id]

        # The only chage: free omni table/blocks. --- MRZ
        self.omni_block_tables[seq_id].free()
        del self.omni_block_tables[seq_id]

    @override
    def get_block_table(self, seq) -> Tuple[List[int], List[int]]:
        block_ids = self.block_tables[seq.seq_id].physical_block_ids
        omni_block_ids = self.omni_block_tables[seq.seq_id].physical_block_ids
        return block_ids, omni_block_ids


@dataclass
class OmniFXAttentionMetadata(FXAttentionMetadata):
    """This metadata class inherits FXAttentionMetadata and adds a few attributes:
        1. omni_block_tables: The block tables used in omni layers.
        2. omni_slot_mapping: The slot mapping used in omni layers.
        3. omni_seq_lens_tensor: The sequence lengths to participate in paged attention in omni layers.
    """

    omni_block_tables: Optional[torch.Tensor] = None
    omni_slot_mapping: Optional[torch.Tensor] = None
    omni_seq_lens_tensor: Optional[torch.Tensor] = None
    omni_seq_lens: Optional[List[int]] = None


class OmniFxMetadataBuilder(FxMetadataBuilder):
    def __init__(self, input_builder: ModelInputForNPUBuilder):
        super().__init__(input_builder)
        self.block_tables: List[List[int]] = []
        self.slot_mapping: List[int] = []
        self.omni_block_tables: List[List[int]] = []
        self.omni_slot_mapping: List[int] = []

    @override
    def _add_seq_group(self, inter_data: ModelInputForNPUBuilder.InterDataForSeqGroup, **kwargs) -> None:
        """Add a sequence group to the metadata. Handle block tables and slot mapping in both
        full and omni layers.

        Args:
            inter_data (ModelInputForNPUBuilder.InterDataForSeqGroup):
                Contains sequence group info like block_tables, seq_lens, etc. Specifically, it has an
                attribute `prompt_lens` which is added by :func:`~ModelInputForNPUBuilder._compute_prompt_length`.
        """
        # For omni, each value of inter_data.block_tables contains a tuple of two integer lists.
        block_tables: Dict[int, Tuple[List[int], List[int]]] = inter_data.block_tables
        is_prompt = inter_data.is_prompt

        for seq_id, token_len, seq_len, curr_seq_len, query_len, context_len, prompt_len in zip(
                inter_data.seq_ids,                         # seq_id
                [len(t) for t in inter_data.input_tokens],  # token_len
                inter_data.orig_seq_lens,                   # seq_len
                inter_data.seq_lens,                        # curr_seq_len
                inter_data.query_lens,                      # query_len
                inter_data.context_lens,                    # context_len
                inter_data.prompt_lens,                     # prompt_len
            ):
            self.context_lens.append(context_len)

            if is_prompt:
                self.num_prefills += 1
                self.num_prefill_tokens += token_len
                self.prefill_seq_lens.append(seq_len)
            else:
                self.num_decode_tokens += query_len
                self.curr_seq_lens.append(curr_seq_len)

            # NOTE(MRZ 0504): This can be True in both profile run and PTA compile run
            is_profile_run = attn_utils.is_block_tables_empty(block_tables)
            start_idx = 0  # wo sliding window, start_idx is surely 0

            # Compute block table and omni block table.
            # NOTE: chunked prefill and prefix caching are discarded.
            block_table, omni_block_table = ([], []) if is_profile_run else block_tables[seq_id]
            if block_tables is not None and not is_prompt:
                self.block_tables.append(block_table)
                self.omni_block_tables.append(omni_block_table)
            else:
                self.block_tables.append([])
                self.omni_block_tables.append([])

            # Compute slot mapping for full layers.
            if is_profile_run and not is_prompt:
                # NOTE(MRZ 0504): This `if` is for PTA compile run where `is_profile_run` is also True
                self.slot_mapping.extend([attn_utils.PAD_SLOT_ID] * query_len)
            else:
                attn_utils.compute_slot_mapping(
                    is_profile_run,
                    self.slot_mapping,
                    seq_id,
                    seq_len,
                    context_len,
                    start_idx,
                    self.block_size,
                    {seq_id: block_table},
                )

            # Compute slot mapping for omni layers.
            if is_profile_run and not is_prompt:
                # NOTE(MRZ 0504): This `if` is for PTA compile run where `is_profile_run` is also True
                self.omni_slot_mapping.extend([attn_utils.PAD_SLOT_ID] * query_len)
            elif is_prompt:
                # NOTE(MRZ 0508): profiling or prefilling
                omni_slots = []
                attn_utils.compute_slot_mapping(
                    is_profile_run,
                    omni_slots,
                    seq_id,
                    min(seq_len, SINK+RECENT),
                    0,
                    start_idx,
                    self.block_size,
                    {seq_id: omni_block_table},
                )
                if seq_len > SINK + RECENT:
                    omni_slots = (
                        omni_slots[:SINK]
                        + [attn_utils.PAD_SLOT_ID] * (seq_len-SINK-RECENT)
                        + omni_slots[SINK:]
                    )
                self.omni_slot_mapping.extend(omni_slots)
            else:
                # NOTE(MRZ 0508): must be decoding
                if seq_len <= SINK + RECENT:
                    omni_context_len, omni_seq_len = context_len, seq_len
                else:
                    omni_context_len = (seq_len - prompt_len - 1) % RECENT + SINK
                    omni_seq_len = omni_context_len + 1
                attn_utils.compute_slot_mapping(
                    is_profile_run,
                    self.omni_slot_mapping,
                    seq_id,
                    omni_seq_len,
                    omni_context_len,
                    start_idx,
                    self.block_size,
                    {seq_id: omni_block_table},
                )

    @override
    def build(
        self,
        seq_lens: List[int],
        query_lens: List[int],
        cuda_graph_pad_size: int,
        batch_size: int,
        use_2d_input,
        any_prefix_cache_hit=False,
    ):
        """Build attention metadata with on-device tensors.

        Args:
            seq_lens: The maybe padded sequence lengths of the input sequences.
            query_lens: The query lengths of the input sequences.
            cuda_graph_pad_size: The padding size for cuda graph.
                                 -1 if cuda graph is not used.
            batch_size: The maybe padded batch size.
        """
        prefix_cache_hit = any(
            [
                inter_data.prefix_cache_hit
                for inter_data in self.input_builder.inter_data_list
            ]
        )
        for inter_data in self.input_builder.inter_data_list:
            self._add_seq_group(inter_data)

        # NOTE(MRZ 0508) slot_mapping and omni_slot_mapping should have same length
        if len(self.slot_mapping) != len(self.omni_slot_mapping):
            raise RuntimeError("Length of `slot_mapping` and `omni_slot_mapping` do not match."
                               f" Got {self.slot_mapping=} and {self.omni_slot_mapping=}.")

        device = self.runner.device
        if cuda_graph_pad_size > 0:
            slot_pads: List[List[int]] = list(
                itertools.repeat([attn_utils.PAD_SLOT_ID] * (query_lens[0] if ENV.dp_size > 1 else 1),
                                 cuda_graph_pad_size)
            )
            slot_pads: List[int] = list(itertools.chain(*slot_pads))
            self.slot_mapping.extend(slot_pads)
            self.block_tables.extend([[]] * cuda_graph_pad_size)
            self.omni_slot_mapping.extend(slot_pads)
            self.omni_block_tables.extend([[]] * cuda_graph_pad_size)  # pad dim 0 for block tables
        input_block_tables = self.runner.graph_block_tables[:batch_size]
        max_blocks = input_block_tables.shape[1]
        for i, block_table in enumerate(self.block_tables):
            if block_table:
                num_blocks = len(block_table)
                if num_blocks <= max_blocks:
                    input_block_tables[i, :num_blocks] = block_table
                else:
                    # It may be possible to have more blocks allocated due
                    # to lookahead slots of multi-step, however, they are
                    # not used anyway, so can be safely ignored.
                    input_block_tables[i, :max_blocks] = block_table[:max_blocks]
        block_tables = torch.from_numpy(input_block_tables).to(
            device, non_blocking=True)

        max_prefill_seq_len = max(self.prefill_seq_lens, default=0)
        max_decode_seq_len = max(self.curr_seq_lens, default=0)
        num_decode_tokens = self.num_decode_tokens
        # Change 1: omit `relay_attn_prefix_block_tables_len`. --- MRZ

        # Only use in decode, prefill will return 0 in metadata.
        decode_query_lens = query_lens[self.num_prefills:]
        if len(decode_query_lens) > 0:
            max_decode_query_len = max(decode_query_lens)
        else:
            max_decode_query_len = 1

        omni_block_tables = make_tensor_with_pad(
            self.omni_block_tables,
            pad=0,
            dtype=torch.int,
            device=device,
            max_len=(SINK + RECENT) // 128,
        )  # pad dim 1 for block tables

        seq_lens_tensor = async_tensor_h2d(seq_lens, torch.int,
                                           device, self.runner.pin_memory)
        query_lens_tensor = async_tensor_h2d(query_lens, torch.int32,
                                             "cpu", self.runner.pin_memory)
        slot_mapping_tensor = async_tensor_h2d(self.slot_mapping, torch.long,
                                               device, self.runner.pin_memory)
        omni_slot_tensor = async_tensor_h2d(self.omni_slot_mapping, torch.long,
                                            device, self.runner.pin_memory)
        omni_seq_lens_tensor = torch.clamp(seq_lens_tensor, max=SINK+RECENT).to(dtype=torch.int64, device='cpu')

        return OmniFXAttentionMetadata(
            num_prefills=self.num_prefills,
            slot_mapping=slot_mapping_tensor,
            num_prefill_tokens=self.num_prefill_tokens,
            num_decode_tokens=num_decode_tokens,
            seq_lens=seq_lens,
            seq_lens_tensor=seq_lens_tensor,
            max_prefill_seq_len=max_prefill_seq_len,
            max_decode_seq_len=max_decode_seq_len,
            query_lens_tensor=query_lens_tensor,
            query_lens=query_lens,
            block_tables=block_tables,
            relay_attn_prefix_block_tables_len=0,
            use_2d_input=use_2d_input,
            use_cuda_graph=False,
            max_decode_query_len=max_decode_query_len,
            chunked_prefill_enabled=self.input_builder.chunked_prefill_enabled and block_tables.numel() > 0,
            omni_block_tables=omni_block_tables,
            omni_slot_mapping=omni_slot_tensor,
            omni_seq_lens_tensor=omni_seq_lens_tensor,
            omni_seq_lens=omni_seq_lens_tensor.tolist(),
        )


class OmniFXBaseAttentionBackend(FXBaseAttentionBackend):
    """This backend simply **redirects** the classes of metadata and metadata builder to
    OmniFXAttentionMetadata and OmniFxMetadataBuilder.
    """
    @staticmethod
    def get_metadata_cls() -> Type[OmniFXAttentionMetadata]:
        return OmniFXAttentionMetadata

    @classmethod
    def make_metadata(cls, *args, **kwargs) -> OmniFXAttentionMetadata:
        return cls.get_metadata_cls()(*args, **kwargs)

    @staticmethod
    def get_builder_cls() -> Type[OmniFxMetadataBuilder]:
        return OmniFxMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(num_blocks, block_size, num_kv_heads, head_size):
        return num_blocks, num_kv_heads, block_size, head_size


class OmniCacheEngine(AscendCacheEngine):
    """Manages the KV cache when OMNI attention is used.

    This class is responsible for initializing and managing the GPU and CPU KV
    caches. It allocates more blocks for full attention layers and less for
    omni attention layers, i.e., those layers where KV cache is pruned.
    """

    def __init__(
        self,
        cache_config: CacheConfig,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        device_config: DeviceConfig,
    ) -> None:
        if not ENV.omni_attention_enabled:
            raise RuntimeError("Error. OMNI attention is not enabled. Shouldn't call this class.")
        super().__init__(cache_config=cache_config,
                         model_config=model_config,
                         parallel_config=parallel_config,
                         device_config=device_config)
        self._log_after_init()

    def _log_after_init(self):
        is_tensor = isinstance(self.gpu_cache[0], torch.Tensor)
        for cache, device in ((self.gpu_cache, current_platform.device_type), (self.cpu_cache, "CPU")):
            msg = f"For OMNI attention, the shape of KV cache on {device} in each layer is:"
            for layer_idx in range(self.num_attention_layers):
                msg += "\n\t" + f"{layer_idx:2d}: ---> " + (str(tuple(cache[layer_idx].shape)) if is_tensor else
                    f"k: {tuple(cache[layer_idx][0].shape)}, v: {tuple(cache[layer_idx][1].shape)}")
            _log_at_rank_0(msg)

        max_concurrency = self.num_gpu_blocks * self.cache_config.block_size / self.model_config.max_model_len
        max_concurrency = min(max_concurrency, self.num_omni_gpu_blocks
                                               * self.cache_config.block_size
                                               / (SINK+RECENT))
        _log_at_rank_0("For OMNI attention, the maximum concurrency for"
                       f" {self.model_config.max_model_len} tokens per request: {max_concurrency:.2f}x")

    @staticmethod
    def adapt_num_blocks(num_blocks: int) -> Tuple[int, int]:
        """Compute the number of blocks for both **full attention** and **omni attention** layers.
        The former would be allocated with many more blocks.
        """
        omni_num_blocks = int(num_blocks * BETA)
        full_num_blocks = 2 * num_blocks - omni_num_blocks
        return full_num_blocks, omni_num_blocks

    @override
    def _allocate_kv_cache(
        self,
        num_blocks: int,
        device: str,
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Allocates KV cache on the specified device, with different shapes per layer. This function is implemented
        based on :func:`~vllm_npu.common.worker.cache_engine._allocate_split_kv_cache`. The implementation should be
        adjusted to match the referred function upon any changes to the latter in the future.
        """
        pattern = ENV.omni_attention_pattern

        # get shapes of two types of layers
        full_num_blocks, omni_num_blocks = self.adapt_num_blocks(num_blocks)
        full_kv_cache_shape = self.attn_backend.get_kv_cache_shape(
            full_num_blocks, self.block_size, self.num_kv_heads, self.head_size)
        omni_kv_cache_shape = self.attn_backend.get_kv_cache_shape(
            omni_num_blocks, self.block_size, self.num_kv_heads, self.head_size)

        # override the original `self.num_gpu_blocks` or `self.num_cpu_blocks` attributes
        if device == 'cpu':
            _log_at_rank_0(f"Overriding `CacheEngine.num_cpu_blocks` from {self.num_cpu_blocks} to {full_num_blocks}.")
            self.num_cpu_blocks = full_num_blocks
            self.num_omni_cpu_blocks = omni_num_blocks
        else:
            _log_at_rank_0(f"Overriding `CacheEngine.num_gpu_blocks` from {self.num_gpu_blocks} to {full_num_blocks}.")
            self.num_gpu_blocks = full_num_blocks
            self.num_omni_gpu_blocks = omni_num_blocks

        pin_memory = is_pin_memory_available() if device == "cpu" else False
        init_kw_args = dict(dtype=self.dtype, pin_memory=pin_memory, device=device)

        kv_cache: List[Tuple[torch.Tensor, torch.Tensor]] = []
        if hasattr(self.model_config.hf_text_config, "model_type") \
                and self.model_config.hf_text_config.model_type in ['deepseek_v2', 'deepseek_v3', 'eagle']:
            full_kr_cache_shape = self.attn_backend.get_kv_cache_shape(
                full_num_blocks, self.block_size, self.num_kv_heads, 64)
            omni_kr_cache_shape = self.attn_backend.get_kv_cache_shape(
                omni_num_blocks, self.block_size, self.num_kv_heads, 64)
            for layer_idx in range(self.num_attention_layers):
                key = torch.zeros(
                    omni_kv_cache_shape if pattern[layer_idx] else full_kv_cache_shape,
                    **init_kw_args,
                )
                value = torch.zeros(
                    omni_kr_cache_shape if pattern[layer_idx] else full_kr_cache_shape,
                    **init_kw_args,
                )
                kv_cache.append((key, value))
        else:
            for layer_idx in range(self.num_attention_layers):
                kv_cache_shape = omni_kv_cache_shape if pattern[layer_idx] else full_kv_cache_shape
                key = torch.empty(kv_cache_shape, **init_kw_args)
                value = torch.empty(kv_cache_shape, **init_kw_args)
                kv_cache.append((key, value))
        return kv_cache


class OmniMLA(AscendDeepseekAttention_MLA):
    @override
    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)

        if self.layer_idx == 0:
            # initialize pattern at first layer
            if ENV.omni_attention_pattern is not None:
                raise RuntimeError("Omni attention pattern is already initialized.")
            global_rank = get_world_group().rank_in_group
            device = f'npu:{torch.npu.current_device()}'

            if global_rank == 0:
                with open(ENV.omni_attention_pattern_file, 'r') as fi:
                    pattern = fi.read().strip()
                pattern = ast.literal_eval(pattern)[:config.num_hidden_layers]
                pattern = torch.tensor(pattern, dtype=torch.int32).to(device)
            else:
                pattern = -1 * torch.ones(config.num_hidden_layers, dtype=torch.int32).to(device)

            get_world_group().broadcast(pattern)
            pattern = pattern.cpu().tolist()
            if not (isinstance(pattern, list) and all([pi in [0, 1] for pi in pattern])):
                raise RuntimeError(f"Pattern should be a list of 0s and 1s, but got {pattern}.")
            ENV.omni_attention_pattern = pattern

            # log at every rank
            sparsity = sum(pattern) / len(pattern) * 100
            logger.warning(f"OMNI attention has been enabled with {SINK=}, {RECENT=}, {BETA=},"
                           f" and pattern={ENV.omni_attention_pattern} (SPARSITY={sparsity:.1f}%).")

        if ENV.omni_attention_pattern is None:
            raise RuntimeError("Omni attention pattern is not correctly initialized.")
        self.omni_attn = ENV.omni_attention_pattern[self.layer_idx] == 1

    @override
    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: OmniFXAttentionMetadata,
    ) -> torch.Tensor:
        if not self.is_init:
            self.W_UK = torch.nn.Parameter(self.W_UK.contiguous(), requires_grad=False)
            self.W_UV = torch.nn.Parameter(self.W_UV.contiguous(), requires_grad=False)
            self.is_init = True

        if attn_metadata.prefill_metadata is not None:
            return self.forward_normal(positions, hidden_states, kv_cache, attn_metadata)
        else:
            return self.forward_absorb(positions, hidden_states, kv_cache, attn_metadata)

    @override
    def forward_normal(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: OmniFXAttentionMetadata,
    ) -> torch.Tensor:
        if self.q_lora_rank is not None:
            if self.merge_qkv:
                qkv = self.qkv_a_proj(hidden_states)[0]
                qkv = tensor_model_parallel_all_gather(qkv, dim=0)
                q, kv_states = splitV(qkv, [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim], dim=-1)
            else:
                q = self.q_a_proj(hidden_states)[0]
                kv_states = self.kv_a_proj_with_mqa(hidden_states)[0]
                if not ENV.enable_pd_separated:
                    q = tensor_model_parallel_all_gather(q, dim=0)
                kv_states = tensor_model_parallel_all_gather(kv_states, dim=0)
            q = self.q_a_layernorm(q)
            q = self.q_b_proj(q)[0].view(-1, self.num_local_heads, self.qk_head_dim)

        else:
            q = self.q_proj(hidden_states)[0].view(-1, self.num_local_heads, self.qk_head_dim)
            kv_states = self.kv_a_proj_with_mqa(hidden_states)[0]
            q = tensor_model_parallel_all_gather(q, dim=0)
            kv_states = tensor_model_parallel_all_gather(kv_states, dim=0)

        # q         shape:  (num_tokens, num_heads, head_dim+rope_dim)
        # kv_states shape:  (num_tokens, lora_dim+rope_dim)

        _, q_pe = splitV(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        q_pe = q_pe.unsqueeze(2)  # (num_tokens, num_heads, 1, rope_dim)
        cos, sin = self.rotary_emb.get_cos_sin(positions)  # (num_tokens, rope_dim)
        q_pe = torch.ops.npu_inference.npu_interleave_rope(q_pe, cos, sin)
        q_pe = q_pe.squeeze(2) # (num_tokens, num_heads, rope_dim)
        q[..., self.qk_nope_head_dim:] = q_pe

        if isinstance(kv_cache, Dict):
            kv_cache = kv_cache.get("kv_cache")
        if kv_cache is not None and isinstance(kv_cache, Tuple) and kv_cache[0].numel() > 0:
            if self.omni_attn:
                slot_mapping = attn_metadata.omni_slot_mapping
            else:
                slot_mapping = attn_metadata.slot_mapping
            _, _, k_pe, kv_a = torch_npu.npu_kv_rmsnorm_rope_cache(
                kv_states.view(-1, 1, 1, 576),
                self.kv_a_layernorm.weight,
                cos.view(-1, 1, 1, self.qk_rope_head_dim),
                sin.view(-1, 1, 1, self.qk_rope_head_dim),
                slot_mapping,
                kv_cache[1].view(-1, 128, 1, 64),
                kv_cache[0].view(-1, 128, 1, 512),
                epsilon=self.kv_a_layernorm.variance_epsilon,
                cache_mode="PA_NZ",
                is_output_kv=True,
            ) # adapter NZ
        else:
            # no cache available -> profiling
            kv_states = kv_states.view(-1, kv_states.size(-1))
            kv_a, _ = splitV(kv_states, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
            kv_states = kv_states.unsqueeze(1)
            kv_a = self.kv_a_layernorm(kv_a)
            k_pe = kv_states[:, :, self.kv_lora_rank:]
            k_pe = k_pe.unsqueeze(2)
            k_pe = torch.ops.npu_inference.npu_interleave_rope(k_pe, cos, sin)
            k_pe = k_pe.squeeze(2)

        # shape of kv is: (num_tokens, num_heads*head_dim*2)
        kv = self.kv_b_proj.forward(kv_a.view(-1, self.kv_lora_rank))[0]
        kv = kv.view(-1, self.num_local_heads, self.qk_nope_head_dim + self.v_head_dim)
        k_nope, v = splitV(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)

        k = torch.empty([k_nope.shape[0], k_nope.shape[1], self.qk_nope_head_dim + self.qk_rope_head_dim],
                        dtype=k_nope.dtype, device=k_nope.device)

        k[..., :self.qk_nope_head_dim] = k_nope
        k[..., self.qk_nope_head_dim:] = k_pe.view(-1, 1, self.qk_rope_head_dim)
        attn_output = self.attn_prefill(q, k, v, kv_cache, attn_metadata)
        # TND  -> TH
        attn_output = attn_output.reshape(-1, self.num_local_heads * self.v_head_dim)
        output, _ = self.o_proj.forward(attn_output)
        return output

    @override
    def forward_absorb(
            self,
            positions: torch.Tensor,
            hidden_states: torch.Tensor,
            kv_cache: torch.Tensor,
            attn_metadata: OmniFXAttentionMetadata,
    ) -> torch.Tensor:
        hidden_states = tensor_model_parallel_all_gather(hidden_states, dim=0)
        key_cache, value_cache = kv_cache
        output_dtype = key_cache.dtype

        if self.q_lora_rank is not None:
            q_lowrank = self.q_a_proj(hidden_states)[0]
        else:
            q_lowrank = self.q_proj(hidden_states)[0]

        with tng.ops.NpuStreamSwitch('11'):
            kv = hidden_states
            kv = self.kv_a_proj_with_mqa(kv)[0]

        tng.ops.npu_wait_tensor(q_lowrank, q_lowrank)
        if self.q_lora_rank is not None:
            q, _ = self.q_a_layernorm(q_lowrank, self.norm_res)
            q = self.q_b_proj(q)[0]
        else:
            q = q_lowrank
        bsz, _ = q.shape
        q_len = 1
        q = q.view(bsz, self.num_local_heads, 1, self.qk_head_dim)
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1) # b,n,s,d

        q_nope = q_nope.view(-1, self.num_local_heads, self.qk_nope_head_dim).transpose(0, 1) # n, bs, d
        q_nope = (
            torch.matmul(q_nope, self.W_UK)
            .transpose(1, 0)
            .view(bsz, q_len, self.num_local_heads, -1)
        )

        if self.omni_attn:
            slot_mapping = attn_metadata.omni_slot_mapping
            block_tables = attn_metadata.omni_block_tables
            seq_lens = attn_metadata.omni_seq_lens
        else:
            slot_mapping = attn_metadata.slot_mapping
            block_tables = attn_metadata.block_tables
            seq_lens = attn_metadata.seq_lens

        with tng.ops.NpuStreamSwitch('11'):
            kv = kv.unsqueeze(1).unsqueeze(1)
            cos, sin = attn_metadata.decode_metadata.cos, attn_metadata.decode_metadata.sin
            block_num, head_size, block_size, _ = key_cache.shape
            k_rope, k_nope, _, _ = torch_npu.npu_kv_rmsnorm_rope_cache(
                kv, self.kv_a_layernorm.weight,
                cos, sin, slot_mapping,
                value_cache.view(-1, 128, 1, 64), key_cache.view(-1, 128, 1, 512),
                epsilon=self.kv_a_layernorm.variance_epsilon, cache_mode="PA_NZ") # adapter NZ

            # adapter nz
            k_nope = k_nope.view(block_num, 1, self.kv_lora_rank // 16, block_size, 16)
            k_rope = k_rope.view(block_num, 1, self.qk_rope_head_dim // 16, block_size, 16)

            tng.ops.npu_wait_tensor(q_pe, k_nope)

            q_pe = torch.ops.npu_inference.npu_interleave_rope(q_pe, cos, sin) # BNSD
            q_nope = q_nope.view(bsz, 1, self.num_local_heads, self.kv_lora_rank)
            q_pe = q_pe.view(bsz, 1, self.num_local_heads, -1)

        bsz, q_len, _, q_dim = q_nope.size()
        attn_output, _ = torch.ops.npu.npu_fused_infer_attention_score(
                q_nope, k_nope, k_nope, query_rope=q_pe, key_rope=k_rope,
                num_heads=self.num_local_heads,
                num_key_value_heads=1, input_layout="BSND_NBSD",
                scale=self.scaling,
                antiquant_mode=0, antiquant_scale=None,
                block_table=block_tables,
                block_size=128,
                actual_seq_lengths_kv=seq_lens,
                )

        # Apply UV, (N, B, L) @ W_UV (N, L, V) -> (N, B, V)
        attn_output = attn_output.view(self.num_local_heads, bsz*q_len, self.kv_lora_rank) # adapter BSND_NBSD
        attn_output = (
            torch.matmul(attn_output, self.W_UV)
            .transpose(1, 0)
            .reshape(bsz, q_len, -1)
        )
        attn_output = attn_output.view(
            -1, self.num_local_heads * self.v_head_dim)
        output, _ = self.o_proj.forward(attn_output)
        return output
