# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional, Tuple, Type, TypeVar, Dict
import itertools
import numpy as np
import torch
import torch_npu
import torch.distributed as dist
import torchair as tng
 
from vllm.attention.backends.abstract import (
    AttentionBackend,
    AttentionMetadata,
    MLAAttentionImpl
)
from vllm.attention.backends.utils import PAD_SLOT_ID
from vllm.config import get_current_vllm_config
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    RowParallelLinear,
    ReplicatedLinear
)
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding
from vllm.distributed import get_world_group 
from vllm.distributed.communication_op import tensor_model_parallel_all_gather
from vllm.distributed.parallel_state import (
    get_tensor_model_parallel_world_size,
)
from omni.models.common.config.model_config import model_extra_config
from vllm.platforms import current_platform
from omni.models.common.layers.attention.attention import AscendAttentionState
from omni.adaptors.vllm.worker.npu_model_runner import NPUModelRunner
from omni.adaptors.vllm.worker.npu_model_runner import DummyAttentionMetadataBuilder

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.worker.gpu_input_batch import InputBatch

KVCACHE_NZ_DIM = 16


def generate_activate_mask(actual_seqs_num, batch_size):
    decode_gear_list = model_extra_config.operator_opt_config.decode_gear_list
    gear = next((g for g in decode_gear_list if g >= batch_size), decode_gear_list[-1])
    mc2_mask = torch.zeros(gear, dtype=torch.bool, device=current_platform.device_type)
    mc2_mask[:actual_seqs_num] = True
    return mc2_mask

def group_request_list(seq_lens, query_lens, block_tables, threshold):
    s_lens_result = []
    q_lens_result = []
    blocks_result = []
    s_lens_current_group = []
    q_lens_current_group = []
    blocks_current_group = []
    current_sum = 0
    for seq_len, query_len, block_table in zip(seq_lens, query_lens, block_tables):
        if current_sum + seq_len > threshold and len(s_lens_current_group) > 0:
            s_lens_result.append(s_lens_current_group)
            q_lens_result.append(q_lens_current_group)
            blocks_result.append(blocks_current_group)
            s_lens_current_group = []
            q_lens_current_group = []
            blocks_current_group = []
            current_sum = 0
        s_lens_current_group.append(seq_len)
        q_lens_current_group.append(query_len)
        blocks_current_group.append(block_table)
        current_sum += seq_len
    if q_lens_current_group:
        s_lens_result.append(s_lens_current_group)
        q_lens_result.append(q_lens_current_group)
        blocks_result.append(blocks_current_group)
    return s_lens_result, q_lens_result, blocks_result

class AscendMLABackend(AttentionBackend):

    accept_output_buffer: bool = True

    @staticmethod
    def get_name() -> str:
        return "VLLM_ASCEND_MLA"

    @staticmethod
    def get_metadata_cls() -> type["AttentionMetadata"]:
        return AscendMLAMetadata

    @staticmethod
    def get_builder_cls():
        return AscendMLAMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(num_blocks: int, block_size: int, num_kv_heads: int,
                           head_size: int) -> tuple[int, ...]:
        return (num_blocks, block_size, 1, 512 + 64)

    @staticmethod
    def get_impl_cls() -> Type["MLAAttentionImpl"]:
        return AscendMLAImpl

    @staticmethod
    def init_kv_cache_each_layer(kv_cache_shape, dtype, device, model_config: "ModelConfig", enable_graph_mode) -> tuple[torch.Tensor, ...]:
        # KVCache needs to store the shape of the reduced dimension as [num_blocks, block_size, 1, kv_lora_rank] [num_blocks, block_size, 1, rope_dim]
        # The shape of the augmented dimension is [num_blocks, block_size, head_num, head_dim]
        layer_kv_cache_nope = torch.zeros(
                        kv_cache_shape[:-2] +
                        (1, model_config.hf_config.kv_lora_rank, ),
                        dtype=dtype,
                        pin_memory=True,
                        device=device)
        layer_kv_cache_pe = torch.zeros(
                            kv_cache_shape[:-2] +
                            (1, model_config.hf_config.qk_rope_head_dim, ),
                            dtype=dtype,
                            pin_memory=True,
                            device=device)
        return (layer_kv_cache_nope, layer_kv_cache_pe)

@dataclass
class AscendMLAPrefillMetadata:
    """ Prefill Specific Metadata for Ascend"""
    attn_mask: torch.Tensor
    query_lens: list[int]
    seq_lens: list[int]
    input_positions: torch.Tensor
    block_table: torch.Tensor
    max_query_len: int

    # adaptor for chunk-prefill & prefix-caching use
    seq_qlen_group: Optional[list] = None
    seq_kvlen_group: Optional[list] = None
    kv_index_list: Optional[list] = None

@dataclass
class AscendMLADecodeMetadata:
    # Input positions for rotrary embeddings since for MLA the rotary
    # position embeddings are applied inside the attention backend
    input_positions: torch.Tensor
    block_table: torch.Tensor
    seq_lens: torch.Tensor
    mc2_mask: Optional[torch.Tensor] = None
    cos: Optional[torch.Tensor] = None
    sin: Optional[torch.Tensor] = None
    best_topk: Optional[torch.Tensor] = None


@dataclass
class AscendMLAMetadata:
    """Metadata for MLACommon.

    NOTE: Please read the comment at the top of the file before trying to
    understand this class
    """
    # NOTE(sang): Definition of context_len, query_len, and seq_len.
    # |---------- N-1 iteration --------|
    # |---------------- N iteration ---------------------|
    # |- tokenA -|......................|-- newTokens ---|
    # |---------- context_len ----------|
    # |-------------------- seq_len ---------------------|
    #                                   |-- query_len ---|

    num_actual_tokens: int  # Number of tokens excluding padding.
    slot_mapping: torch.Tensor

    # New for MLA (compared to FlashAttention)
    # For handling prefill decode split
    num_decodes: int
    num_decode_tokens: int
    num_prefills: int

    # For logging.
    num_input_tokens: int = 0  # Number of tokens including padding.

    # The dimension of the attention heads
    head_dim: Optional[int] = None
    attn_mask: torch.Tensor = None
    # chunked prefill by default if no attn_states passed
    attn_state: AscendAttentionState = AscendAttentionState.ChunkedPrefill

    decode: Optional[AscendMLADecodeMetadata] = None
    prefill: Optional[AscendMLAPrefillMetadata] = None

    def __post_init__(self):
        pass


M = TypeVar("M", bound=AscendMLAMetadata)


class AscendMLAMetadataBuilder(DummyAttentionMetadataBuilder):
    """
    NOTE: Please read the comment at the top of the file before trying to
    understand this class
    """

    def __init__(self,
                 runner: "NPUModelRunner",
                 metadata_cls: Optional[AscendMLAMetadata] = None):
        self.metadata_cls: Optional[AscendMLAMetadata] = metadata_cls \
            if metadata_cls is not None else AscendMLAMetadata  # type: ignore
        self.runner = runner
        scheduler_config = runner.scheduler_config
        self.chunked_prefill_enabled = scheduler_config.chunked_prefill_enabled
        self.block_size = self.runner.block_size
        self.base_index = np.array(list(range(0, self.block_size)))
        self.base_block = self.block_size * np.ones([1, self.block_size])

    def reorder_batch(self, input_batch: "InputBatch",
                      scheduler_output: "SchedulerOutput") -> bool:
        # We now want to reorder the batch so that the "decode" requests are at
        # the front and the "prefill" requests are at the using the least amount
        # swaps possible. (NOTE for now we loosely use "decode" to mean requests
        # where attention is likely memory-bound and "prefill" to mean requests
        # where attention is likely compute-bound
        decodes = []
        prefills = []
        num_decode_tokens = 0
        num_prefill_tokens = 0

        for i, req_id in enumerate(input_batch.req_ids):
            num_tokens = scheduler_output.num_scheduled_tokens[req_id]
            # for now treat 1 scheduled token as "decode" even if its not,
            # we should update this to something like < 8 in the future but
            # currently the TritonMLA._forward_decode only supports
            # num_tokens = 1
            # Only in decode the spec tokens are scheduled
            if req_id in scheduler_output.scheduled_spec_decode_tokens or num_tokens == 1:
                decodes.append(i)
                num_decode_tokens += num_tokens
            else:
                prefills.append(i)
                num_prefill_tokens += num_tokens

        # We hope that this is fairly minimal since decodes
        # should be around for a number of iterations so hopefully they are
        # relatively stationary (and new request are generally appended to the
        # persistent batch so already should be at the back)
        # To achieve this we loop over the decodes in descending order and
        # the prefills in ascending order. We swap decodes from the  "back"
        # i.e. past where the last decode should be in the reodorered with
        # prefills from the front of the batch.
        # `decodes` and `prefills` are already in ascending order just based on
        # the above loop
        num_decodes = len(decodes)
        num_prefills = len(prefills)
        first_prefill = 0
        modified_batch = False

        for i in range(1, min(num_decodes, num_prefills) + 1):
            # If the decode is at the "back" of the batch, i, we can swap it
            # with the prefill closest to the front of the batch
            if decodes[num_decodes - i] >= num_decodes:
                input_batch.swap_states(prefills[first_prefill],
                                        decodes[num_decodes - i])
                first_prefill += 1
                modified_batch = True
            else:
                break

        # Save for next `build` call
        self._num_decodes = num_decodes
        self._num_prefills = num_prefills
        self._num_decode_tokens = num_decode_tokens
        self._num_prefill_tokens = num_prefill_tokens

        return modified_batch

    def cal_best_topk(self, cur_batch_size):
        world_size = get_world_group().world_size
        batch_size = cur_batch_size * world_size
        top_k = self.runner.model.config.num_experts_per_tok
        step = batch_size // world_size * top_k
        global_rank = get_world_group().rank_in_group
        experts_tp_size = 1
        cur_topk_list = [
            i % self.runner.model.config.n_routed_experts for i in range(
            global_rank // experts_tp_size * step, (global_rank // experts_tp_size + 1) * step)]
        return torch.Tensor(cur_topk_list).to(dtype=torch.int32, device="npu", non_blocking=True).view(batch_size // world_size, -1)

    def _get_graph_runner_block_tables(
            self, num_decode_tokens: int, block_tables: torch.Tensor) -> torch.Tensor:

        max_batch_size, max_blocks = self.runner.graph_block_tables.shape
        if max_batch_size < num_decode_tokens:
            raise RuntimeError("max_batch_size must be greater than or equal to num_decode_tokens")

        if isinstance(self.runner.graph_block_tables, np.ndarray):
            graph_block_tables = torch.zeros((max_batch_size, max_blocks),
                                             dtype=block_tables.dtype,
                                             device=block_tables.device)
        else:
            graph_block_tables = self.runner.graph_block_tables.to(
                device=block_tables.device, dtype=block_tables.dtype, non_blocking=True)

        num_blocks = block_tables.size(1)
        if num_blocks <= max_blocks:
            graph_block_tables[:num_decode_tokens, :
                               num_blocks] = block_tables[:num_decode_tokens, :
                                                          num_blocks]
        else:
            graph_block_tables[:num_decode_tokens, :
                               max_blocks] = block_tables[:num_decode_tokens, :
                                                          max_blocks]

        return graph_block_tables

    def get_kv_index(self, seq_lens, block_tables):
        kv_index = []
        for seq_len, block_table in zip(seq_lens, block_tables):
            index = self.base_index + np.expand_dims(block_table.cpu().numpy(), axis=-1) * self.base_block.repeat(block_table.shape[0], axis=0)
            kv_index.append(index.reshape(-1)[:seq_len])
        return torch.tensor(np.concatenate(kv_index, axis=0), dtype=torch.long, device="cpu").npu()

    def build(self,
              num_reqs: int,
              num_actual_tokens: int,
              max_query_len: int,
              common_prefix_len: Optional[int] = None,
              graph_pad_size: int = -1) -> AscendMLAMetadata:
        if self._num_decodes + self._num_prefills != num_reqs:
            raise RuntimeError("self._num_decodes + self._num_prefills must be equal to num_reqs")

        # Note(simon): be careful about the CPU <> GPU memory movement in this
        # function. We should avoid GPU -> CPU sync as much as possible because
        # it blocks on all previous kernels.
        device = self.runner.device
        block_table = (self.runner.input_batch.block_table[0].get_device_tensor()[:num_reqs])

        slot_mapping = self.runner.slot_mapping_cpu[:num_actual_tokens].to(
            device, non_blocking=True)
        input_positions = self.runner.positions_cpu[:num_actual_tokens].to(
            device, non_blocking=True)

        seq_lens = self.runner.seq_lens_cpu[:num_reqs].to(device, non_blocking=True)
        seq_lens_list = self.runner.seq_lens_cpu[:num_reqs].tolist()

        # prefill也需要做padding，不然算子shape会报错
        if graph_pad_size > 0:
            padding = torch.full((graph_pad_size, ),
                                    PAD_SLOT_ID,
                                    dtype=slot_mapping.dtype,
                                    device=device)
            slot_mapping = torch.cat([slot_mapping, padding])
            padding_0 = torch.zeros(graph_pad_size,
                                    dtype=input_positions.dtype,
                                    device=device)
            input_positions = torch.cat([input_positions, padding_0])

        prefill_metadata = None
        if self._num_prefills > 0:
            query_lens_list = seq_lens_list

            reqs_start = self._num_decodes  # prefill_start
            tokens_start = self._num_decode_tokens

            # Group request for Chunk-Prefill
            seq_kvlen_group, seq_qlen_group, block_groups = group_request_list(
                seq_lens_list,
                query_lens_list,
                block_table,
                self.runner.max_num_tokens)
            
            # Prepare kv index for prefill get kv_latent from kv_cache
            kv_index_list = None
            # if block_table is not None and block_table.numel() > 0:
            #     for seq_lens, block_tables in zip(seq_kvlen_group, block_groups):
            #         kv_index = self.get_kv_index(seq_lens, block_tables)
            #         kv_index_list.append(kv_index)

            seq_qlen_group = [list(itertools.accumulate(sub_list)) for sub_list in seq_qlen_group]
            seq_kvlen_group = [list(itertools.accumulate(sub_list)) for sub_list in seq_kvlen_group]

            prefill_metadata = AscendMLAPrefillMetadata(
                attn_mask=self.runner.attn_mask,
                query_lens=query_lens_list[reqs_start:],
                seq_lens=seq_lens_list,
                input_positions=input_positions[tokens_start:],
                block_table=block_table[reqs_start:, ...],
                max_query_len=max_query_len,
                seq_qlen_group=seq_qlen_group,
                seq_kvlen_group=seq_kvlen_group,
                kv_index_list=kv_index_list
            )


        decode_metadata = None

        if self._num_decodes > 0:
            if self.runner.attn_state == AscendAttentionState.DecodeOnly:
                if self._num_decode_tokens % self._num_decodes != 0:
                    raise RuntimeError("self._num_decode_tokens must be divisible by self._num_decodes")
                num_tokens_per_req = self._num_decode_tokens // self._num_decodes
                seq_lens = (input_positions + 1).to(seq_lens.dtype)
                block_table = block_table[:self._num_decodes, ...]
                # has speculative tokens
                if num_tokens_per_req > 1:
                    block_table = block_table.unsqueeze(1).repeat(1, num_tokens_per_req, 1).view(-1, block_table.shape[-1])
                block_table_padding = torch.zeros(
                    (graph_pad_size, ) + block_table.shape[1:],
                    dtype=block_table.dtype,
                    device=block_table.device)
                block_table = torch.cat([block_table, block_table_padding],
                                        dim=0)
                block_table = self._get_graph_runner_block_tables(
                    self._num_decode_tokens, block_table)

                mc2_mask = generate_activate_mask(num_actual_tokens, num_actual_tokens + graph_pad_size)
                cos, sin = self.runner.model.model.layers[0].self_attn.rotary_emb.get_cos_sin(input_positions)
                best_topk = None
                if model_extra_config.operator_opt_config.best_ep:
                    best_topk = self.cal_best_topk(num_reqs + graph_pad_size)
            else:
                raise NotImplementedError("Chunked prefill mode is not supported currently.")

            decode_metadata = AscendMLADecodeMetadata(
                input_positions=input_positions,
                block_table=block_table,
                seq_lens=seq_lens,
                mc2_mask=mc2_mask,
                cos=cos,
                sin=sin,
                best_topk=best_topk)

        return self.metadata_cls(  # type: ignore
            num_actual_tokens=num_actual_tokens,
            slot_mapping=slot_mapping,
            num_decodes=self._num_decodes,
            num_decode_tokens=self._num_decode_tokens,
            num_prefills=self._num_prefills,
            attn_mask=self.runner.attn_mask,
            attn_state=self.runner.attn_state,
            prefill=prefill_metadata,
            decode=decode_metadata,
        )

    def build_dummy(self, num_tokens: int, max_pad_size:int = -1) -> AscendMLAMetadata:
        if max_pad_size == -1:
            max_pad_size = self.runner.max_batch_size
        input_positions = torch.zeros(max_pad_size,
                                  dtype=self.runner.positions_cpu.dtype,
                                  device=self.runner.device)
        slot_mapping = torch.zeros(max_pad_size,
                                dtype=self.runner.slot_mapping_cpu.dtype,
                                device=self.runner.device)
        if isinstance(self.runner.graph_block_tables, np.ndarray):
            graph_block_tables = torch.zeros((max_pad_size, self.runner.graph_block_tables.shape[1]))
        block_table = graph_block_tables.to(
            device=self.runner.device, 
            dtype=self.runner.input_batch.block_table[0].get_device_tensor().dtype
        )
 
        seq_lens = torch.ones(max_pad_size, dtype=torch.long, device=self.runner.device, pin_memory=True) * 2
        cos, sin = self.runner.model.model.layers[0].self_attn.rotary_emb.get_cos_sin(input_positions)
        best_topk = None
        mc2_mask = generate_activate_mask(num_tokens, max_pad_size)
        if model_extra_config.operator_opt_config.best_ep:
            best_topk = self.cal_best_topk(max_pad_size)
        decode_metadata = AscendMLADecodeMetadata(
                input_positions=input_positions,
                block_table=block_table,
                seq_lens=seq_lens,
                mc2_mask=mc2_mask,
                cos=cos,
                sin=sin,
                best_topk=best_topk)
        return self.metadata_cls(  # type: ignore
            num_actual_tokens=num_tokens,
            slot_mapping=slot_mapping,
            num_decodes=num_tokens,
            num_decode_tokens=num_tokens,
            num_prefills=0,
            attn_mask=self.runner.attn_mask,
            attn_state=self.runner.attn_state,
            prefill=None,
            decode=decode_metadata,
        )

class AscendMLAImpl(MLAAttentionImpl):
    """
    NOTE: Please read the comment at the top of the file before trying to
    understand this class
    """

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: Optional[list[float]],
        sliding_window: Optional[int],
        kv_cache_dtype: str,
        blocksparse_params: Optional[dict[str, Any]],
        logits_soft_cap: Optional[float],
        attn_type: str,
        # MLA Specific Arguments
        q_lora_rank: Optional[int],
        kv_lora_rank: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        qk_head_dim: int,
        v_head_dim: int,
        rotary_emb: RotaryEmbedding,
        # q_proj should be q_b_proj if q_lora_rank is not None, but from an
        # attention backend perspective we rely on the layer to pass in the
        # correct matrix
        q_proj: ColumnParallelLinear,
        kv_b_proj: ColumnParallelLinear,
        o_proj: RowParallelLinear,
        qkv_a_proj: ReplicatedLinear,
        q_a_layernorm: RMSNorm,
        q_b_proj: ColumnParallelLinear,
        q_a_proj: ReplicatedLinear,
        prefix: str = "",
        **kwargs,
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        self.kv_cache_dtype = kv_cache_dtype

        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_head_dim
        self.v_head_dim = v_head_dim
        # we found npu_flash_attention can only works on 128 divisible head_dim, we pad it to target size here
        # and slice the final result to guarantee its functionality.
        self.padding_head_dim = (
            (self.qk_nope_head_dim + self.qk_rope_head_dim - 1) // 128 +
            1) * 128

        # Hack for V1 for now to avoid torch library overhead (since we are
        # already inside an attention custom op), pull out the forward
        # method from the rotary embedding and call it directly
        self.rotary_emb = rotary_emb

        self.q_proj = q_proj
        self.kv_b_proj = kv_b_proj
        self.o_proj = o_proj

        self.kv_a_proj_with_mqa = kwargs.get('kv_a_proj_with_mqa', None)
        self.kv_a_layernorm = kwargs.get('kv_a_layernorm', None)
        # Handle the differences between the flash_attn_varlen from flash_attn
        # and the one from vllm_flash_attn. The former is used on RoCM and the
        # latter has an additional parameter to control FA2 vs FA3
        # self.flash_attn_varlen_func = flash_attn_varlen_func
        # if self.vllm_flash_attn_version is not None:
        #     self.flash_attn_varlen_func = \
        #         functools.partial(flash_attn_varlen_func,
        #                           fa_version=self.vllm_flash_attn_version)

        self.enable_graph_mode = False
        additional_config = get_current_vllm_config().additional_config
        if additional_config:
            self.enable_graph_mode = additional_config.get(
                "enable_graph_mode", False)
        
        self.attn_mask = ~torch.tril(
            torch.ones((2048, 2048), dtype=torch.bool, device=current_platform.device_type)
        )
        self.merge_qkv = model_extra_config.operator_opt_config.merge_qkv
        self.qkv_a_proj = qkv_a_proj
        self.q_a_layernorm = q_a_layernorm
        self.q_b_proj = q_b_proj

        self.num_local_heads = num_heads
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.scaling = self.qk_head_dim ** -0.5
        self.q_a_proj = q_a_proj
        self.kv_scale = None
        self.use_faquant = False
        kv_b_proj_weight = self.kv_b_proj.weight.T

        expected_shape = (
            self.kv_lora_rank,
            self.num_local_heads * (self.qk_nope_head_dim + self.v_head_dim)
        )
        if kv_b_proj_weight.shape != expected_shape:
            raise RuntimeError(f"{kv_b_proj_weight.shape} != {expected_shape}")
 
 
 
 
        kv_b_proj_weight = kv_b_proj_weight.view(
            self.kv_lora_rank,
            self.num_local_heads,
            self.qk_nope_head_dim + self.v_head_dim,
        )
        self.W_UK, self.W_UV = kv_b_proj_weight.split(
            [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        self.W_UK = self.W_UK.permute(1, 2, 0)
        self.W_UV = self.W_UV.transpose(0, 1)
        self.is_init = False
        self.prefix = prefix
        # adaptor addrmsnormquant
        tp_size = get_tensor_model_parallel_world_size()
        self.norm_res = {}
        self.actual_seq_lengths = {}
        for batch_size in model_extra_config.operator_opt_config.decode_gear_list:
            self.norm_res[batch_size] = torch.zeros([batch_size * tp_size, self.q_lora_rank], dtype=torch.bfloat16, device=current_platform.device_type)
            self.actual_seq_lengths[batch_size] = torch.tensor(list(range(1, batch_size * tp_size + 1)), dtype=torch.int64, device="npu")
            torch._dynamo.mark_static(self.norm_res[batch_size])
            torch._dynamo.mark_static(self.actual_seq_lengths[batch_size])

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: Optional[torch.Tensor],
        attn_metadata: AttentionMetadata,
 
    ) -> torch.Tensor:
        if not self.is_init:
            self.W_UK = torch.nn.Parameter(self.W_UK.contiguous(), requires_grad=False)
            self.W_UV = torch.nn.Parameter(self.W_UV.contiguous(), requires_grad=False)
            self.empty_out = torch.empty(1, dtype=torch.bfloat16)
            self.is_init = True
        if attn_metadata is None or attn_metadata.prefill is not None:
            output = self._forward_prefill(positions, hidden_states, kv_cache, attn_metadata)
        else:
            output = self._forward_decode(
                positions, hidden_states, kv_cache, attn_metadata, use_rmsnorm_rope_cache=model_extra_config.operator_opt_config.enable_kv_rmsnorm_rope_cache
            )
        return output
 
    def _forward_prefill(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: Optional[torch.Tensor],
        attn_metadata: AttentionMetadata
    ) -> torch.Tensor:
        if self.q_lora_rank is not None:
            if self.merge_qkv:
                qkv = self.qkv_a_proj(hidden_states)[0]
                qkv = tensor_model_parallel_all_gather(qkv, dim=0)
                q, latent_cache = torch.split(qkv, [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim], dim=-1)
 
                q = self.q_a_layernorm(q)
                q = self.q_b_proj(q)[0].view(-1, self.num_local_heads, self.qk_head_dim)
            else:
                q = self.q_a_proj(hidden_states)[0]
                latent_cache = self.kv_a_proj_with_mqa(hidden_states)[0]
                # q = tensor_model_parallel_all_gather(q, dim=0)
                latent_cache = tensor_model_parallel_all_gather(latent_cache, dim=0)
 
                q = self.q_a_layernorm(q)
                q_quant, q_scale = torch_npu.npu_dynamic_quant(q)
                # Quantizing before all_gather can reduce communication overhead.
                q_quant = tensor_model_parallel_all_gather(q_quant, dim=0)
                q_scale = tensor_model_parallel_all_gather(q_scale, dim=0)
                q = {'x_int8':q_quant, 'pertoken_scale':q_scale}
                q = self.q_b_proj(q)[0].view(-1, self.num_local_heads, self.qk_head_dim)
        else:
            q = self.q_proj(hidden_states)[0].view(-1, self.num_local_heads, self.qk_head_dim)
            latent_cache = self.kv_a_proj_with_mqa(hidden_states)[0]
            q = tensor_model_parallel_all_gather(q, dim=0)
            latent_cache = tensor_model_parallel_all_gather(latent_cache, dim=0)
 
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim],  dim=-1)
        # k_pe:BNS,64 kv_a:BNS, 512, kv_states:bnsd, cos,sin:bnsd, kv cache:bsnd
        q_pe = q_pe.unsqueeze(2)
        cos, sin = self.rotary_emb.get_cos_sin(positions)
        q_pe = torch_npu.npu_interleave_rope(q_pe, cos, sin) # BNSD
        q_pe = q_pe.squeeze(2) #BSH
        q[..., self.qk_nope_head_dim:] = q_pe
        if isinstance(kv_cache, Dict):
            kv_cache = kv_cache.get("kv_cache")
        if kv_cache is not None and isinstance(kv_cache, Tuple) and kv_cache[0].numel() > 0:
            # k_pe:BNS,64 kv_a:BNS, 512, kv_states:bnsd, cos,sin:bnsd,kv cache:bsnd
            _, _, k_pe, kv_a = torch_npu.npu_kv_rmsnorm_rope_cache(
                latent_cache.view(-1, 1, 1, 576), # bnsd
                self.kv_a_layernorm.weight,
                cos.view(-1, 1, 1, self.qk_rope_head_dim),
                sin.view(-1, 1, 1, self.qk_rope_head_dim),
                attn_metadata.slot_mapping,
                kv_cache[1],
                kv_cache[0],
                k_rope_scale=None, 
                c_kv_scale=torch.reciprocal(self.kv_scale).repeat(self.kv_lora_rank).view(-1) if self.use_faquant else None,
                k_rope_offset=None, c_kv_offset=None,
                epsilon=self.kv_a_layernorm.variance_epsilon,
                cache_mode="PA_NZ",
                is_output_kv=True) # adapter NZ
        else:
            latent_cache = latent_cache.view(-1, latent_cache.size(-1))
            # adapt end
            kv_a, _ = torch.split(latent_cache, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
            latent_cache = latent_cache.unsqueeze(1)
            kv_a = self.kv_a_layernorm(kv_a)
            k_pe = latent_cache[:, :, self.kv_lora_rank:]
            k_pe = k_pe.unsqueeze(2)
            k_pe = torch_npu.npu_interleave_rope(k_pe, cos, sin)
            k_pe = k_pe.squeeze(2)
        attn_output = torch.empty(q.shape[0],
                        self.num_local_heads,
                        self.v_head_dim,
                        device=q_nope.device,
                        dtype=q_nope.dtype)
 
        if attn_metadata is not None:
            prefill_metadata = attn_metadata.prefill
            computed_tokens = 0
 
            for iter, (actual_seq_qlen, actual_seq_kvlen) in enumerate(zip(
                prefill_metadata.seq_qlen_group,
                prefill_metadata.seq_kvlen_group)
            ):
                if prefill_metadata.kv_index_list and kv_cache is not None and isinstance(kv_cache, Tuple) and\
                        kv_cache[0].numel() > 0:
                    # adapt nz
                    block_num, block_size, head_size, _ = kv_cache[0].shape
                    kv_cache_a = (kv_cache[0]
                                .view(block_num, 1, self.kv_lora_rank // KVCACHE_NZ_DIM, block_size, KVCACHE_NZ_DIM))
                    kv_cache_pe = (kv_cache[1]
                                .view(block_num, 1, self.qk_rope_head_dim // KVCACHE_NZ_DIM, block_size, KVCACHE_NZ_DIM))
                    kv_cache_a = kv_cache_a.transpose(1, 3)
                    kv_cache_pe = kv_cache_pe.transpose(1, 3)
                    # adapt end
                    kv_a = kv_cache_a.reshape(-1, kv_cache[0].shape[-1]) \
                        .index_select(0, prefill_metadata.kv_index_list[iter]).contiguous()
                    k_pe = kv_cache_pe.reshape(-1, kv_cache[1].shape[-1]) \
                        .index_select(0, prefill_metadata.kv_index_list[iter]).contiguous()
                prefill_kv_a = kv_a[:actual_seq_kvlen[-1]]
                prefill_k_pe = k_pe[:actual_seq_kvlen[-1]]
 
                kv = self.kv_b_proj.forward(prefill_kv_a)[0]
                kv = kv.view(-1, self.num_local_heads, self.qk_nope_head_dim + self.v_head_dim)
                k_nope, v = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
                if prefill_metadata.max_query_len > 1:
                    attn_mask = self.attn_mask
                else:
                    attn_mask = None
                prefill_k_rope = prefill_k_pe.view(-1, 1, self.qk_rope_head_dim).repeat(1, self.num_local_heads, 1)
                attn_output[computed_tokens:computed_tokens+actual_seq_qlen[-1]] = \
                    torch.ops.npu.npu_fused_infer_attention_score(
                        q_nope[computed_tokens:computed_tokens+actual_seq_qlen[-1]],
                        k_nope,
                        v,
                        query_rope=q_pe[computed_tokens:computed_tokens+actual_seq_qlen[-1]],
                        key_rope=prefill_k_rope,
                        num_heads=self.num_local_heads,
                        num_key_value_heads=self.num_local_heads,
                        input_layout="TND",
                        atten_mask=attn_mask,
                        sparse_mode=2,
                        actual_seq_lengths=actual_seq_qlen,
                        actual_seq_lengths_kv=actual_seq_kvlen,
                        scale=self.scaling,
                        next_tokens=0)[0]
                computed_tokens += actual_seq_qlen[-1]
        else:
            attn_output.fill_(0)
 
        if model_extra_config.operator_opt_config.prefill_enable_mla_alltoall:
            attn_output = attn_output.reshape(-1)
            all_to_all_attn_output = torch.empty([sum(prefill_metadata.seq_lens) * self.num_local_heads * self.qk_nope_head_dim], dtype=attn_output.dtype, device=current_platform.device_type)
            dist.all_to_all_single(all_to_all_attn_output, attn_output)  # (total_experts,) --> (total_ranks * n_routed_experts_per_rank)
            attn_output = all_to_all_attn_output.view(get_tensor_model_parallel_world_size(), sum(prefill_metadata.seq_lens) // get_tensor_model_parallel_world_size(), self.num_local_heads * self.qk_nope_head_dim).transpose(0, 1).contiguous()
            output, _ = self.o_proj.forward(attn_output.reshape(sum(prefill_metadata.seq_lens) // get_tensor_model_parallel_world_size(), -1))
        else:
            attn_output = attn_output.view(-1, self.num_local_heads * self.v_head_dim)
            output = self.o_proj.forward(attn_output)[0]
        return output
 
    def _forward_decode(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: Optional[torch.Tensor],
        attn_metadata: AttentionMetadata,
        use_rmsnorm_rope_cache: bool = True
    ) -> torch.Tensor:
        if use_rmsnorm_rope_cache:
            hidden_states = tensor_model_parallel_all_gather(hidden_states, dim=0)
            key_cache, value_cache = kv_cache
 
            q_len = 1
            if model_extra_config.operator_opt_config.use_mlaprolog:
                block_num, block_size, head_size, _ = key_cache.shape
                bsz, _ = hidden_states.view(-1, 7168).shape
                hidden_states_int8, pertoken_scale = torch_npu.npu_dynamic_quant(hidden_states)
                cos, sin = attn_metadata.decode.cos, attn_metadata.decode.sin
                cache_index = attn_metadata.slot_mapping.view(bsz, -1)
 
                q_nope, q_pe, k_nope, k_rope, dequant_scale_q_nope = torch.ops.npu.npu_mla_prolog_v2(token_x = hidden_states_int8.view(bsz, 1, -1),
                    weight_dq=self.q_a_proj.weight, weight_uq_qr=self.q_b_proj.weight,
                    weight_uk=self.W_UK, weight_dkv_kr=self.kv_a_proj_with_mqa.weight,
                    rmsnorm_gamma_cq=self.q_a_layernorm.weight, rmsnorm_gamma_ckv=self.kv_a_layernorm.weight,
                    rope_sin=sin.squeeze(1), rope_cos=cos.squeeze(1), cache_index=cache_index, 
                    kv_cache=key_cache.view(-1, 128, 1, 512), kr_cache=value_cache.view(-1, 128, 1, 64),
                    dequant_scale_x=pertoken_scale.view(-1, 1), # pertoken quant
                    dequant_scale_w_dq=self.q_a_proj.weight_scale.view(1, -1),
                    dequant_scale_w_uq_qr=self.q_b_proj.weight_scale.view(1, -1),
                    dequant_scale_w_dkv_kr=self.kv_a_proj_with_mqa.weight_scale.view(1, -1),
                    quant_scale_ckv=torch.reciprocal(self.kv_scale).repeat(self.kv_lora_rank).view(1, -1) if self.use_faquant else None,
                    quant_scale_ckr=None,
                    smooth_scales_cq=None,
                    rmsnorm_epsilon_cq=self.q_a_layernorm.variance_epsilon,
                    rmsnorm_epsilon_ckv=self.kv_a_layernorm.variance_epsilon,
                    cache_mode = "PA_NZ")
 
                k_nope = k_nope.view(block_num, 1, self.kv_lora_rank // (32 if self.use_faquant else 16), block_size, (32 if self.use_faquant else 16))
                k_rope = k_rope.view(block_num, 1, self.qk_rope_head_dim // 16, block_size, 16)
                q_nope = q_nope.view(bsz, self.num_local_heads, self.kv_lora_rank)
                q_pe = q_pe.view(bsz, self.num_local_heads, -1)
            else:
                if self.q_lora_rank is not None:
                    q_lowrank = self.q_a_proj(hidden_states)[0]
                else:
                    q_lowrank = self.q_proj(hidden_states)[0]
 
                if model_extra_config.operator_opt_config.moe_multi_stream_tune:
                    with tng.scope.npu_stream_switch('11'):
                        kv = self.kv_a_proj_with_mqa(hidden_states)[0]
 
                    tng.scope.npu_wait_tensor(q_lowrank, q_lowrank)
                else:
                    kv = self.kv_a_proj_with_mqa(hidden_states)[0]
                    
                if self.q_lora_rank is not None:
                    q, _ = self.q_a_layernorm(q_lowrank, self.norm_res[q_lowrank.shape[0]])
                    q = self.q_b_proj(q)[0]
                else:
                    q = q_lowrank
                bsz, _ = q.shape
                q = q.view(bsz, self.num_local_heads, 1, self.qk_head_dim)
                q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1) # b,n,s,d
 
                q_nope = q_nope.view(-1, self.num_local_heads, self.qk_nope_head_dim).transpose(0, 1) # n, bs, d
                q_nope = (
                    torch.matmul(q_nope, self.W_UK)
                    .transpose(1, 0)
                    .view(bsz, q_len, self.num_local_heads, -1)
                )
 
                if model_extra_config.operator_opt_config.moe_multi_stream_tune:
                    with tng.scope.npu_stream_switch('11'):
                        kv = kv.unsqueeze(1).unsqueeze(1)
                        cos, sin = attn_metadata.decode.cos, attn_metadata.decode.sin
                        # cos, sin = self.rotary_emb.get_cos_sin(positions)
                        tmp_slot_mapping = attn_metadata.slot_mapping
                        block_num, block_size, head_size, _ = key_cache.shape
                        k_rope, k_nope, _, _ = torch_npu.npu_kv_rmsnorm_rope_cache(
                            kv, self.kv_a_layernorm.weight,
                            cos, sin, tmp_slot_mapping,
                            value_cache, key_cache,
                            epsilon=self.kv_a_layernorm.variance_epsilon, cache_mode="PA_NZ") # adapter NZ
 
                        # adapter nz
                        k_nope = k_nope.view(block_num, 1, self.kv_lora_rank // KVCACHE_NZ_DIM, block_size, KVCACHE_NZ_DIM)
                        k_rope = k_rope.view(block_num, 1, self.qk_rope_head_dim // KVCACHE_NZ_DIM, block_size, KVCACHE_NZ_DIM)
 
                        tng.scope.npu_wait_tensor(q_pe, k_nope)
 
                        # cos, sin = self.rotary_emb.get_cos_sin(positions)
                        q_pe = torch_npu.npu_interleave_rope(q_pe, cos, sin) # BNSD
                        q_nope = q_nope.view(bsz, self.num_local_heads, self.kv_lora_rank)
                        q_pe = q_pe.view(bsz, self.num_local_heads, -1)
                else:
                    kv = kv.unsqueeze(1).unsqueeze(1)
                    cos, sin = attn_metadata.decode.cos, attn_metadata.decode.sin
                    # cos, sin = self.rotary_emb.get_cos_sin(positions)
                    tmp_slot_mapping = attn_metadata.slot_mapping
                    block_num, block_size, head_size, _ = key_cache.shape
                    k_rope, k_nope, _, _ = torch_npu.npu_kv_rmsnorm_rope_cache(
                        kv, self.kv_a_layernorm.weight,
                        cos, sin, tmp_slot_mapping,
                        value_cache, key_cache,
                        epsilon=self.kv_a_layernorm.variance_epsilon, cache_mode="PA_NZ") # adapter NZ
 
                    # adapter nz
                    k_nope = k_nope.view(block_num, 1, self.kv_lora_rank // KVCACHE_NZ_DIM, block_size, KVCACHE_NZ_DIM)
                    k_rope = k_rope.view(block_num, 1, self.qk_rope_head_dim // KVCACHE_NZ_DIM, block_size, KVCACHE_NZ_DIM)
 
                    # cos, sin = self.rotary_emb.get_cos_sin(positions)
                    q_pe = torch_npu.npu_interleave_rope(q_pe, cos, sin) # BNSD
                    q_nope = q_nope.view(bsz, self.num_local_heads, self.kv_lora_rank)
                    q_pe = q_pe.view(bsz, self.num_local_heads, -1)
 
            bsz, _, q_dim = q_nope.size()
            input_layout = "TND_NTD" if model_extra_config.operator_opt_config.use_a3_high_performance_cann else "TND"
            # FIA super kernel wait for support
            if False and model_extra_config.operator_opt_config.use_super_kernel:
                with tng.scope.super_kernel(self.prefix, 'option_xxx'):
                    attn_output, _ = torch.ops.npu.npu_fused_infer_attention_score(
                            q_nope, k_nope, k_nope, query_rope=q_pe, key_rope=k_rope,
                            num_heads=self.num_local_heads,
                            num_key_value_heads=1, input_layout=input_layout,
                            scale=self.scaling,
                            antiquant_mode=0, antiquant_scale=None,
                            block_table=attn_metadata.decode.block_table,
                            block_size=128,
                            actual_seq_lengths=torch.arange(1, bsz + 1, dtype=attn_metadata.decode.seq_lens.dtype, device=attn_metadata.decode.seq_lens.device),
                            actual_seq_lengths_kv=attn_metadata.decode.seq_lens,
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
            else:
                attn_output, _ = torch.ops.npu.npu_fused_infer_attention_score(
                        q_nope, k_nope, k_nope, query_rope=q_pe, key_rope=k_rope,
                        num_heads=self.num_local_heads,
                        num_key_value_heads=1, input_layout=input_layout,
                        scale=self.scaling,
                        antiquant_mode=0, antiquant_scale=None,
                        block_table=attn_metadata.decode.block_table,
                        block_size=128,
                        actual_seq_lengths=self.actual_seq_lengths[bsz],
                        actual_seq_lengths_kv=attn_metadata.decode.seq_lens,
                        )
 
                # Apply UV, (N, B, L) @ W_UV (N, L, V) -> (N, B, V)
                if model_extra_config.operator_opt_config.use_a3_high_performance_cann:
                    attn_output = attn_output.view(self.num_local_heads, bsz*q_len, self.kv_lora_rank) # adapter BSND_NBSD
                else:
                    attn_output = attn_output.squeeze(1).transpose(0, 1)
                # attn_output = pp_matmul(attn_output, self.W_UV, mm_type=4)
                attn_output = (
                    torch.matmul(attn_output, self.W_UV)
                    .transpose(1, 0)
                    .reshape(bsz, q_len, -1)
                )
                attn_output = attn_output.view(
                    -1, self.num_local_heads * self.v_head_dim)
                output, _ = self.o_proj.forward(attn_output)
        else:
            hidden_states = tensor_model_parallel_all_gather(hidden_states, dim=0)
            key_cache, value_cache = kv_cache
            output_dtype = key_cache.dtype
 
            if self.q_lora_rank is not None:
                q_lowrank = self.q_a_proj(hidden_states)[0]
            else:
                q_lowrank = self.q_proj(hidden_states)[0]
 
            kv = hidden_states
            kv = self.kv_a_proj_with_mqa(kv)[0]
 
            if self.q_lora_rank is not None:
                q = self.q_a_layernorm(q_lowrank)
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
 
            kv = kv.unsqueeze(1).unsqueeze(1)
            cos, sin = attn_metadata.decode.cos, attn_metadata.decode.sin
            # cos, sin = self.rotary_emb.get_cos_sin(positions)
            tmp_slot_mapping = attn_metadata.slot_mapping
            block_num, block_size, head_size, _ = key_cache.shape
            k_rope, k_nope, _, _ = torch_npu.npu_kv_rmsnorm_rope_cache(
                kv, self.kv_a_layernorm.weight,
                cos, sin, tmp_slot_mapping,
                value_cache, key_cache,
                epsilon=self.kv_a_layernorm.variance_epsilon, cache_mode="PA_NZ") # adapter NZ
 
            # adapter nz
            k_nope = k_nope.view(block_num, 1, self.kv_lora_rank // KVCACHE_NZ_DIM, block_size, KVCACHE_NZ_DIM)
            k_rope = k_rope.view(block_num, 1, self.qk_rope_head_dim // KVCACHE_NZ_DIM, block_size, KVCACHE_NZ_DIM)
 
            # cos, sin = self.rotary_emb.get_cos_sin(positions)
            q_pe = torch_npu.npu_interleave_rope(q_pe, cos, sin) # BNSD
            q_nope = q_nope.view(bsz, 1, self.num_local_heads, self.kv_lora_rank)
            q_pe = q_pe.view(bsz, 1, self.num_local_heads, -1)
 
            bsz, q_len, _, q_dim = q_nope.size()
            attn_output, _ = torch.ops.npu.npu_fused_infer_attention_score(
                    q_nope, k_nope, k_nope, query_rope=q_pe, key_rope=k_rope,
                    num_heads=self.num_local_heads,
                    num_key_value_heads=1, input_layout="BSND",
                    scale=self.scaling,
                    antiquant_mode=0, antiquant_scale=None,
                    block_table=attn_metadata.decode.block_table,
                    block_size=128,
                    actual_seq_lengths_kv=attn_metadata.decode.seq_lens,
                    )
 
            # Apply UV, (N, B, L) @ W_UV (N, L, V) -> (N, B, V)
            attn_output = attn_output.squeeze(1).transpose(0, 1)
            # attn_output = attn_output.view(self.num_local_heads, bsz*q_len, self.kv_lora_rank) # adapter BSND_NBSD
            # attn_output = pp_matmul(attn_output, self.W_UV, mm_type=4)
            attn_output = (
                torch.matmul(attn_output, self.W_UV)
                .transpose(1, 0)
                .reshape(bsz, q_len, -1)
            )
            attn_output = attn_output.view(
                -1, self.num_local_heads * self.v_head_dim)
            output, _ = self.o_proj.forward(attn_output)
        return output