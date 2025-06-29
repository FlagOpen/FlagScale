# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

from dataclasses import dataclass
import torch

from vllm.logger import init_logger
from vllm.utils import cdiv
from vllm.config import VllmConfig
from vllm.v1.kv_cache_interface import (
    KVCacheSpec,
    AttentionSpec,
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheTensor,
)
from vllm.v1.core.kv_cache_utils import create_kv_cache_group_specs
from vllm.v1.worker.block_table import BlockTable, MultiGroupBlockTable


logger = init_logger("vllm.v1.omni")

SINK = 128
RECENT = 256
BETA = 0.2


@dataclass
class OmniKVCacheConfig(KVCacheConfig):
    """
    The KV cache configuration of a model with different block numbers
    for different KV cache groups.
    """

    num_blocks_per_group: dict[type[KVCacheSpec], int]
    """The number of KV cache blocks per kv cache group"""


@dataclass
class OmniAttentionSpec(AttentionSpec):
    sink: int = SINK
    recent: int = RECENT

    def __post_init__(self):
        if self.sink % self.block_size != 0 or self.recent % self.block_size != 0:
            raise ValueError("Sink and recent values should be divisible by block_size.")
        self.max_compressed_len = self.sink + self.recent
        self.max_num_blocks = self.max_compressed_len // self.block_size

    @property
    def type_id(self) -> str:
        return f"omni_attention_{self.sink}_{self.recent}_{self.block_size}_{self.page_size_bytes}"

    @property
    def max_memory_usage_bytes(self, vllm_config: VllmConfig) -> int:
        return self.max_num_blocks * self.page_size_bytes


class OmniMultiGroupBlockTable(MultiGroupBlockTable):
    def __init__(self, max_num_reqs: int, max_model_len: int,
                 max_num_batched_tokens: int, pin_memory: bool,
                 device: torch.device, kv_cache_config: OmniKVCacheConfig) -> None:
        max_num_blocks_per_req = [
            cdiv(max_model_len, g.kv_cache_spec.block_size)
            if not isinstance(g.kv_cache_spec, OmniAttentionSpec) else g.kv_cache_spec.max_num_blocks
            for g in kv_cache_config.kv_cache_groups
        ]

        self.block_tables = [
            BlockTable(max_num_reqs, max_num_blocks_per_req[i],
                       max_num_batched_tokens, pin_memory, device)
            for i in range(len(kv_cache_config.kv_cache_groups))
        ]


def _get_kv_cache_config_omni_type(vllm_config: VllmConfig,
                                   kv_cache_spec: dict[str, KVCacheSpec],
                                   available_memory: int) -> OmniKVCacheConfig:
    """
    Generates the KV cache configuration for a model with two types of KV cache.
    It's assumed that the numbers of layers with these two types are approximately same.
    The ratio of memory allocated to them is now determined by a hyperparameter BETA.

    Args:
        vllm_config: The global VllmConfig
        kv_cache_spec: The kv cache spec of each attention layer in the model
        available_memory: Memory available for KV cache in bytes.

    Returns:
        The generated KVCacheConfig
    """
    page_sizes = {layer.page_size_bytes for layer in kv_cache_spec.values()}
    if len(page_sizes) != 1:
        raise ValueError("page_sizes must have exactly one element.")
    page_size = page_sizes.pop()

    # the original number of blocks if layers were uniform
    num_blocks = int(available_memory // page_size // len(kv_cache_spec))
    num_blocks = max(num_blocks, 0)

    # Here we implicitly assume the numbers of full and swa layers are close,
    # since the sum of one full layer and one swa layer equals to two original layers.
    omni_num_blocks = int(num_blocks * BETA)
    full_num_blocks = 2 * num_blocks - omni_num_blocks

    # logging
    num_tokens = num_blocks * vllm_config.cache_config.block_size
    full_num_tokens = full_num_blocks * vllm_config.cache_config.block_size
    omni_num_tokens = omni_num_blocks * vllm_config.cache_config.block_size
    logger.info(f"GPU KV cache size: {num_tokens:,} tokens")
    logger.info(f"With Omni enabled, GPU KV Cache size: {full_num_tokens:,}"
                f" tokens in full layers and {omni_num_tokens:,} in swa layers.")
    max_model_len_str = f"{vllm_config.model_config.max_model_len:,}"
    max_concurrency = num_tokens / vllm_config.model_config.max_model_len
    omni_max_concur = min(full_num_tokens / vllm_config.model_config.max_model_len,
                          omni_num_tokens / (SINK + RECENT))
    logger.info("Maximum concurrency for %s tokens per request changes from %.2fx to %.2fx",
                max_model_len_str, max_concurrency, omni_max_concur)

    # create a KVCacheConfig with exactly two groups
    # 1. divide the layers to full and omni groups
    # 2. compute size for each layer
    grouped_layer_names = [[], []]
    layer2size = {}
    for layer_name, layer_spec in kv_cache_spec.items():
        if isinstance(layer_spec, FullAttentionSpec):
            if layer_spec.sliding_window is not None:
                ValueError("Omni attention implements its own sliding window. \
                    Manually setting sliding window is not supported.")
            grouped_layer_names[0].append(layer_name)
            layer2size[layer_name] = KVCacheTensor(size=page_size*full_num_blocks)
        elif isinstance(layer_spec, OmniAttentionSpec):
            grouped_layer_names[1].append(layer_name)
            layer2size[layer_name] = KVCacheTensor(size=page_size*omni_num_blocks)
        else:
            raise RuntimeError(f"Unsupported KV Cache Spec type {type(layer_spec)}.")
    kv_cache_config = OmniKVCacheConfig(
        num_blocks=full_num_blocks,
        num_blocks_per_group={FullAttentionSpec: full_num_blocks, OmniAttentionSpec: omni_num_blocks},
        tensors=layer2size,
        kv_cache_groups=create_kv_cache_group_specs(kv_cache_spec,
                                                    grouped_layer_names)
    )
    return kv_cache_config
