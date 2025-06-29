# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

import os
import numpy as np
import torch
from pathlib import Path
from typing import Optional, Tuple
from omni_planner import omni_placement

class ExpertMapping:
    def __init__(self, pattern_path, device: str = "npu", rank: int = 0, num_devices_per_host: int = 8, max_redundants_per_expert: int = 20):
        self.pattern_path = pattern_path
        self.device = device
        self.num_devices_per_host = num_devices_per_host
        self.placement_pattern = self._load_placement_pattern_with_validation()

        max_redundants_per_expert = self.get_max_redundant_expert_num()
        if self.placement_pattern is not None:
            num_divices, num_layers, num_eps = self.placement_pattern.shape
            self.local_expert_mapping = torch.zeros(num_layers,
                                                    num_eps,
                                                    dtype = torch.int32,
                                                    device = self.device)
            self.local_expert_offsets = self._calc_expert_offset_each_layer()

            self.global_expert_mapping = torch.zeros(num_layers,
                                                    num_eps,
                                                    max_redundants_per_expert, # max_redundants_per_expert
                                                    dtype = torch.int32,
                                                    device = self.device)
            self.redundant_count_per_expert = torch.zeros(num_layers,
                                                    num_eps,
                                                    dtype = torch.int32,
                                                    device = self.device)

            self.redundant_expert_mapping = torch.zeros(num_layers,
                                                    max_redundants_per_expert,
                                                    num_eps,
                                                    dtype = torch.int32,
                                                    device = self.device)

            self.placement_pattern_cpu = self.placement_pattern.cpu()
            self.placement_mapping = omni_placement.PlacementMapping("",  # TODO: pattern path, parse pattern in native C++
                                                                     rank,
                                                                     num_devices_per_host,
                                                                     self.redundant_expert_mapping.data_ptr(),
                                                                     list(self.redundant_expert_mapping.size()),
                                                                     self.global_expert_mapping.data_ptr(),
                                                                     list(self.global_expert_mapping.size()),
                                                                     self.redundant_count_per_expert.data_ptr(),
                                                                     list(self.redundant_count_per_expert.size()),
                                                                     self.placement_pattern_cpu.data_ptr(),
                                                                     list(self.placement_pattern_cpu.size()))

    def _resolve_pattern_path(self) -> Optional[Path]:
        """Resolve placement pattern path from configuration."""
        raw_path = self.pattern_path
        if not raw_path or raw_path == "":
            return None
        return self._convert_pattern_path(raw_path)

    def _convert_pattern_path(self, path: str) -> str:
        # Check if the path is a relative path
        if not os.path.isabs(path):
            # Get the absolute path of the current script file
            current_dir = os.path.dirname(os.path.abspath(__file__))
            # Create a Path object and resolve the relative path
            absolute_path = (Path(current_dir) / path).resolve()
            return str(absolute_path)
        else:
            # If it's already an absolute path, return it directly
            return path

    def _load_placement_pattern_with_validation(self) -> Optional[torch.Tensor]:
        """Load and validate placement pattern from config."""
        pattern_path = self._resolve_pattern_path()
        if not pattern_path:
            return None
        if not os.path.exists(pattern_path):
            raise FileNotFoundError(f"Placement pattern file not found: {pattern_path}")
        try:
            pattern = torch.tensor(
                np.load(pattern_path).astype(np.int32),
                dtype=torch.int32,
                device=self.device
            )
            # Validate pattern shape against num_devices_per_host
            if pattern.shape[0] % self.num_devices_per_host != 0:
                print(f"Warning: Number of devices in pattern ({pattern.shape[0]}) is not "
                      f"evenly divisible by num_devices_per_host ({self.num_devices_per_host})")
            return pattern
        except Exception as e:
            raise RuntimeError(f"Error loading placement pattern: {e}") from e

    # @calculate_time
    def is_expert_on_current_rank(
        self,
        layer_idx_moe: int,
        expert_id: int,
        current_rank: int,
        experts_per_rank: int
    ) -> Tuple[bool, int]:
        """
        Check if expert is deployed on current rank and get its position.

        Args:
            layer_idx_moe: ID of the MoE layer
            expert_id: Expert ID within the layer
            current_rank: Target device rank to check
            experts_per_rank: Experts per device in default deployment

        Returns:
            Tuple (exists_on_rank, local_position)
        """
        if layer_idx_moe > 57:
            return self._default_deployment_check(expert_id, current_rank, experts_per_rank)
        if self.placement_pattern is None:
            return self._default_deployment_check(expert_id, current_rank, experts_per_rank)

        layer_mapping = self.placement_pattern[current_rank, layer_idx_moe]
        exists = layer_mapping[expert_id] > 0.5
        position = int(torch.sum(layer_mapping[:expert_id]).item())
        return exists, position

    def _default_deployment_check(
        self,
        expert_id: int,
        current_rank: int,
        experts_per_rank: int
    ) -> Tuple[bool, int]:
        """Check default sequential expert deployment."""
        start = current_rank * experts_per_rank
        end = (current_rank + 1) * experts_per_rank
        in_range = start <= expert_id < end
        position = expert_id - start if in_range else -1
        return in_range, position

    def _apply_local_expert_mapping(
        self,
        layer_idx_moe: Optional[int] = None,
        token_expert_ids: Optional[torch.Tensor] = None
    ) -> Optional[torch.Tensor]:
        return self.local_expert_mapping[layer_idx_moe, token_expert_ids]

    def _none_local_expert_mapping(
        self,
        layer_idx_moe: Optional[int] = None,
        token_expert_ids: Optional[torch.Tensor] = None
    ) -> Optional[torch.Tensor]:
        return token_expert_ids

    def get_num_of_redundant_experts(self, moe_layer_idx: int, num_expert_per_device_origin=16, rank_device=0) -> int:
        """
        Calculate the number of redundant experts for a specific device and MoE layer.

        Args:
            moe_layer_idx : int
                Index of the MoE layer to query expert distribution.
            num_expert_per_device_origin : int, optional (default=16)
                Original number of experts assigned to this device/layer.
            rank_device : int, optional (default=0)
                Rank identifier of the target device in the distributed system.

        Returns:
            int
                Number of redundant experts, calculated as: (current experts count) - (original experts count).
        """
        if self.placement_pattern is None:
            return 0

        experts_here = self.placement_pattern[rank_device][moe_layer_idx]
        num_redundant_experts = round(torch.sum(experts_here).item() - num_expert_per_device_origin)
        return num_redundant_experts

    def get_world_size(self) -> int:
        return self.placement_pattern.shape[0]

    def get_total_num_expert(self) -> int:
        num_expert = self.placement_pattern.shape[2]
        return num_expert

    def get_total_num_layers(self) -> int:
        num_layers = self.placement_pattern.shape[1]
        return num_layers

    def get_total_deployed_experts(self) -> int:
        total_deployed_experts = int(torch.sum(self.placement_pattern[:, 0, :]).item())
        return total_deployed_experts

    def get_deployed_experts_per_layer(self) -> list:
        deployed_experts_per_layer = torch.sum(self.placement_pattern, dim=(0, 2)).tolist()
        return deployed_experts_per_layer

    def get_redundant_enable_per_layer(self) -> list:
        deployed_experts_per_layer = self.get_deployed_experts_per_layer()
        num_logits_expert_per_rank = self.get_total_num_expert()
        redundant_enable_per_layer = [not (value==num_logits_expert_per_rank) for value in deployed_experts_per_layer]
        return redundant_enable_per_layer

    def _calc_expert_offset_each_layer(self) :
        """
        初始化时预计算每个 layer 中每个 rank 的expert offset。
        """
        # 计算每个 rank 在每个 layer 中的专家数，形状：(rank, layer)
        rank_expert_counts = self.placement_pattern.sum(dim=-1)  # 沿 expert 轴求和

        # 计算累积和，沿 rank 轴，形状仍为 (rank, layer)
        # cumsum[i, j] 表示第 j 层中，前 i个 rank 的专家总数
        cumsum_experts = torch.cumsum(rank_expert_counts, dim=0)

        # offset[i, j] 表示第 j 层中，第 i 个 rank 前的专家总数
        # 用零填充第 0 个 rank 的 offset，并取 cumsum 的前一行
        local_expert_offsets = torch.zeros_like(cumsum_experts)
        local_expert_offsets[1:] = cumsum_experts[:-1]  # 第 i 个 rank 的 offset 是前 i-1 个 rank 的和
        return local_expert_offsets

    def get_local_expert_indices_offset(self, layer_idx_moe: int, current_rank: int, default_experts_per_rank: int) -> int:
        if self.placement_pattern is None:
            return current_rank * default_experts_per_rank

        return self.local_expert_offsets[current_rank, layer_idx_moe].item()

    def get_max_redundant_expert_num(self) :
        if self.placement_pattern is None:
            return 1 #only one deployment each expert

        pattern = self.placement_pattern.to(dtype=torch.int64)
        redundant_expert_num = pattern.sum(dim=0)
        max_redundant_expert_num = redundant_expert_num.max()
        return max_redundant_expert_num

    def get_default_placement_layers(self):
        """
        Vectorized check for whether each layer satisfies the default placement requirements.

        Returns:
            list[bool]: A list of length num_layers, where True indicates the layer satisfies
                        the default placement, and False indicates it does not.
        """
        placement_pattern = self.placement_pattern
        world_size, num_layers, num_experts = placement_pattern.shape

        # 创建专家 ID 和预期 rank
        expert_ids = torch.arange(num_experts, dtype=torch.long, device=placement_pattern.device)
        num_experts_per_rank = num_experts // world_size
        expected_ranks = expert_ids // num_experts_per_rank

        # 检查预期 rank 是否在 world_size 范围内
        if torch.any(expected_ranks >= world_size):
            raise RuntimeError(f"Some experts require ranks beyond world_size {world_size}")

        # 存储每层的布尔结果
        valid_layers = []

        # 为每一层检查放置情况
        for layer in range(num_layers):
            # 提取当前层的 placement
            layer_placement = placement_pattern[:, layer, :]

            # 检查每个专家在预期 rank 上的值是否为 1
            valid_expected = layer_placement[expected_ranks, expert_ids] == 1

            # 检查每个专家在非预期 rank 上的值是否为 0
            valid_non_expected = torch.ones(num_experts, dtype=torch.bool, device=placement_pattern.device)
            for rank in range(world_size):
                # 跳过预期 rank
                mask = expected_ranks != rank
                if mask.any():
                    # 检查非预期 rank 上的值是否为 0
                    valid_non_expected[mask] &= (layer_placement[rank, expert_ids[mask]] == 0)

            # 该层有效当且仅当预期 rank 值为 1 且非预期 rank 值为 0
            valid_layers.append(torch.all(valid_expected & valid_non_expected).item())

        return valid_layers