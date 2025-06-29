# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

import csv
import sys
import os
from pathlib import Path
from typing import Optional, Tuple, cast
import numpy as np
import torch
import torch_npu
import ctypes

from typing import Optional
from omni_planner.cluster_status import ClusterStatus
from omni_planner.placement_handler import create_cluster_activation, create_placement_manager, init_dram_weights
from omni_planner.optim.optimizers import Optimizer
from omni_planner.optim.optimizers_loader import _create_optimizers
from omni_planner.config import Config
from omni_planner.expert_mapping import ExpertMapping
from omni_planner.utils import calculate_time

import time

class OmniPlannerMeta(type):
    """Metaclass to implement singleton pattern for OmniPlanner."""
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]

    @classmethod
    def cleanup(cls):
        if cls in cls._instances and not cls._cleanup_called:
            cls._instances[cls].cleanup()
            del cls._instances[cls]
            cls._cleanup_called = True

class OmniPlanner(metaclass=OmniPlannerMeta):
    """
    Optimizes token-to-expert mapping using multiple optimizers.
    Manages expert deployment across distributed systems.

    Attributes:
        config: Configuration object for planner settings
        cluster_status: Cluster status monitor
        optimizers: List of optimization algorithms
        expert_mapping: Expert deployment pattern mapping
    """
    def __init__(self, config_file: str = "/etc/omni/config.yaml", device: str = "npu",
                 rank: int = 0, world_size: int = 16, num_devices_per_host: int = 8):
        """Initialize OmniPlanner with configuration and distributed settings.

        Args:
            config_file: Path to configuration YAML file
            device: Target device type (e.g., "npu", "cuda")
            rank: Process rank in distributed environment
            world_size: Total number of processes in distributed environment
            num_devices_per_host: Number of devices per host machine (default: 8)
        """
        # Load configuration
        self.config = Config(config_file)
        self.device = torch.device(device)

        # Initialize distributed settings with fallback
        self._init_distributed(rank, world_size, num_devices_per_host)

        # Load and validate placement pattern
        self.expert_mapping = ExpertMapping(self.config.pattern_path, self.device, self.rank, self.num_devices_per_host)
        self.total_deployed_experts = self.expert_mapping.get_total_deployed_experts()

        # Calculate max_num_redundant_expert
        self.max_num_deployed_expert_per_rank = max(max(self.get_deployed_experts_per_layer()) // self.world_size, 1)
        self.max_redundant_num = self.expert_mapping.get_max_redundant_expert_num()

        """Initialize cluster status and optimizers."""
        self.cluster_status = ClusterStatus(self.config, self.expert_mapping, self.rank)
        self.optimizers = _create_optimizers(self.config.Optimizers, self.cluster_status)
        self.optimizer = self.optimizers[0]

        # Initialize placement manager
        self._init_placement_manager()

        batch_size = getattr(self.config, 'max_batch_size', 256)
        top_k_count = getattr(self.config, 'max_top_k', 8)
        selector = torch.arange(batch_size, device=self.device) % self.max_redundant_num  # Shape: (batch_size,)
        self.selector = selector.view(batch_size, 1).expand(batch_size, top_k_count)  # Broadcast to (batch_size, top_k_count)

        self.enable_dump = getattr(self.config, 'enable_dump', False)

        # redundant_enable_per_layer, True is redundant layer, False is Origin Layer
        self.redundant_enable_per_layer = self.expert_mapping.get_redundant_enable_per_layer()
        self.num_logits_expert_per_rank = max(self.expert_mapping.get_total_num_expert()//self.world_size, 1)


        print("OmniPlanner successfully initialized.")

    @classmethod
    def cleanup(cls):
        if cls in cls._instances:
            del cls._instances[cls]

    def __del__(self):
        if hasattr(self, 'cluster_activation'):
            self.cluster_activation.stop_thread()
            del self.cluster_activation # 显示析构 C++对象
            time.sleep(1) # 等待C++对象析构以及其创建的线程被回收

    def _init_distributed(self, rank: int, world_size: int, num_devices_per_host: int) -> None:
        """Initialize distributed settings with fallback to provided values.

        Args:
            rank: Process rank in distributed environment
            world_size: Total number of processes in distributed environment
            num_devices_per_host: Number of devices per host machine
        """
        try:
            self.rank = torch.distributed.get_rank()
            self.world_size = torch.distributed.get_world_size()
        except RuntimeError:
            self.rank, self.world_size = rank, world_size

        self.num_devices_per_host = os.environ.get("ASCEND_RT_VISIBLE_DEVICES")  # omni_planner config file
        self.num_devices_per_host = len(self.num_devices_per_host.split(",")) if self.num_devices_per_host else num_devices_per_host
        # Validate that world_size is consistent with num_devices_per_host
        if self.world_size % self.num_devices_per_host != 0:
            print(f"Warning: world_size ({self.world_size}) is not evenly divisible by "
                  f"num_devices_per_host ({self.num_devices_per_host})")

    def _init_placement_manager(self) -> None:
        """Initialize placement handler, and activation tracking."""
        num_layers = self.expert_mapping.get_total_num_layers()
        self.npu_activation_count = torch.zeros(
            (num_layers, self.get_max_num_deployed_expert_per_rank()),
            device=self.device,
            dtype=torch.int64
        )
        torch.npu.synchronize() # 确保 npu_activatio_count在显存中已经完成初始化

        self.cluster_activation = create_cluster_activation(
            self.rank,
            self.world_size,
            self.expert_mapping.get_total_num_layers(),
            self.get_max_num_deployed_expert_per_rank(),
            self.npu_activation_count
        )

        # self.placement_manager = create_placement_manager(
        #     self.rank,
        #     self.world_size,
        #     self.num_devices_per_host,
        #     self.cluster_activation,
        #     self.cluster_status.expert_mapping
        # )

    # @calculate_time
    def is_expert_on_current_rank(
        self,
        layer_id: int,
        expert_id: int,
        current_rank: int,
        experts_per_rank: int
    ) -> Tuple[bool, int]:
        """
        Check if expert is deployed on current rank and get its position.

        Args:
            layer_id: ID of the MoE layer
            expert_id: Expert ID within the layer
            current_rank: Target device rank to check
            experts_per_rank: Experts per device in default deployment

        Returns:
            Tuple (exists_on_rank, local_position)
        """
        return self.expert_mapping.is_expert_on_current_rank(layer_id, expert_id, current_rank, experts_per_rank)

    def expert_mapping_on_current_layer(
        self,
        layer_idx_moe: torch.tensor,
        is_prefill=False) -> torch.tensor:
        if layer_idx_moe > 57:
            return None
        return self.cluster_status.expert_mapping.redundant_expert_mapping[layer_idx_moe]

    def plan(
        self,
        layer_idx_moe: Optional[int] = None,
        tokens: Optional[torch.Tensor] = None,
        token_expert_ids: Optional[torch.Tensor] = None,
        token_expert_scores: Optional[torch.Tensor] = None,
        top_k: int = 8,
        expert_mapping: Optional[torch.Tensor] = None,
        is_prefill=True
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Optimize token-to-expert mapping using configured optimizers.

        This method takes input tokens and their initially assigned experts and scores. It computes
        expert loads, updates the cluster status accordingly, and then optimizes the assignment
        of tokens to experts by applying the configured optimization strategies.

        Args:
            layer_idx_moe: Identifier for the current layer (optional)
            tokens: Input tokens tensor with shape [num_tokens, ...]
            token_expert_ids: Initial expert assignments, shape [num_tokens, top_k], -1 indicates unassigned
            token_expert_scores: Importance scores for expert assignments, shape [num_tokens, top_k]

        Returns:
            Tuple containing (original tokens, optimized expert IDs, optimized scores)
        """
        if layer_idx_moe > 57:
            return tokens, token_expert_ids, token_expert_scores
        # Input validation check
        # if tokens is None or token_expert_ids is None:
        #     return tokens, token_expert_ids, token_expert_scores
        if self.redundant_enable_per_layer[layer_idx_moe]:
            batch_size = token_expert_ids.shape[0]
            token_expert_ids = expert_mapping[self.selector[:batch_size, :top_k], token_expert_ids]
        else:
            token_expert_ids = expert_mapping[0][token_expert_ids]

        return tokens, token_expert_ids, token_expert_scores


    def _validate_input(
        self,
        tokens: torch.Tensor,
        expert_ids: torch.Tensor,
        scores: Optional[torch.Tensor] = None
    ) -> None:
        """Validate dimensional consistency of input parameters"""
        if expert_ids.ndim != 2:
            raise ValueError("token_expert_ids must be 2-dimensional")
        if scores is None:
            return
        if scores.shape != expert_ids.shape:
            raise ValueError("token_expert_scores must match the shape of token_expert_ids")
        for token_scores in scores:
            for score in token_scores:
                if score.dtype not in (torch.int, torch.float32) or score < 0:
                    raise ValueError("Scores must be non-negative numbers")

    def _compute_expert_loads(
        self,
        layer_idx_moe: Optional[int] = None,
        token_expert_ids: Optional[torch.Tensor] = None
    ) -> None:
        """
        Compute current load distribution across experts.
        Args:
            token_expert_ids (torch.Tensor): optimized expert assignments, shape [num_tokens, top_k], -1 indicates unassigned
        Returns:
            None
        """
        self.npu_activation_count[layer_idx_moe] += torch.histc(token_expert_ids,
                                            bins=self.total_deployed_experts, min=0, max=self.total_deployed_experts-1)


    @staticmethod
    def get_deepseek_v3_moe_layer_idx(prefix: str) -> int:
        """
        Calculate the adjusted DeepSeek-V3 MoE layer index from a model layer prefix.

        The function parses a prefix string of format `model.layers.{N}.mlp.experts` to extract the
        layer index `N`, then adjusts this index by subtracting a fixed offset of dense layers
        (FIRST_K_DENSE_REPLACE) as per the DeepSeek-V3 model configuration.

        Args:
            prefix: A layer path string formatted as `model.layers.{N}.mlp.experts`
                (e.g., "model.layers.5.mlp.experts" represents layer 5)

        Returns:
            int: The adjusted layer index after subtracting FIRST_K_DENSE_REPLACE.
                Formula: parsed_layer_id - FIRST_K_DENSE_REPLACE

        Note:
            - LAYER_ID_IDX (2): Indicates layer ID position after splitting the prefix by '.'
            (e.g., ["model", "layers", "5", "mlp", "experts"] -> index 2 is "5")
            - FIRST_K_DENSE_REPLACE (3): Number of initial dense layers from the model's config.json
            that should be excluded when working with MoE layers.

        Example:
            >>> get_deepseek_v3_moe_layer_idx("model.layers.5.mlp.experts")
            2   # 5 (parsed) - 3 (offset) = 2
        """
        # Parses prefix string like 'model.layers.3.mlp.experts'
        LAYER_ID_IDX = 2               # Position of the layer ID after splitting by '.'
        FIRST_K_DENSE_REPLACE = 3      # From config.json: initial dense layers count

        return int(prefix.split(sep='.')[LAYER_ID_IDX]) - FIRST_K_DENSE_REPLACE

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
        if moe_layer_idx > 57:
            return 0
        return self.expert_mapping.get_num_of_redundant_experts(moe_layer_idx, num_expert_per_device_origin, rank_device)

    def get_local_expert_indices_offset(self, layer_idx_moe: int, current_rank: int, default_experts_per_rank: int) -> int:
        return self.expert_mapping.get_local_expert_indices_offset(layer_idx_moe, current_rank, default_experts_per_rank)

    def get_deployed_experts_per_layer(self) -> list:
        return self.expert_mapping.get_deployed_experts_per_layer()

    def get_max_num_deployed_expert_per_rank(self)-> int:
        return self.max_num_deployed_expert_per_rank

    def init_dram_weights(self, param_dict, first_k_dense_replace=3):
        return
        moe_weights = self.placement_manager.get_moe_weights()
        local_rank_pattern = self.expert_mapping.placement_pattern[self.rank].bool()
        init_dram_weights(moe_weights, param_dict, local_rank_pattern, first_k_dense_replace)

    def dump(self,step):
        enable_dump = self.enable_dump
        dump_dir = getattr(self.config, 'dump_dir', None)
        if not enable_dump:
            if dump_dir is not None:
                print(f"Warning: dump_dir is setting to {dump_dir}, If You Want to Dump Experts Activation Pls set enable_dump to True")
            return
        if dump_dir is None:
            raise RuntimeError("dump_dir must not be None, Pls Set dump_dir")

        if step==0:
            self.cluster_activation.stopDump()
            if not hasattr(self, "prefill_count"):
                # 属性不存在，初始化为 0
                self.prefill_count  = 0
            if not hasattr(self, "last_npu_activation_count"):
                self.last_npu_activation_count = torch.zeros_like(self.npu_activation_count)

        if step==0 or step==1:
            self.prefill_count  += 1
            prefill_dump_dir = os.path.join(dump_dir, "prefill")
            os.makedirs(prefill_dump_dir, exist_ok=True)
            file_path = os.path.join(prefill_dump_dir, f"activation_counts_recordstep_{self.prefill_count}_rank_{self.rank}.txt")
            npu_activation_count = self.npu_activation_count-self.last_npu_activation_count
            # 打开文件进行写入
            with open(file_path, 'w') as f:
                # 遍历每一行
                for row in npu_activation_count:
                    # 将每一行的元素转换为字符串，用 \t 分隔
                    row_str = '\t'.join(str(x.item()) for x in row)
                    # 写入一行并添加 \n
                    f.write(row_str + '\n')

        elif step >=32:
            decoder_dump_dir = os.path.join(dump_dir, "decoder")
            os.makedirs(decoder_dump_dir, exist_ok=True)
            self.cluster_activation.setDumpDir(decoder_dump_dir)

        self.last_npu_activation_count = self.npu_activation_count.clone() # 置到最新的激活值


# Example usage
if __name__ == "__main__":
    from optimizer.ada_router_optimizer import AdaRouter
    from optimizer.token_balance_optimizer import TokenBalance

    # Example input: 3 tokens, 4 experts each, with importance scores
    input_token = torch.tensor([
        [0, 1, 2, 3],  # Token 1
        [1, 0, 3, 2],  # Token 2
        [3, 2, 1, 0]   # Token 3
    ], dtype=torch.float32).npu()

    input_expert_id = torch.tensor([
        [0, 1, 2, 3],  # Token 1 expert
        [1, 0, 3, 2],  # Token 2 expert
        [3, 2, 1, 0]   # Token 3 expert
    ], dtype=torch.long).npu()

    input_expert_score = torch.tensor([
        [0.9, 0.5, 0.3, 0.7],  # Token 1 expert score
        [0.4, 0.8, 0.6, 0.2],  # Token 2 expert score
        [0.7, 0.3, 0.9, 0.5]   # Token 3 expert score
    ], dtype=torch.float32).npu()

    planner = OmniPlanner("./config.yaml")

    token, token_expert_ids, token_scores = planner.plan(
        layer_id=0,
        tokens=input_token,
        token_expert_ids=input_expert_id,
        token_expert_scores=input_expert_score
    )

    print("Input mapping:")
    print(input_token, input_expert_id, input_expert_score)

    print("\nOptimized mapping:")
    print(token, token_expert_ids, token_scores)