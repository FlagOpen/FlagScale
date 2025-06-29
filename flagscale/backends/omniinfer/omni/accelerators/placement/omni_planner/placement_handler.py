# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

import torch
import ctypes
from omni_planner import omni_placement
from collections import defaultdict
from omni_planner.utils import filter_dict_keys,convert_param_dict_to_list,convert_param_to_ctype,get_expert_ids,calculate_time
import re

def deepseek_filter_func(key, first_k_dense_replace=3):
    NAMES = ("w13_weight", "w2_weight", "w2_weight_offset", "w2_weight_scale", "w13_weight_offset", "w13_weight_scale")
    pattern = r"^.*\.layers\.(\d+)\..*\.(.+)$"
    match = re.match(pattern, key)

    if match:
        layer = int(match.group(1))  # 提取layer数字
        name = match.group(2)        # 提取name
        return layer >= first_k_dense_replace and name in NAMES
    return False

def deepseek_get_layer_idx_func(key,first_k_dense_replace=3):
    pattern = r"^.*\.layers\.(\d+)\..*\.(.+)$"
    match = re.match(pattern, key)
    if not match:
        raise RuntimeError(f"current key is {key},  layer.layer_idx ")
    layer = int(match.group(1))  # 提取layer数字
    return layer - first_k_dense_replace

@calculate_time
def init_dram_weights(moe_weights,param_dict,local_rank_pattern,first_k_dense_replace):
    """
    Args:
        moeweights: omni_placement.MoEWeights
        param_dict: 权重信息 Dict(name: torch.Tensor)
        local_rank_pattern (torch.Tensor): pattern, dtype:bool, shape: [num_layers, num_experts]
    """
    # Type checking
    if not isinstance(moe_weights, omni_placement.MoEWeights):
        raise TypeError("moe_weights must be an instance of omni_placement.MoEWeights")
    if not isinstance(param_dict, dict):
        raise TypeError("param_dict must be a dictionary")
    if not isinstance(local_rank_pattern, torch.Tensor):
        raise TypeError("local_rank_pattern must be a torch.Tensor")

    # Validate local_rank_pattern
    if local_rank_pattern.dtype != torch.bool:
        raise ValueError("local_rank_pattern must have dtype torch.bool")
    if local_rank_pattern.dim() != 2:
        raise ValueError("local_rank_pattern must be a 2D tensor with shape [num_layers, num_experts]")

    filter_func_params = {"first_k_dense_replace":first_k_dense_replace}
    param_dict = filter_dict_keys(param_dict,deepseek_filter_func,filter_func_params) # 传入过滤函数， 过滤出专家权重
    get_layer_func_params = {"first_k_dense_replace":first_k_dense_replace}
    param_list = convert_param_dict_to_list(param_dict,deepseek_get_layer_idx_func,get_layer_func_params) # 传入layer识别函数， 权重从 Dict转化为list
    ctype_param_list = convert_param_to_ctype(param_list) # 取权重地址，转化为c++接收类型

    # 临时取
    experts_id_list = get_expert_ids(local_rank_pattern) # expert_idx -> expert_ids 转化，

    # 调用C++端的init_weights方法
    moe_weights.init_weights(ctype_param_list,experts_id_list)


def create_placement_manager(rank, world_size, num_devices_per_host, cluster_activation=None, expert_mapping=None):
    """
    Creates a Placement manage.

    Args:
        rank (int): Rank of the current process in the distributed system.
        world_size (int): Total number of processes in the distributed system.
        num_devices_per_host (int): Number of devices per host machine.
        cluster_activation (optional): Cluster activation object; defaults to None.
        expert_mapping (optional): Expert mapping object containing placement data; defaults to None.

    Returns:
        omni_placement.Placement: A Placement object managing MoE expert placement.
    """
    # Map torch.dtype to c10::ScalarType integer values
    dtype_map = {
        torch.int32: 3,  # c10::ScalarType::Int
        # Add more types as needed
    }
    scalar_type = dtype_map[torch.int32]

    local_expert_mapping = expert_mapping.local_expert_mapping
    local_expert_shape = list(local_expert_mapping.size())
    placement_pattern = expert_mapping.placement_pattern.cpu()
    placement_shape = list(placement_pattern.size())

    # Instantiate Placement object
    placement = omni_placement.Placement(
        rank,
        world_size,
        num_devices_per_host,
        cluster_activation,
        local_expert_mapping.data_ptr(),
        local_expert_shape,
        scalar_type,
        placement_pattern.data_ptr(),
        placement_shape,
        scalar_type
    )
    return placement


def create_cluster_activation(rank, world_size, num_layers,total_deployed_experts, count_activation):
    """
    Creates a ClusterActivation object for managing cluster-level activations.

    Args:
        rank (int): Rank of the current process in the distributed system.
        world_size (int): Total number of processes in the distributed system.
        expert_mapping: Expert mapping object providing layer and expert information.
        count_activation (torch.Tensor): Tensor containing activation count data.

    Returns:
        omni_placement.ClusterActivation: A ClusterActivation object for tracking activations.
    """
    # Extract shape information from expert_mapping
    activation_window_size = 10  # Default activation window size

    length = count_activation.numel()
    element_size = count_activation.element_size()
    address = count_activation.data_ptr()

    tensor = omni_placement.Tensor(
        data_ptr=address,
        length=length,
        element_size=element_size,
        name="",
    )

    # Instantiate ClusterActivation object
    cluster_activation = omni_placement.ClusterActivation(
        tensor,
        num_layers,
        total_deployed_experts,
        activation_window_size,
        world_size,
        rank
    )
    return cluster_activation
