# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

import torch

from vllm.distributed import (
    get_tp_group,
    get_ep_group,
)
from omni.adaptors.vllm.distributed.parallel_state import (
    get_local_world_group,
    get_world_group_from_list,
    get_local_group_from_list,
    get_cross_group_from_list,
    get_mlp_world_group,
)
from omni.models.common.config.model_config import model_extra_config


def expert_parallel_all_reduce(input_: torch.Tensor) -> torch.Tensor:
    """All-reduce the input tensor across model parallel group."""
    return get_ep_group().all_reduce(input_)

def expert_parallel_all_gather(input_: torch.Tensor,  dim=-1) -> torch.Tensor:
    """All-reduce the input tensor across model parallel group."""
    return get_ep_group().all_gather(input_,  dim)


def tensor_model_parallel_reduce_scatter(input_: torch.Tensor) -> torch.Tensor:
    """All-reduce the input tensor across model parallel group."""
    return get_tp_group().reduce_scatter(input_)


def reduce_scatter_two_stage(input_: torch.Tensor, idx: int, reverse=False) -> torch.Tensor:
    if model_extra_config.operator_opt_config.two_stage_comm == 0:
        return get_world_group_from_list(idx).reduce_scatter(input_)
    if reverse:
        stage1 = get_cross_group_from_list(idx).reduce_scatter(input_)
        return get_local_group_from_list(idx).reduce_scatter(stage1)
    stage1 = get_local_group_from_list(idx).reduce_scatter(input_)
    return get_cross_group_from_list(idx).reduce_scatter(stage1)


def all_gather_two_stage(input_: torch.Tensor, idx: int, dim=-1, reverse=False) -> torch.Tensor:
    if model_extra_config.operator_opt_config.two_stage_comm == 0:
        return get_world_group_from_list(idx).all_gather(input_, dim)
    if reverse:
        stage1 = get_local_group_from_list(idx).all_gather(input_, dim)
        return get_cross_group_from_list(idx).all_gather(stage1, dim)
    stage1 = get_cross_group_from_list(idx).all_gather(input_, dim)
    return get_local_group_from_list(idx).all_gather(stage1, dim)


def reduce_scatter_local(input_: torch.Tensor, idx: int) -> torch.Tensor:
    return get_local_group_from_list(idx).reduce_scatter(input_)


def reduce_scatter_cross(input_: torch.Tensor, idx: int) -> torch.Tensor:
    return get_cross_group_from_list(idx).reduce_scatter(input_)


def all_gather_local(input_: torch.Tensor, idx: int, dim=-1) -> torch.Tensor:
    return get_local_group_from_list(idx).all_gather(input_, dim)


def all_gather_cross(input_: torch.Tensor, idx: int, dim=-1) -> torch.Tensor:
    return get_cross_group_from_list(idx).all_gather(input_, dim)


def local_rank_all_gather(input_: torch.Tensor, dim=-1):
    return get_local_world_group().all_gather(input_, dim)


def mlp_all_gather(input_: torch.Tensor, dim=-1):
    return get_mlp_world_group().all_gather(input_, dim)


def mlp_reduce_scatter(input_: torch.Tensor) -> torch.Tensor:
    return get_mlp_world_group().reduce_scatter(input_)
