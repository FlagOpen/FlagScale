# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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

from typing import Optional, List

import torch
import torch.distributed
from vllm.distributed import GroupCoordinator as GroupCoordinatorGPU
from vllm.distributed import (
    parallel_state,
    init_model_parallel_group,
    get_world_group,
    get_dp_group,
    get_ep_group
)
from vllm.logger import logger
from omni.models.common.config.model_config import model_extra_config

initialize_model_parallel_default = parallel_state.initialize_model_parallel


class GroupCoordinator(GroupCoordinatorGPU):

    def all_to_all(
        self,
        input_: torch.Tensor,
        scatter_dim: int = 0,
        gather_dim: int = -1,
        scatter_sizes: Optional[List[int]] = None,
        gather_sizes: Optional[List[int]] = None,
    ) -> torch.Tensor:
        if self.world_size == 1:
            return input_
        return self.device_communicator.all_to_all(input_, scatter_dim, gather_dim, scatter_sizes, gather_sizes)

    def reduce_scatter(self, input_: torch.Tensor) -> torch.Tensor:
        if self.world_size == 1:
            return input_
        return self.device_communicator.reduce_scatter(input_)


_NUM_COMM_GROUP = 2
_LOCAL_COMM_LIST = None
_CROSS_COMM_LIST = None
_GLOBAL_COMM_LIST = None

_LOCAL_WORLD: Optional[GroupCoordinator] = None


def initialize_model_parallel(
    tensor_model_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
    enable_expert_parallel: bool = False,
    backend: Optional[str] = None,
) -> None:
    initialize_model_parallel_default(
        tensor_model_parallel_size,
        pipeline_model_parallel_size,
        enable_expert_parallel,
        backend,
    )

    initialize_local_world_group(backend)
    if model_extra_config.operator_opt_config.two_stage_comm:
        initialize_cross_comm_group_list(backend)
        initialize_local_comm_group_list(backend)
    else:
        initialize_world_comm_group_list(backend)
        initialize_local_comm_group_list(backend)


def get_mlp_tp_size():
    # Can be enabled
    if model_extra_config.operator_opt_config.enable_node_mlp:
        return get_local_group_world_size_from_list(0)
    else:
        return get_expert_parallel_world_size()


def get_mlp_tp_rank():
    if model_extra_config.operator_opt_config.enable_node_mlp:
        return get_local_group_rank_from_list(0)
    else:
        return get_expert_parallel_rank()


def get_mlp_world_group():
    return get_local_group_from_list(0)


def calculate_effective_local_size(local_size: int, world_size: int) -> int:
    """
    Calculate the effective local size based on available devices and world size.

    Args:
        local_size (int): Number of available NPU devices.
        world_size (int): Total number of processes in the distributed setup.

    Returns:
        int: The effective local size (minimum of local_size and world_size).

    Notes:
        - Logs a warning if not all devices are used.
        - Ensures world_size is divisible by the effective local size (raises AssertionError otherwise).
    """
    effective_local_size = min(local_size, world_size)
    if effective_local_size < local_size:
        logger.info(f"Note: Using only {effective_local_size} of {local_size} available NPU devices")

    if world_size % effective_local_size != 0:
        raise AssertionError(
            f"world_size ({world_size}) must be divisible by effective_local_size ({effective_local_size})"
        )
    return effective_local_size


def initialize_local_world_group(backend) -> None:
    """
    Initialize model parallel groups.

    Arguments:
        tensor_model_parallel_size: number of GPUs used for tensor model
            parallelism.
        pipeline_model_parallel_size: number of GPUs used for pipeline model
            parallelism.

    Let's say we have a total of 8 GPUs denoted by g0 ... g7 and we
    use 2 GPUs to parallelize the model tensor, and 4 GPUs to parallelize
    the model pipeline. The present function will
    create 4 tensor model-parallel groups and 2 pipeline model-parallel groups:
        4 tensor model-parallel groups:
            [g0, g1], [g2, g3], [g4, g5], [g6, g7]
        2 pipeline model-parallel groups:
            [g0, g2, g4, g6], [g1, g3, g5, g7]
    Note that for efficiency, the caller should make sure adjacent ranks
    are on the same DGX box. For example if we are using 2 DGX-1 boxes
    with a total of 16 GPUs, rank 0 to 7 belong to the first box and
    ranks 8 to 15 belong to the second box.
    """
    # Get world size and rank. Ensure some consistencies.
    if not torch.distributed.is_initialized():
        raise RuntimeError("torch.distributed must be initialized")
    world_size: int = torch.distributed.get_world_size()
    local_size = calculate_effective_local_size(torch.npu.device_count(), world_size)

    backend = backend or torch.distributed.get_backend(get_world_group().device_group)

    num_local_groups: int = world_size // local_size
    global _LOCAL_WORLD
    if _LOCAL_WORLD is not None:
        raise RuntimeError("_LOCAL_WORLD must be None")
    group_ranks = []
    for i in range(num_local_groups):
        ranks = list(range(i * local_size, (i + 1) * local_size))
        group_ranks.append(ranks)

    # message queue broadcaster is only used in tensor model parallel group
    _LOCAL_WORLD = init_model_parallel_group(
        group_ranks,
        get_world_group().local_rank,
        backend,
        use_message_queue_broadcaster=True,
        group_name="world_local",
    )


def initialize_local_comm_group_list(backend) -> None:
    # Get world size and rank. Ensure some consistencies.
    if not torch.distributed.is_initialized():
        raise RuntimeError("torch.distributed must be initialized")
    world_size: int = torch.distributed.get_world_size()
    local_size = calculate_effective_local_size(torch.npu.device_count(), world_size)

    backend = backend or torch.distributed.get_backend(get_world_group().device_group)

    num_local_groups: int = world_size // local_size
    global _LOCAL_COMM_LIST
    if _LOCAL_COMM_LIST is not None:
        raise RuntimeError("_LOCAL_COMM_LIST must be None")
    _LOCAL_COMM_LIST = list()
    group_ranks = []
    for i in range(num_local_groups):
        ranks = list(range(i * local_size, (i + 1) * local_size))
        group_ranks.append(ranks)

    # message queue broadcaster is only used in tensor model parallel group
    for i in range(_NUM_COMM_GROUP):
        _LOCAL_COMM_LIST.append(
            init_model_parallel_group(
                group_ranks,
                get_world_group().local_rank,
                backend,
                use_message_queue_broadcaster=True,
                group_name="world_local",
            )
        )


def initialize_cross_comm_group_list(backend) -> None:
    # Get world size and rank. Ensure some consistencies.
    if not torch.distributed.is_initialized():
        raise RuntimeError("torch.distributed must be initialized")
    world_size: int = torch.distributed.get_world_size()
    local_size = calculate_effective_local_size(torch.npu.device_count(), world_size)

    server_size = world_size // local_size

    backend = backend or torch.distributed.get_backend(get_world_group().device_group)

    # Build the pipeline model-parallel groups.
    num_cross_groups: int = world_size // server_size
    global _CROSS_COMM_LIST
    if _CROSS_COMM_LIST is not None:
        raise RuntimeError("pipeline model parallel group is already initialized")
    _CROSS_COMM_LIST = list()
    group_ranks = []
    for i in range(num_cross_groups):
        ranks = list(range(i, world_size, num_cross_groups))
        group_ranks.append(ranks)
    # pipeline parallel does not need custom allreduce

    for i in range(_NUM_COMM_GROUP):
        _CROSS_COMM_LIST.append(
            init_model_parallel_group(
                group_ranks,
                get_world_group().local_rank,
                backend,
                group_name="world_cross",
            )
        )


def initialize_world_comm_group_list(backend) -> None:
    # Get world size and rank. Ensure some consistencies.
    if not torch.distributed.is_initialized():
        raise RuntimeError("torch.distributed must be initialized")
    world_size: int = torch.distributed.get_world_size()

    backend = backend or torch.distributed.get_backend(get_world_group().device_group)

    global _GLOBAL_COMM_LIST
    if _GLOBAL_COMM_LIST is not None:
        raise RuntimeError("_GLOBAL_COMM_LIST must be None")
    _GLOBAL_COMM_LIST = list()
    group_ranks = [range(world_size)]
    for i in range(_NUM_COMM_GROUP):
        _GLOBAL_COMM_LIST.append(
            init_model_parallel_group(
                group_ranks,
                get_world_group().local_rank,
                backend,
                use_message_queue_broadcaster=True,
                group_name="world_local",
            )
        )


def get_local_world_group() -> GroupCoordinator:
    return _LOCAL_WORLD


def get_local_group_from_list(idx: int) -> GroupCoordinator:
    return _LOCAL_COMM_LIST[idx]


def get_cross_group_from_list(idx: int) -> GroupCoordinator:
    return _CROSS_COMM_LIST[idx]


def get_world_group_from_list(idx: int) -> GroupCoordinator:
    return _GLOBAL_COMM_LIST[idx]


def get_data_parallel_world_size():
    """Return world size for the tensor model parallel group."""
    group = get_dp_group()
    if group is not None:
        return group.world_size
    else:
        return 1


def get_data_parallel_rank():
    """Return my rank for the tensor model parallel group."""
    group = get_dp_group()
    if group is not None:
        return group.rank_in_group
    else:
        return 0


def get_expert_parallel_world_size():
    """Return world size for the tensor model parallel group."""
    return get_ep_group().world_size


def get_expert_parallel_rank():
    """Return my rank for the tensor model parallel group."""
    return get_ep_group().rank_in_group


def get_local_group_world_size_from_list(idx: int):
    return _LOCAL_COMM_LIST[idx].world_size


def get_local_group_rank_from_list(idx: int):
    return _LOCAL_COMM_LIST[idx].rank_in_group
