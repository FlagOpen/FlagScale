# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

import operator
from functools import reduce
from typing import Callable, List, Optional, Tuple, Union

import torch

from megatron import core
from megatron.core import ModelParallelConfig
from megatron.core.parallel_state import (
    get_pipeline_model_parallel_group,
    get_pipeline_model_parallel_next_rank,
    get_pipeline_model_parallel_prev_rank,
    get_pipeline_model_parallel_rank,
)
from megatron.core.pipeline_parallel.p2p_communication import _communicate

from flagscale.train import get_parallel_context  

# Types
Shape = Union[List[int], torch.Size]


def recv_forward_hetero(tensor_shape: Shape, config: ModelParallelConfig) -> torch.Tensor:
    """ Receive tensor from previous rank in pipeline (forward receive).

    See _communicate for argument details.
    """

    if core.parallel_state.is_pipeline_first_stage():
        input_tensor = None
    else:
        if config.timers is not None:
            config.timers('forward-recv', log_level=2).start()
        rank = torch.distributed.get_rank()
        para_ctx = get_parallel_context()
        tensor_slices = para_ctx.get_inter_mesh_tensor_slices(
            rank=rank, local_tensor_shape=tensor_shape, next=False
        )
        input_tensor = torch.empty(tensor_shape, device=torch.cuda.current_device())
        for tensor_slice in tensor_slices:
            dst_rank, (dp_start, dp_end), (sp_start, sp_end), local_hidden_size = tensor_slice
            tensor_shape_sliced = (dp_end - dp_start, sp_end - sp_start, local_hidden_size)
            group = para_ctx.get_inter_mesh_process_group(rank, dst_rank) 
            input_tensor_sliced, _, _ = _communicate(
                tensor_send_next=None,
                tensor_send_prev=None,
                recv_prev=True,
                recv_next=False,
                tensor_shape=tensor_shape_sliced,
                config=config,
                group=group,
            )
            input_tensor[dp_start:dp_end, sp_start:sp_end, :] = input_tensor_sliced
        if config.timers is not None:
            config.timers('forward-recv').stop()
    return input_tensor


def recv_backward_hetero(tensor_shape: Shape, config: ModelParallelConfig) -> torch.Tensor:
    """Receive tensor from next rank in pipeline (backward receive).

    See _communicate for argument details.
    """
    if core.parallel_state.is_pipeline_last_stage():
        output_tensor_grad = None
    else:
        if config.timers is not None:
            config.timers('backward-recv', log_level=2).start()
        rank = torch.distributed.get_rank()
        para_ctx = get_parallel_context()
        tensor_slices = para_ctx.get_inter_mesh_tensor_slices(
            rank=rank, local_tensor_shape=tensor_shape, next=True
        )
        output_tensor_grad = torch.empty(tensor_shape, device=torch.cuda.current_device())
        for tensor_slice in tensor_slices:
            dst_rank, (dp_start, dp_end), (sp_start, sp_end), local_hidden_size = tensor_slice
            tensor_shape_sliced = (dp_end - dp_start, sp_end - sp_start, local_hidden_size)
            group = para_ctx.get_inter_mesh_process_group(rank, dst_rank) 
            _, output_tensor_grad, _ = _communicate(
                tensor_send_next=None,
                tensor_send_prev=None,
                recv_prev=False,
                recv_next=True,
                tensor_shape=tensor_shape_sliced,
                config=config,
                group=group,
            )
            output_tensor_grad[dp_start:dp_end, sp_start:sp_end, :] = output_tensor_grad
        if config.timers is not None:
            config.timers('backward-recv').stop()
    return output_tensor_grad


def send_forward_hetero(output_tensor: torch.Tensor, config: ModelParallelConfig) -> None:
    """Send tensor to next rank in pipeline (forward send).

    See _communicate for argument details.
    """

    if not core.parallel_state.is_pipeline_last_stage():
        if config.timers is not None:
            config.timers('forward-send', log_level=2).start()
        rank = torch.distributed.get_rank()
        para_ctx = get_parallel_context()
        tensor_slices = para_ctx.get_inter_mesh_tensor_slices(
            rank=rank, local_tensor_shape=output_tensor.shape, next=True
        )
        for tensor_slice in tensor_slices:
            dst_rank, (dp_start, dp_end), (sp_start, sp_end), local_hidden_size = tensor_slice
            output_tensor_sliced = output_tensor[dp_start:dp_end, sp_start:sp_end, :]
            group = para_ctx.get_inter_mesh_process_group(rank, dst_rank) 
            _communicate(
                tensor_send_next=output_tensor_sliced,
                tensor_send_prev=None,
                recv_prev=False,
                recv_next=False,
                tensor_shape=None,
                config=config,
                group=group,
            )
        if config.timers is not None:
            config.timers('forward-send').stop()


def send_backward_hetero(input_tensor_grad: torch.Tensor, config: ModelParallelConfig) -> None:
    """Send tensor to previous rank in pipeline (backward send).

    See _communicate for argument details.
    """
    if not core.parallel_state.is_pipeline_first_stage():
        if config.timers is not None:
            config.timers('backward-send', log_level=2).start()
        rank = torch.distributed.get_rank()
        para_ctx = get_parallel_context()
        tensor_slices = para_ctx.get_inter_mesh_tensor_slices(
            rank=rank, local_tensor_shape=input_tensor_grad.shape, next=False
        )
        for tensor_slice in tensor_slices:
            dst_rank, (dp_start, dp_end), (sp_start, sp_end), local_hidden_size = tensor_slice
            input_tensor_grad_sliced = input_tensor_grad[dp_start:dp_end, sp_start:sp_end, :]
            group = para_ctx.get_inter_mesh_process_group(rank, dst_rank) 
            _communicate(
                tensor_send_next=None,
                tensor_send_prev=input_tensor_grad_sliced,
                recv_prev=False,
                recv_next=False,
                tensor_shape=None,
                config=config,
                group=group,
            )
        if config.timers is not None:
            config.timers('backward-send').stop()


def send_forward_recv_backward_hetero(
    output_tensor: torch.Tensor, tensor_shape: Shape, config: ModelParallelConfig
) -> torch.Tensor:
    """Batched send and recv with next rank in pipeline.

    See _communicate for argument details.
    """
    if core.parallel_state.is_pipeline_last_stage():
        output_tensor_grad = None
    else:
        if config.timers is not None:
            config.timers('forward-send-backward-recv', log_level=2).start()
        rank = torch.distributed.get_rank()
        para_ctx = get_parallel_context()
        tensor_slices = para_ctx.get_inter_mesh_tensor_slices(
            rank=rank, local_tensor_shape=output_tensor.shape, next=True
        )
        output_tensor_grad = torch.empty(tensor_shape, device=torch.cuda.current_device())
        for tensor_slice in tensor_slices:
            dst_rank, (dp_start, dp_end), (sp_start, sp_end), local_hidden_size = tensor_slice
            output_tensor_sliced = output_tensor[dp_start:dp_end, sp_start:sp_end, :]
            tensor_shape_sliced = (dp_end - dp_start, sp_end - sp_start, local_hidden_size)
            group = para_ctx.get_inter_mesh_process_group(rank, dst_rank) 
            _, output_tensor_grad_sliced, _ = _communicate(
                tensor_send_next=output_tensor_sliced,
                tensor_send_prev=None,
                recv_prev=False,
                recv_next=True,
                tensor_shape=tensor_shape_sliced,
                config=config,
                group=group,
            )
            output_tensor_grad[dp_start:dp_end, sp_start:sp_end, :] = output_tensor_grad_sliced
        if config.timers is not None:
            config.timers('forward-send-backward-recv').stop()
    return output_tensor_grad


def send_backward_recv_forward_hetero(
    input_tensor_grad: torch.Tensor, tensor_shape: Shape, config: ModelParallelConfig
) -> torch.Tensor:
    """Batched send and recv with previous rank in pipeline.

    See _communicate for argument details.
    """
    if core.parallel_state.is_pipeline_first_stage():
        input_tensor = None
    else:
        if config.timers is not None:
            config.timers('backward-send-forward-recv', log_level=2).start()
        rank = torch.distributed.get_rank()
        para_ctx = get_parallel_context()
        tensor_slices = para_ctx.get_inter_mesh_tensor_slices(
            rank=rank, local_tensor_shape=input_tensor_grad.shape, next=False
        )
        input_tensor = torch.empty(tensor_shape, device=torch.cuda.current_device())
        for tensor_slice in tensor_slices:
            dst_rank, (dp_start, dp_end), (sp_start, sp_end), local_hidden_size = tensor_slice
            input_tensor_grad_sliced = input_tensor_grad[dp_start:dp_end, sp_start:sp_end, :]
            tensor_shape_sliced = (dp_end - dp_start, sp_end - sp_start, local_hidden_size)
            group = para_ctx.get_inter_mesh_process_group(rank, dst_rank) 
            input_tensor_sliced, _, _ = _communicate(
                tensor_send_next=None,
                tensor_send_prev=input_tensor_grad_sliced,
                recv_prev=True,
                recv_next=False,
                tensor_shape=tensor_shape_sliced,
                config=config,
                group=group,
            )
            input_tensor[dp_start:dp_end, sp_start:sp_end, :] = input_tensor_sliced
        if config.timers is not None:
            config.timers('backward-send-forward-recv').stop()
    return input_tensor
