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
from flagscale.train.hetero.parallel_context import ParallelContext
# Types
Shape = Union[List[int], torch.Size]

def warm_up_comm_group_hetero(config: ModelParallelConfig):
    """ Warm up the communication for all PP groups, to avoid the hang issue.
    
    P2P comm would call batch_isend_irecv API, which requires
    all ranks of the group to participate if this API is the 
    first collective call in the group passed to `dist.P2POp`.

    See batch_isend_irecv for more details.
    """
    group = None
    rank = torch.distributed.get_rank()
    para_ctx = get_parallel_context()
    pp_groups = para_ctx.get_pipeline_model_parallel_group()
    # This is arbitrary because the shape of the recv tensor needs 
    # to be specified when communicating. 
    # It can be changed into any other shape.
    tensor_shape = [1]
    to_send_tensor= torch.empty(
            tensor_shape,
            requires_grad=True,
            device=torch.cuda.current_device() if pp_groups[0].name() != "gloo" else torch.device("cpu"),
            dtype=config.pipeline_dtype,
        )
    to_recv_tensor= torch.empty(
            tensor_shape,
            requires_grad=True,
            device=torch.device("cpu") if pp_groups[0].name() != "gloo" else torch.cuda.current_device(),
            dtype=config.pipeline_dtype,
        )

    for pp_group in pp_groups:
        group_ranks = torch.distributed.get_process_group_ranks(pp_group)
        if rank == group_ranks[0]:
            _communicate(
                tensor_send_next=to_send_tensor,
                tensor_send_prev=None,
                recv_prev=False,
                recv_next=False,
                tensor_shape=to_recv_tensor.shape,
                config=config,
                group=pp_group,
            )
        elif rank == group_ranks[-1]:
            _communicate(
                tensor_send_next=None,
                tensor_send_prev=None,
                recv_prev=True,
                recv_next=False,
                tensor_shape=to_recv_tensor.shape,
                config=config,
                group=pp_group,
            )
        elif rank in group_ranks:
            _communicate(
                tensor_send_next=to_send_tensor,
                tensor_send_prev=None,
                recv_prev=True,
                recv_next=False,
                tensor_shape=to_recv_tensor.shape,
                config=config,
                group=pp_group,
            )


def is_inter_mesh_comm(para_ctx: ParallelContext, comm_with_front_layer: bool):
    """ Judge if the p2p communication across meshes.
    
    comm_with_front_layer: if this communication is established with front layer, 
        including send_backward and recv_forward
    """

    assert para_ctx is not None, "Specify ParallelContext Necessary"
    assert comm_with_front_layer is not None, "Specify Communication Direction Necessary"
    if comm_with_front_layer:
        # To judge if current rank is in the first stage of current mesh.
        # In this condition, its pp rank should equal to 
        # the sum of pp size of all previous meshes
        total_prev_pipeline_model_parallel_size = 0
        for i in range(0, para_ctx._current_process_mesh_index):
            total_prev_pipeline_model_parallel_size += para_ctx._process_meshes[i]._rank_generator.pp
        return get_pipeline_model_parallel_rank() == total_prev_pipeline_model_parallel_size
    else:
        # To judge if current rank is in the last stage of current mesh.
        # In this condition, its pp rank should equal to 
        # the (sum of pp size of all previous and current meshes) - 1
        total_current_pipeline_model_parallel_size = 0
        for i in range(0, min(para_ctx._current_process_mesh_index + 1, len(para_ctx._process_meshes))):
            total_current_pipeline_model_parallel_size += para_ctx._process_meshes[i]._rank_generator.pp
        return get_pipeline_model_parallel_rank() == total_current_pipeline_model_parallel_size - 1        

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
        pp_groups = para_ctx.get_pipeline_model_parallel_group()
        if not is_inter_mesh_comm(para_ctx=para_ctx, comm_with_front_layer=True):
            group = None
            for pp_group in pp_groups:
                if rank in torch.distributed.get_process_group_ranks(pp_group):
                    group = pp_group
                    break
            input_tensor, _, _ = _communicate(
                tensor_send_next=None,
                tensor_send_prev=None,
                recv_prev=True,
                recv_next=False,
                tensor_shape=tensor_shape,
                config=config,
                group=group,
            )
        else:
            tensor_slices = para_ctx.get_inter_mesh_tensor_slices(
                rank=rank, local_tensor_shape=tensor_shape, next=False
            )
            input_tensor = torch.empty(tensor_shape, 
                                       device=torch.cuda.current_device() if pp_groups[0].name() != "gloo" else torch.device("cpu"), 
                                       dtype=config.pipeline_dtype, 
                                       requires_grad=True)
            if tensor_slices is not None:
                for tensor_slice in tensor_slices:
                    dst_rank, (dp_start, dp_end), (sp_start, sp_end), local_hidden_size = tensor_slice
                    tensor_shape_sliced = (sp_end - sp_start, dp_end - dp_start, local_hidden_size)
                    group = None
                    pp_groups = para_ctx.get_pipeline_model_parallel_group()
                    for pp_group in pp_groups:
                        pp_group_ranks = torch.distributed.get_process_group_ranks(pp_group)
                        if rank in pp_group_ranks and dst_rank in pp_group_ranks:
                            group = pp_group
                            break
                    input_tensor_sliced, _, _ = _communicate(
                        tensor_send_next=None,
                        tensor_send_prev=None,
                        recv_prev=True,
                        recv_next=False,
                        tensor_shape=tensor_shape_sliced,
                        config=config,
                        group=group,
                    )
                    input_tensor.data[sp_start:sp_end, dp_start:dp_end, :] = input_tensor_sliced
        if config.timers is not None:
            config.timers('forward-recv').stop()
    if input_tensor is not None and input_tensor.device == torch.device("cpu"):
        input_tensor = input_tensor.to(torch.cuda.current_device())
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
        pp_groups = para_ctx.get_pipeline_model_parallel_group()
        if not is_inter_mesh_comm(para_ctx=para_ctx, comm_with_front_layer=False):
            group = None
            for pp_group in pp_groups:
                if rank in torch.distributed.get_process_group_ranks(pp_group):
                    group = pp_group
                    break
            _, output_tensor_grad, _ = _communicate(
                tensor_send_next=None,
                tensor_send_prev=None,
                recv_prev=False,
                recv_next=True,
                tensor_shape=tensor_shape,
                config=config,
                group=group,
            )
        else:
            tensor_slices = para_ctx.get_inter_mesh_tensor_slices(
                rank=rank, local_tensor_shape=tensor_shape, next=True
            )
            output_tensor_grad = torch.empty(tensor_shape, 
                                             device=torch.cuda.current_device()if pp_groups[0].name() != "gloo" else torch.device("cpu"), 
                                             dtype=config.pipeline_dtype, 
                                             requires_grad=True)
            if tensor_slices is not None:
                for tensor_slice in tensor_slices:
                    dst_rank, (dp_start, dp_end), (sp_start, sp_end), local_hidden_size = tensor_slice
                    tensor_shape_sliced = (sp_end - sp_start, dp_end - dp_start, local_hidden_size)
                    group = None
                    for pp_group in pp_groups:
                        pp_group_ranks = torch.distributed.get_process_group_ranks(pp_group)
                        if rank in pp_group_ranks and dst_rank in pp_group_ranks:
                            group = pp_group
                            break
                    _, output_tensor_grad_sliced, _ = _communicate(
                        tensor_send_next=None,
                        tensor_send_prev=None,
                        recv_prev=False,
                        recv_next=True,
                        tensor_shape=tensor_shape_sliced,
                        config=config,
                        group=group,
                    )
                    output_tensor_grad.data[sp_start:sp_end, dp_start:dp_end, :] = output_tensor_grad_sliced
        if config.timers is not None:
            config.timers('backward-recv').stop()

    if output_tensor_grad is not None and output_tensor_grad.device == torch.device("cpu"):
        output_tensor_grad = output_tensor_grad.to(torch.cuda.current_device())
    
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
        if not is_inter_mesh_comm(para_ctx=para_ctx, comm_with_front_layer=False):
            group = None
            pp_groups = para_ctx.get_pipeline_model_parallel_group()
            for pp_group in pp_groups:
                if rank in torch.distributed.get_process_group_ranks(pp_group):
                    group = pp_group
                    break
            _communicate(
                tensor_send_next=output_tensor if group.name() != "gloo" else output_tensor.cpu(),
                tensor_send_prev=None,
                recv_prev=False,
                recv_next=False,
                tensor_shape=None,
                config=config,
                group=group,
            )
        else:
            tensor_slices = para_ctx.get_inter_mesh_tensor_slices(
                rank=rank, local_tensor_shape=output_tensor.shape, next=True
            )
            if tensor_slices is not None:
                for tensor_slice in tensor_slices:
                    dst_rank, (dp_start, dp_end), (sp_start, sp_end), local_hidden_size = tensor_slice
                    output_tensor_sliced = output_tensor[sp_start:sp_end, dp_start:dp_end, :]
                    group = None
                    pp_groups = para_ctx.get_pipeline_model_parallel_group()
                    for pp_group in pp_groups:
                        pp_group_ranks = torch.distributed.get_process_group_ranks(pp_group)
                        if rank in pp_group_ranks and dst_rank in pp_group_ranks:
                            group = pp_group
                            break
                    _communicate(
                        tensor_send_next=output_tensor_sliced.clone() if group.name() != "gloo" else output_tensor_sliced.cpu(),
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
        if not is_inter_mesh_comm(para_ctx=para_ctx, comm_with_front_layer=True):
            group = None
            pp_groups = para_ctx.get_pipeline_model_parallel_group()
            for pp_group in pp_groups:
                if rank in torch.distributed.get_process_group_ranks(pp_group):
                    group = pp_group
                    break
            _communicate(
                tensor_send_next=None,
                tensor_send_prev=input_tensor_grad if group.name() != "gloo" else input_tensor_grad.cpu(),
                recv_prev=False,
                recv_next=False,
                tensor_shape=None,
                config=config,
                group=group,
            )
        else:
            tensor_slices = para_ctx.get_inter_mesh_tensor_slices(
                rank=rank, local_tensor_shape=input_tensor_grad.shape, next=False
            )
            if tensor_slices is not None:
                for tensor_slice in tensor_slices:
                    dst_rank, (dp_start, dp_end), (sp_start, sp_end), local_hidden_size = tensor_slice
                    input_tensor_grad_sliced = input_tensor_grad[sp_start:sp_end, dp_start:dp_end, :]
                    group = None
                    pp_groups = para_ctx.get_pipeline_model_parallel_group()
                    for pp_group in pp_groups:
                        pp_group_ranks = torch.distributed.get_process_group_ranks(pp_group)
                        if rank in pp_group_ranks and dst_rank in pp_group_ranks:
                            group = pp_group
                            break
                    _communicate(
                        tensor_send_next=None,
                        tensor_send_prev=input_tensor_grad_sliced.clone() if group.name() != "gloo" else input_tensor_grad_sliced.cpu(),
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
        pp_groups = para_ctx.get_pipeline_model_parallel_group()
        if not is_inter_mesh_comm(para_ctx=para_ctx, comm_with_front_layer=False):
            group = None
            for pp_group in pp_groups:
                if rank in torch.distributed.get_process_group_ranks(pp_group):
                    group = pp_group
                    break
            _, output_tensor_grad, _ = _communicate(
                tensor_send_next=output_tensor if group.name() != "gloo" else output_tensor.cpu(),
                tensor_send_prev=None,
                recv_prev=False,
                recv_next=True,
                tensor_shape=tensor_shape,
                config=config,
                group=group,
            )
        else:
            tensor_slices = para_ctx.get_inter_mesh_tensor_slices(
                rank=rank, local_tensor_shape=output_tensor.shape, next=True
            )
            output_tensor_grad = torch.empty(tensor_shape, 
                                             device=torch.cuda.current_device() if pp_groups[0].name() != "gloo" else torch.device("cpu"),
                                             dtype=config.pipeline_dtype, 
                                             requires_grad=True)
            if tensor_slices is not None:
                for tensor_slice in tensor_slices:
                    dst_rank, (dp_start, dp_end), (sp_start, sp_end), local_hidden_size = tensor_slice
                    output_tensor_sliced = output_tensor[sp_start:sp_end, dp_start:dp_end, :]
                    tensor_shape_sliced = (sp_end - sp_start, dp_end - dp_start, local_hidden_size)
                    group = None
                    for pp_group in pp_groups:
                        pp_group_ranks = torch.distributed.get_process_group_ranks(pp_group)
                        if rank in pp_group_ranks and dst_rank in pp_group_ranks:
                            group = pp_group
                            break
                    _, output_tensor_grad_sliced, _ = _communicate(
                        tensor_send_next=output_tensor_sliced.clone() if group.name() != "gloo" else output_tensor_sliced.cpu(),
                        tensor_send_prev=None,
                        recv_prev=False,
                        recv_next=True,
                        tensor_shape=tensor_shape_sliced,
                        config=config,
                        group=group,
                    )
                    output_tensor_grad.data[sp_start:sp_end, dp_start:dp_end, :] = output_tensor_grad_sliced
        if config.timers is not None:
            config.timers('forward-send-backward-recv').stop()
    if output_tensor_grad is not None and output_tensor_grad.device == torch.device("cpu"):
        output_tensor_grad = output_tensor_grad.to(torch.cuda.current_device())
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
        pp_groups = para_ctx.get_pipeline_model_parallel_group()
        if not is_inter_mesh_comm(para_ctx=para_ctx, comm_with_front_layer=True):
            group = None
            for pp_group in pp_groups:
                if rank in torch.distributed.get_process_group_ranks(pp_group):
                    group = pp_group
                    break
            input_tensor, _, _ = _communicate(
                tensor_send_next=None,
                tensor_send_prev=input_tensor_grad if group.name() != "gloo" else input_tensor_grad.cpu(),
                recv_prev=True,
                recv_next=False,
                tensor_shape=tensor_shape,
                config=config,
                group=group,
            )
        else:
            tensor_slices = para_ctx.get_inter_mesh_tensor_slices(
                rank=rank, local_tensor_shape=input_tensor_grad.shape, next=False
            )
            input_tensor = torch.empty(tensor_shape, 
                                       device=torch.cuda.current_device() if pp_groups[0].name() != "gloo" else torch.device("cpu"),
                                       dtype=config.pipeline_dtype, 
                                       requires_grad=True)
            if tensor_slices is not None:
                for tensor_slice in tensor_slices:
                    dst_rank, (dp_start, dp_end), (sp_start, sp_end), local_hidden_size = tensor_slice
                    input_tensor_grad_sliced = input_tensor_grad[sp_start:sp_end, dp_start:dp_end, :]
                    tensor_shape_sliced = (sp_end - sp_start, dp_end - dp_start, local_hidden_size)
                    group = None
                    for pp_group in pp_groups:
                        pp_group_ranks = torch.distributed.get_process_group_ranks(pp_group)
                        if rank in pp_group_ranks and dst_rank in pp_group_ranks:
                            group = pp_group
                            break
                    input_tensor_sliced, _, _ = _communicate(
                        tensor_send_next=None,
                        tensor_send_prev=input_tensor_grad_sliced.clone() if group.name() != "gloo" else input_tensor_grad_sliced.cpu(),
                        recv_prev=True,
                        recv_next=False,
                        tensor_shape=tensor_shape_sliced,
                        config=config,
                        group=group,
                    )
                    input_tensor.data[sp_start:sp_end, dp_start:dp_end, :] = input_tensor_sliced
        if config.timers is not None:
            config.timers('backward-send-forward-recv').stop()
    if input_tensor is not None and input_tensor.device == torch.device("cpu"):
        input_tensor = input_tensor.to(torch.cuda.current_device())
    return input_tensor
