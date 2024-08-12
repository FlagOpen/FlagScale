# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

import operator
from functools import reduce
from typing import Callable, List, Optional, Tuple, Union

import torch
import time

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

def debug_print_value(**kwargs):
    return
    def debug_stream(**kwargs):
        for var_name, value in kwargs.items():
            print(f"{var_name}: {value}")
        yield
    next(debug_stream(**kwargs))


def warm_up_comm_group_hetero(config: ModelParallelConfig):
    group = None
    rank = torch.distributed.get_rank()
    para_ctx = get_parallel_context()
    pp_groups = para_ctx.get_pipeline_model_parallel_group()
    tensor_shape = [1, 3]
    to_send_tensor= torch.empty(
            tensor_shape,
            requires_grad=True,
            device=torch.cuda.current_device(),
            dtype=config.pipeline_dtype,
        )
    to_recv_tensor= torch.empty(
            tensor_shape,
            requires_grad=True,
            device=torch.cuda.current_device(),
            dtype=config.pipeline_dtype,
        )

    for pp_group in pp_groups:
        group_ranks = torch.distributed.get_process_group_ranks(pp_group)
        debug_print_value(group_ranks=group_ranks)
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
    assert para_ctx is not None, "Specify ParallelContext Necessary"
    assert comm_with_front_layer is not None, "Specify Communication Direction Necessary"
    # debug_print_value(get_pipeline_model_parallel_rank=core.parallel_state.get_pipeline_model_parallel_rank(),
    #                   _current_process_mesh_index=para_ctx._current_process_mesh_index)
    if comm_with_front_layer:
        total_prev_pipeline_model_parallel_size = 0
        for i in range(0, para_ctx._current_process_mesh_index):
            total_prev_pipeline_model_parallel_size += para_ctx._process_meshes[i]._rank_generator.pp
        # debug_print_value(total_prev_pipeline_model_parallel_size=total_prev_pipeline_model_parallel_size)
        return core.parallel_state.get_pipeline_model_parallel_rank() == total_prev_pipeline_model_parallel_size
    else:
        total_current_pipeline_model_parallel_size = 0
        for i in range(0, min(para_ctx._current_process_mesh_index + 1, len(para_ctx._process_meshes))):
            total_current_pipeline_model_parallel_size += para_ctx._process_meshes[i]._rank_generator.pp
        # debug_print_value(total_current_pipeline_model_parallel_size=total_current_pipeline_model_parallel_size)
        # if para_ctx._current_process_mesh_index == len(para_ctx._process_meshes)-1:
        return core.parallel_state.get_pipeline_model_parallel_rank() == total_current_pipeline_model_parallel_size - 1        

def recv_forward_hetero(tensor_shape: Shape, config: ModelParallelConfig) -> torch.Tensor:
    """ Receive tensor from previous rank in pipeline (forward receive).

    See _communicate for argument details.
    """
    debug_print_value(recv_forward_hetero=True, 
                      is_pipeline_first_stage=core.parallel_state.is_pipeline_first_stage())

    if core.parallel_state.is_pipeline_first_stage():
        input_tensor = None
    else:
        if config.timers is not None:
            config.timers('forward-recv', log_level=2).start()
        rank = torch.distributed.get_rank()
        para_ctx = get_parallel_context()
        debug_print_value(is_inter_mesh_comm=is_inter_mesh_comm(para_ctx=para_ctx, comm_with_front_layer=True))
        if not is_inter_mesh_comm(para_ctx=para_ctx, comm_with_front_layer=True):
            # [0] is not correct
            group = None
            pp_groups = para_ctx.get_pipeline_model_parallel_group()
            for pp_group in pp_groups:
                if rank in torch.distributed.get_process_group_ranks(pp_group):
                    group = pp_group
                    break
            # group = para_ctx.get_pipeline_model_parallel_group()[0]
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
            debug_print_value(tensor_slices=tensor_slices)
            input_tensor = torch.empty(tensor_shape, device=torch.cuda.current_device(), dtype=config.pipeline_dtype, requires_grad=True)
            if tensor_slices is not None:
                for tensor_slice in tensor_slices:
                    dst_rank, (dp_start, dp_end), (sp_start, sp_end), local_hidden_size = tensor_slice
                    # tensor_shape_sliced = (dp_end - dp_start, sp_end - sp_start, local_hidden_size)
                    tensor_shape_sliced = (sp_end - sp_start, dp_end - dp_start, local_hidden_size)
                    # group = para_ctx.get_inter_mesh_process_group(rank, dst_rank) 
                    group = None
                    pp_groups = para_ctx.get_pipeline_model_parallel_group()
                    for pp_group in pp_groups:
                        pp_group_ranks = torch.distributed.get_process_group_ranks(pp_group)
                        if rank in pp_group_ranks and dst_rank in pp_group_ranks:
                            group = pp_group
                            break
                        debug_print_value(pp_group_ranks=pp_group_ranks, rank=rank, dst_rank=dst_rank)
                    input_tensor_sliced, _, _ = _communicate(
                        tensor_send_next=None,
                        tensor_send_prev=None,
                        recv_prev=True,
                        recv_next=False,
                        tensor_shape=tensor_shape_sliced,
                        config=config,
                        group=group,
                    )
                    debug_print_value(tensor_shape=tensor_shape, 
                                      tensor_shape_sliced=tensor_shape_sliced,
                                      input_tensor_sliced_shape=input_tensor_sliced.shape,
                                      sp_start=sp_start,
                                      sp_end=sp_end,
                                      dp_start=dp_start,
                                      dp_end=dp_end)
                    input_tensor.data[sp_start:sp_end, dp_start:dp_end, :] = input_tensor_sliced
        if config.timers is not None:
            config.timers('forward-recv').stop()
    return input_tensor


def recv_backward_hetero(tensor_shape: Shape, config: ModelParallelConfig) -> torch.Tensor:
    """Receive tensor from next rank in pipeline (backward receive).

    See _communicate for argument details.
    """
    debug_print_value(recv_backward_hetero=True, 
                      is_pipeline_last_stage=core.parallel_state.is_pipeline_last_stage())

    if core.parallel_state.is_pipeline_last_stage():
        output_tensor_grad = None
    else:
        if config.timers is not None:
            config.timers('backward-recv', log_level=2).start()
        rank = torch.distributed.get_rank()
        para_ctx = get_parallel_context()
        # if para_ctx._current_process_mesh_index == len(para_ctx._process_meshes):
        debug_print_value(is_inter_mesh_comm=is_inter_mesh_comm(para_ctx=para_ctx, comm_with_front_layer=False))
        if not is_inter_mesh_comm(para_ctx=para_ctx, comm_with_front_layer=False):
            # group = para_ctx.get_pipeline_model_parallel_group()[0]
            group = None
            pp_groups = para_ctx.get_pipeline_model_parallel_group()
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
            debug_print_value(tensor_slices=tensor_slices)
            output_tensor_grad = torch.empty(tensor_shape, device=torch.cuda.current_device(), dtype=config.pipeline_dtype, requires_grad=True)
            if tensor_slices is not None:
                for tensor_slice in tensor_slices:
                    dst_rank, (dp_start, dp_end), (sp_start, sp_end), local_hidden_size = tensor_slice
                    # tensor_shape_sliced = (dp_end - dp_start, sp_end - sp_start, local_hidden_size)
                    tensor_shape_sliced = (sp_end - sp_start, dp_end - dp_start, local_hidden_size)
                    # group = para_ctx.get_inter_mesh_process_group(rank, dst_rank) 
                    group = None
                    pp_groups = para_ctx.get_pipeline_model_parallel_group()
                    for pp_group in pp_groups:
                        pp_group_ranks = torch.distributed.get_process_group_ranks(pp_group)
                        if rank in pp_group_ranks and dst_rank in pp_group_ranks:
                            group = pp_group
                            break
                        debug_print_value(pp_group_ranks=pp_group_ranks, rank=rank, dst_rank=dst_rank)
                    _, output_tensor_grad_sliced, _ = _communicate(
                        tensor_send_next=None,
                        tensor_send_prev=None,
                        recv_prev=False,
                        recv_next=True,
                        tensor_shape=tensor_shape_sliced,
                        config=config,
                        group=group,
                    )
                    debug_print_value(tensor_shape=tensor_shape, 
                                      tensor_shape_sliced=tensor_shape_sliced,
                                      output_tensor_grad_sliced=output_tensor_grad_sliced.shape,
                                      sp_start=sp_start,
                                      sp_end=sp_end,
                                      dp_start=dp_start,
                                      dp_end=dp_end)
                    output_tensor_grad.data[sp_start:sp_end, dp_start:dp_end, :] = output_tensor_grad_sliced
        if config.timers is not None:
            config.timers('backward-recv').stop()
    return output_tensor_grad


def send_forward_hetero(output_tensor: torch.Tensor, config: ModelParallelConfig) -> None:
    """Send tensor to next rank in pipeline (forward send).

    See _communicate for argument details.
    """
    debug_print_value(send_forward_hetero=True, 
                      is_pipeline_last_stage=core.parallel_state.is_pipeline_last_stage())
    if not core.parallel_state.is_pipeline_last_stage():
        if config.timers is not None:
            config.timers('forward-send', log_level=2).start()
        rank = torch.distributed.get_rank()
        para_ctx = get_parallel_context()
        # if para_ctx._current_process_mesh_index == len(para_ctx._process_meshes):
        debug_print_value(is_inter_mesh_comm=is_inter_mesh_comm(para_ctx=para_ctx, comm_with_front_layer=False))
        if not is_inter_mesh_comm(para_ctx=para_ctx, comm_with_front_layer=False):
            # group = para_ctx.get_pipeline_model_parallel_group()[0]
            group = None
            pp_groups = para_ctx.get_pipeline_model_parallel_group()
            for pp_group in pp_groups:
                if rank in torch.distributed.get_process_group_ranks(pp_group):
                    group = pp_group
                    break
            _communicate(
                tensor_send_next=output_tensor,
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
            debug_print_value(tensor_slices=tensor_slices)
            if tensor_slices is not None:
                for tensor_slice in tensor_slices:
                    dst_rank, (dp_start, dp_end), (sp_start, sp_end), local_hidden_size = tensor_slice
                    output_tensor_sliced = output_tensor[sp_start:sp_end, dp_start:dp_end, :]
                    # group = para_ctx.get_inter_mesh_process_group(rank, dst_rank) 
                    group = None
                    pp_groups = para_ctx.get_pipeline_model_parallel_group()
                    for pp_group in pp_groups:
                        pp_group_ranks = torch.distributed.get_process_group_ranks(pp_group)
                        if rank in pp_group_ranks and dst_rank in pp_group_ranks:
                            group = pp_group
                            break
                        debug_print_value(pp_group_ranks=pp_group_ranks, rank=rank, dst_rank=dst_rank)
                    _communicate(
                        tensor_send_next=output_tensor_sliced,
                        # tensor_send_next=output_tensor[sp_start:sp_end, dp_start:dp_end, :],
                        tensor_send_prev=None,
                        recv_prev=False,
                        recv_next=False,
                        tensor_shape=None,
                        config=config,
                        group=group,
                    )
                    debug_print_value(output_tensor_sliced_shape=output_tensor_sliced.shape,
                        sp_start=sp_start,
                        sp_end=sp_end,
                        dp_start=dp_start,
                        dp_end=dp_end)
        if config.timers is not None:
            config.timers('forward-send').stop()


def send_backward_hetero(input_tensor_grad: torch.Tensor, config: ModelParallelConfig) -> None:
    """Send tensor to previous rank in pipeline (backward send).

    See _communicate for argument details.
    """
    debug_print_value(send_backward_hetero=True, 
                      is_pipeline_first_stage=core.parallel_state.is_pipeline_first_stage())
    if not core.parallel_state.is_pipeline_first_stage():
        if config.timers is not None:
            config.timers('backward-send', log_level=2).start()
        rank = torch.distributed.get_rank()
        para_ctx = get_parallel_context()
        # if para_ctx._current_process_mesh_index == 0 :
        debug_print_value(is_inter_mesh_comm=is_inter_mesh_comm(para_ctx=para_ctx, comm_with_front_layer=True))
        if not is_inter_mesh_comm(para_ctx=para_ctx, comm_with_front_layer=True):
            # group = para_ctx.get_pipeline_model_parallel_group()[0]
            group = None
            pp_groups = para_ctx.get_pipeline_model_parallel_group()
            for pp_group in pp_groups:
                if rank in torch.distributed.get_process_group_ranks(pp_group):
                    group = pp_group
                    break
            _communicate(
                tensor_send_next=None,
                tensor_send_prev=input_tensor_grad,
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
            debug_print_value(tensor_slices=tensor_slices)
            if tensor_slices is not None:
                for tensor_slice in tensor_slices:
                    dst_rank, (dp_start, dp_end), (sp_start, sp_end), local_hidden_size = tensor_slice
                    input_tensor_grad_sliced = input_tensor_grad[sp_start:sp_end, dp_start:dp_end, :]
                    # group = para_ctx.get_inter_mesh_process_group(rank, dst_rank) 
                    group = None
                    pp_groups = para_ctx.get_pipeline_model_parallel_group()
                    for pp_group in pp_groups:
                        pp_group_ranks = torch.distributed.get_process_group_ranks(pp_group)
                        if rank in pp_group_ranks and dst_rank in pp_group_ranks:
                            group = pp_group
                            break
                        debug_print_value(pp_group_ranks=pp_group_ranks, rank=rank, dst_rank=dst_rank)
                    _communicate(
                        tensor_send_next=None,
                        tensor_send_prev=input_tensor_grad_sliced,
                        recv_prev=False,
                        recv_next=False,
                        tensor_shape=None,
                        config=config,
                        group=group,
                    )
                    debug_print_value(input_tensor_grad_sliced_shape=input_tensor_grad_sliced.shape,
                        sp_start=sp_start,
                        sp_end=sp_end,
                        dp_start=dp_start,
                        dp_end=dp_end)
        if config.timers is not None:
            config.timers('backward-send').stop()


def send_forward_recv_backward_hetero(
    output_tensor: torch.Tensor, tensor_shape: Shape, config: ModelParallelConfig
) -> torch.Tensor:
    """Batched send and recv with next rank in pipeline.

    See _communicate for argument details.
    """
    debug_print_value(send_forward_recv_backward_hetero=True, 
                      is_pipeline_last_stage=core.parallel_state.is_pipeline_last_stage())
    if core.parallel_state.is_pipeline_last_stage():
        output_tensor_grad = None
    else:
        if config.timers is not None:
            config.timers('forward-send-backward-recv', log_level=2).start()
        rank = torch.distributed.get_rank()
        para_ctx = get_parallel_context()
        # if para_ctx._current_process_mesh_index == len(para_ctx._process_meshes):
        debug_print_value(is_inter_mesh_comm=is_inter_mesh_comm(para_ctx=para_ctx, comm_with_front_layer=False))
        if not is_inter_mesh_comm(para_ctx=para_ctx, comm_with_front_layer=False):
            # group = para_ctx.get_pipeline_model_parallel_group()[0]
            group = None
            pp_groups = para_ctx.get_pipeline_model_parallel_group()
            for pp_group in pp_groups:
                if rank in torch.distributed.get_process_group_ranks(pp_group):
                    group = pp_group
                    break
            _, output_tensor_grad, _ = _communicate(
                tensor_send_next=output_tensor,
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
            debug_print_value(tensor_slices=tensor_slices)
            output_tensor_grad = torch.empty(tensor_shape, device=torch.cuda.current_device(), dtype=config.pipeline_dtype, requires_grad=True)
            if tensor_slices is not None:
                for tensor_slice in tensor_slices:
                    dst_rank, (dp_start, dp_end), (sp_start, sp_end), local_hidden_size = tensor_slice
                    output_tensor_sliced = output_tensor[sp_start:sp_end, dp_start:dp_end, :]
                    # tensor_shape_sliced = (dp_end - dp_start, sp_end - sp_start, local_hidden_size)
                    tensor_shape_sliced = (sp_end - sp_start, dp_end - dp_start, local_hidden_size)
                    group = None
                    pp_groups = para_ctx.get_pipeline_model_parallel_group()
                    for pp_group in pp_groups:
                        pp_group_ranks = torch.distributed.get_process_group_ranks(pp_group)
                        if rank in pp_group_ranks and dst_rank in pp_group_ranks:
                            group = pp_group
                            break
                        debug_print_value(pp_group_ranks=pp_group_ranks, rank=rank, dst_rank=dst_rank)
                    # group = para_ctx.get_inter_mesh_process_group(rank, dst_rank) 
                    debug_print_value(tensor_shape=tensor_shape, 
                                      tensor_shape_sliced=tensor_shape_sliced,
                                      output_tensor_sliced_shape=output_tensor_sliced.shape,
                                    #   output_tensor_grad_sliced_shape=output_tensor_grad_sliced.shape,
                                      sp_start=sp_start,
                                      sp_end=sp_end,
                                      dp_start=dp_start,
                                      dp_end=dp_end)
                    _, output_tensor_grad_sliced, _ = _communicate(
                        tensor_send_next=output_tensor_sliced,
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
    return output_tensor_grad


def send_backward_recv_forward_hetero(
    input_tensor_grad: torch.Tensor, tensor_shape: Shape, config: ModelParallelConfig
) -> torch.Tensor:
    """Batched send and recv with previous rank in pipeline.

    See _communicate for argument details.
    """
    debug_print_value(send_backward_recv_forward_hetero=True, 
                      is_pipeline_first_stage=core.parallel_state.is_pipeline_first_stage())
    if core.parallel_state.is_pipeline_first_stage():
        input_tensor = None
    else:
        if config.timers is not None:
            config.timers('backward-send-forward-recv', log_level=2).start()
        rank = torch.distributed.get_rank()
        para_ctx = get_parallel_context()
        # if para_ctx._current_process_mesh_index == 0 :
        debug_print_value(is_inter_mesh_comm=is_inter_mesh_comm(para_ctx=para_ctx, comm_with_front_layer=True))
        if not is_inter_mesh_comm(para_ctx=para_ctx, comm_with_front_layer=True):
            # group = para_ctx.get_pipeline_model_parallel_group()[0]
            group = None
            pp_groups = para_ctx.get_pipeline_model_parallel_group()
            for pp_group in pp_groups:
                if rank in torch.distributed.get_process_group_ranks(pp_group):
                    group = pp_group
                    break
            input_tensor, _, _ = _communicate(
                tensor_send_next=None,
                tensor_send_prev=input_tensor_grad,
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
            debug_print_value(tensor_slices=tensor_slices)
            input_tensor = torch.empty(tensor_shape, device=torch.cuda.current_device(), dtype=config.pipeline_dtype, requires_grad=True)
            if tensor_slices is not None:
                for tensor_slice in tensor_slices:
                    dst_rank, (dp_start, dp_end), (sp_start, sp_end), local_hidden_size = tensor_slice
                    input_tensor_grad_sliced = input_tensor_grad[sp_start:sp_end, dp_start:dp_end, :]
                    # tensor_shape_sliced = (dp_end - dp_start, sp_end - sp_start, local_hidden_size)
                    tensor_shape_sliced = (sp_end - sp_start, dp_end - dp_start, local_hidden_size)
                    # group = para_ctx.get_inter_mesh_process_group(rank, dst_rank) 
                    group = None
                    pp_groups = para_ctx.get_pipeline_model_parallel_group()
                    for pp_group in pp_groups:
                        pp_group_ranks = torch.distributed.get_process_group_ranks(pp_group)
                        if rank in pp_group_ranks and dst_rank in pp_group_ranks:
                            group = pp_group
                            break
                        debug_print_value(pp_group_ranks=pp_group_ranks, rank=rank, dst_rank=dst_rank)
                    # debug_print_value(group=group,
                    #                   rank=rank,
                    #                   dst_rank=dst_rank,
                    #                   input_tensor_grad=input_tensor_grad,
                    #                   dp_start=dp_start,
                    #                   dp_end=dp_end,
                    #                   sp_start=sp_start,
                    #                   sp_end=sp_end,
                    #                   tensor_shape_sliced=tensor_shape_sliced)
                    input_tensor_sliced, _, _ = _communicate(
                        tensor_send_next=None,
                        tensor_send_prev=input_tensor_grad_sliced,
                        recv_prev=True,
                        recv_next=False,
                        tensor_shape=tensor_shape_sliced,
                        config=config,
                        group=group,
                    )
                    input_tensor.data[sp_start:sp_end, dp_start:dp_end, :] = input_tensor_sliced
        if config.timers is not None:
            config.timers('backward-send-forward-recv').stop()
    return input_tensor
