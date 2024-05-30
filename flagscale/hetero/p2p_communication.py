# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

import operator
from functools import reduce
from typing import Callable, List, Optional, Tuple, Union

import torch

from megatron.core import ModelParallelConfig
#from flagscale.hetero import parallel_state
from megatron.core.parallel_state import (
    get_pipeline_model_parallel_group,
    get_pipeline_model_parallel_next_rank,
    get_pipeline_model_parallel_prev_rank,
    get_pipeline_model_parallel_rank,
    get_same_tp_pipeline_model_parallel_group,
    get_diff_tp_pipeline_model_parallel_group,
    get_same_tp_pipeline_model_parallel_next_rank,
    get_same_tp_pipeline_model_parallel_prev_rank,
    get_diff_tp_pipeline_model_parallel_next_rank,
    get_diff_tp_pipeline_model_parallel_prev_rank,
    is_pipeline_first_stage,
    is_pipeline_last_stage    
)

# Types
Shape = Union[List[int], torch.Size]

def _batched_p2p_ops_tp_hetero(
    *,
    tensor_send_prev: Optional[torch.Tensor],
    tensor_recv_prev: Optional[torch.Tensor],
    tensor_send_next: Optional[torch.Tensor],
    tensor_recv_next: Optional[torch.Tensor],
    send_group_ids: int,
    recv_group_ids: int,
):
    reqs = []
    max_group_ids = max(send_group_ids, recv_group_ids)
    group_ops = [[] for _ in range(max_group_ids)]
    if tensor_send_prev is not None:
        if len(tensor_send_prev) == send_group_ids:
            for i in range(send_group_ids):
                group_ops[i].append(torch.distributed.P2POp(
                    torch.distributed.isend,
                    tensor_send_prev[i],
                    get_pipeline_model_parallel_prev_rank()[i],
                    get_same_tp_pipeline_model_parallel_group() if get_pipeline_model_parallel_prev_rank()[i] == get_same_tp_pipeline_model_parallel_prev_rank() \
                        else get_diff_tp_pipeline_model_parallel_group()[i]
                ))
        else:
            for i in range(send_group_ids):
                group_ops[i].append(torch.distributed.P2POp(
                    torch.distributed.isend,
                    tensor_send_prev[0],
                    get_pipeline_model_parallel_prev_rank()[i],
                    get_same_tp_pipeline_model_parallel_group() if get_pipeline_model_parallel_prev_rank()[i] == get_same_tp_pipeline_model_parallel_prev_rank() \
                        else get_diff_tp_pipeline_model_parallel_group()[i]
                ))
    if tensor_recv_prev is not None:
        for i in range(recv_group_ids):
            group_ops[i].append(torch.distributed.P2POp(
                torch.distributed.irecv,
                tensor_recv_prev[i],
                get_pipeline_model_parallel_prev_rank()[i],
                get_same_tp_pipeline_model_parallel_group() if get_pipeline_model_parallel_prev_rank()[i] == get_same_tp_pipeline_model_parallel_prev_rank() \
                        else get_diff_tp_pipeline_model_parallel_group()[i]
            ))
    if tensor_send_next is not None:
        if len(tensor_send_next) == send_group_ids:
            for i in range(send_group_ids):
                group_ops[i].append(torch.distributed.P2POp(
                    torch.distributed.isend,
                    tensor_send_next[i],
                    get_pipeline_model_parallel_next_rank()[i],
                    get_same_tp_pipeline_model_parallel_group() if get_pipeline_model_parallel_next_rank()[i] == get_same_tp_pipeline_model_parallel_next_rank() \
                        else get_diff_tp_pipeline_model_parallel_group()[i]
                ))
        else:
            for i in range(send_group_ids):
                group_ops[i].append(torch.distributed.P2POp(
                    torch.distributed.isend,
                    tensor_send_next[0],
                    get_pipeline_model_parallel_next_rank()[i],
                    get_same_tp_pipeline_model_parallel_group() if get_pipeline_model_parallel_next_rank()[i] == get_same_tp_pipeline_model_parallel_next_rank() \
                        else get_diff_tp_pipeline_model_parallel_group()[i]
                ))
    if tensor_recv_next is not None:
        for i in range(recv_group_ids):
            group_ops[i].append(torch.distributed.P2POp(
                torch.distributed.irecv,
                tensor_recv_next[i],
                get_pipeline_model_parallel_next_rank()[i],
                get_same_tp_pipeline_model_parallel_group() if get_pipeline_model_parallel_next_rank()[i] == get_same_tp_pipeline_model_parallel_next_rank() \
                        else get_diff_tp_pipeline_model_parallel_group()[i]
            ))
    
    for i in range(len(group_ops)):
        if len(group_ops[i]) > 0:
            reqs.append(torch.distributed.batch_isend_irecv(group_ops[i]))
    
    return reqs

def _communicate_tp_hetero(
    *,
    tensor_send_next: Optional[torch.Tensor],
    tensor_send_prev: Optional[torch.Tensor],
    recv_prev: bool,
    recv_next: bool,
    tensor_shapes: list,
    group_ids: Union[int, List[int]],
    config: ModelParallelConfig,
    wait_on_reqs: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    
    tensor_recv_prev = None
    tensor_recv_next = None

    recv_prev_shape = tensor_shapes
    recv_next_shape = tensor_shapes
    
    '''
    if not config.variable_seq_lengths:
        recv_prev_shape = tensor_shape
        recv_next_shape = tensor_shape
    else:
        recv_prev_shape, recv_next_shape = _communicate_shapes(
            tensor_send_next, tensor_send_prev, recv_prev, recv_next, config
        )
    '''
        
    if recv_prev:
        if config.pipeline_dtype is None:
            raise RuntimeError("pipeline_dtype must be provided if recv_prev is True")
        if tensor_shapes is None:
            raise RuntimeError(
                "tensor_shape must be specified if recv_prev is True. "
                "Common tensor_shape is (seq_length, micro_batch_size, hidden_size)"
            )
        if tensor_send_prev != None:
            assert isinstance(group_ids, list), 'send_backward and recv_forward need 2 group_ids!' 
            send_group_ids, recv_group_ids = group_ids[0], group_ids[1]
        else:
            send_group_ids, recv_group_ids = 0, group_ids
        tensor_recv_prev = []
        for i in range(recv_group_ids):
            tensor_recv_prev.append(torch.empty(
                recv_prev_shape[i] if len(recv_prev_shape) == recv_group_ids else recv_prev_shape[0],
                requires_grad=True,
                device=torch.cuda.current_device(),
                dtype=config.pipeline_dtype,
            ))
    else:
        if tensor_send_prev != None:
            send_group_ids, recv_group_ids = group_ids, 0
       
    if recv_next:
        if config.pipeline_dtype is None:
            raise RuntimeError("dtype must be provided if recv_next is True")
        if tensor_shapes is None:
            raise RuntimeError(
                "tensor_shape must be specified if recv_next is True. "
                "Common tensor_shape is (seq_length, micro_batch_size, hidden_size)"
            )
        if tensor_send_next != None:
            assert isinstance(group_ids, list), 'send_forward and recv_backward need 2 group_ids!' 
            send_group_ids, recv_group_ids = group_ids[0], group_ids[1]            
        else:
            send_group_ids, recv_group_ids = 0, group_ids
        tensor_recv_next = []
        for i in range(recv_group_ids):
            tensor_recv_next.append(torch.empty(
                recv_next_shape[i] if len(recv_next_shape) == recv_group_ids else recv_next_shape[0],
                requires_grad=True,
                device=torch.cuda.current_device(),
                dtype=config.pipeline_dtype,
            ))
    else:
        if tensor_send_next != None:
            send_group_ids, recv_group_ids = group_ids, 0

    #Note: normal p2p_ops hang
    #p2p_func = _p2p_ops_tp_hetero
    assert wait_on_reqs
    p2p_func = _batched_p2p_ops_tp_hetero
    reqs = p2p_func(
            tensor_send_prev=tensor_send_prev,
            tensor_recv_prev=tensor_recv_prev,
            tensor_send_next=tensor_send_next,
            tensor_recv_next=tensor_recv_next,
            send_group_ids=send_group_ids,
            recv_group_ids=recv_group_ids,
    )
    
    if wait_on_reqs and len(reqs) > 0:
        for req in reqs:
            if isinstance(req, list):
                for op in req:
                    op.wait()
            else:
                req.wait()
        reqs = None

    if config.batch_p2p_comm and config.batch_p2p_sync:
        # To protect against race condition when using batch_isend_irecv().
        # User should assert that we have a modern enough PyTorch to not need this
        torch.cuda.synchronize()

    return tensor_recv_prev, tensor_recv_next, reqs

def tp_hetero_recv_forward(tensor_shapes, group_ids, config):
    """ Receive tensor from previous rank in pipeline (forward receive).

    See _communicate for argument details.
    """
    
    if is_pipeline_first_stage():
        input_tensor = None
    else:
        if config.timers is not None:
            config.timers('forward-recv', log_level=2).start()
        input_tensor, _, _ = _communicate_tp_hetero(
            tensor_send_next=None,
            tensor_send_prev=None,
            recv_prev=True,
            recv_next=False,
            tensor_shapes=tensor_shapes,
            group_ids=group_ids,
            config=config,
        )
        if config.timers is not None:
            config.timers('forward-recv').stop()
    return input_tensor

def tp_hetero_recv_backward(tensor_shapes, group_ids, config):
    """Receive tensor from next rank in pipeline (backward receive).

    See _communicate for argument details.
    """
    if is_pipeline_last_stage():
        output_tensor_grad = None
    else:
        if config.timers is not None:
            config.timers('backward-recv', log_level=2).start()
        _, output_tensor_grad, _ = _communicate_tp_hetero(
            tensor_send_next=None,
            tensor_send_prev=None,
            recv_prev=False,
            recv_next=True,
            tensor_shapes=tensor_shapes,
            group_ids=group_ids,
            config=config,
        )
        if config.timers is not None:
            config.timers('backward-recv').stop()
    return output_tensor_grad

def tp_hetero_send_forward(output_tensors, group_ids, config):
    """Send tensor to next rank in pipeline (forward send).

    See _communicate for argument details.
    """
    
    if not is_pipeline_last_stage():
        if config.timers is not None:
            config.timers('forward-send', log_level=2).start()
        _communicate_tp_hetero(
            tensor_send_next=output_tensors,
            tensor_send_prev=None,
            recv_prev=False,
            recv_next=False,
            tensor_shapes=None,
            group_ids=group_ids,
            config=config,
        )
        if config.timers is not None:
            config.timers('forward-send').stop()

def tp_hetero_send_backward(input_tensor_grads, group_ids, config):
    """Send tensor to previous rank in pipeline (backward send).

    See _communicate for argument details.
    """
    if not is_pipeline_first_stage():
        if config.timers is not None:
            config.timers('backward-send', log_level=2).start()
        _communicate_tp_hetero(
            tensor_send_next=None,
            tensor_send_prev=input_tensor_grads,
            recv_prev=False,
            recv_next=False,
            tensor_shapes=None,
            group_ids=group_ids,
            config=config,
        )
        if config.timers is not None:
            config.timers('backward-send').stop()

def tp_hetero_send_forward_recv_backward(output_tensors, tensor_shapes, send_group_ids, recv_group_ids, config):
    """Batched send and recv with next rank in pipeline.

    See _communicate for argument details.
    """
    if is_pipeline_last_stage():
        output_tensor_grad = None
    else:
        if config.timers is not None:
            config.timers('forward-send-backward-recv', log_level=2).start()
        _, output_tensor_grad, _ = _communicate_tp_hetero(
            tensor_send_next=output_tensors,
            tensor_send_prev=None,
            recv_prev=False,
            recv_next=True,
            tensor_shapes=tensor_shapes,
            group_ids=[send_group_ids, recv_group_ids],
            config=config,
        )
        if config.timers is not None:
            config.timers('forward-send-backward-recv').stop()
    return output_tensor_grad    

def tp_hetero_send_backward_recv_forward(input_tensor_grads, tensor_shapes, send_group_ids, recv_group_ids, config):
    """Batched send and recv with previous rank in pipeline.

    See _communicate for argument details.
    """
    if is_pipeline_first_stage():
        input_tensor = None
    else:
        if config.timers is not None:
            config.timers('backward-send-forward-recv', log_level=2).start()
        input_tensor, _, _ = _communicate_tp_hetero(
            tensor_send_next=None,
            tensor_send_prev=input_tensor_grads,
            recv_prev=True,
            recv_next=False,
            tensor_shapes=tensor_shapes,
            group_ids=[send_group_ids, recv_group_ids],
            config=config,
        )
        if config.timers is not None:
            config.timers('backward-send-forward-recv').stop()
    return input_tensor