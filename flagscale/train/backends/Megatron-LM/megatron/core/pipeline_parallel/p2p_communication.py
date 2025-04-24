# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

from typing import List, Optional, Tuple, Union

import torch
import torch.distributed

from megatron import core
from megatron.core import ModelParallelConfig
from megatron.core.parallel_state import (
    get_pipeline_model_parallel_group,
    get_pipeline_model_parallel_next_rank,
    get_pipeline_model_parallel_prev_rank,
    get_pipeline_model_parallel_rank,
    get_pipeline_model_parallel_world_size,
)
from flagscale.train import get_parallel_context  
from flagscale.train.hetero.parallel_context import ParallelContext

# Types
Shape = Union[List[int], torch.Size]

def _communicate_shapes(tensor_send_next, tensor_send_prev, recv_prev, recv_next, config):
    """Communicate tensor shapes between stages. Used to communicate
    tensor shapes before the actual tensor communication happens.
    This is required when the sequence lengths across micro batches
    are not uniform.

    Args:
        tensor_send_next: tensor to send to next rank (no tensor sent if
                          set to None).
        tensor_send_prev: tensor to send to prev rank (no tensor sent if
                          set to None).
        recv_prev: boolean for whether tensor should be received from
                   previous rank.
        recv_next: boolean for whether tensor should be received from
                   next rank.
    Returns:
        (recv_prev_shape, recv_next_shape)
    """

    recv_prev_shape_tensor = None
    recv_next_shape_tensor = None
    send_prev_shape_tensor = None
    send_next_shape_tensor = None
    if recv_prev:
        recv_prev_shape_tensor = torch.empty(
            (3), device=torch.cuda.current_device(), dtype=torch.int64
        )
    if recv_next:
        recv_next_shape_tensor = torch.empty(
            (3), device=torch.cuda.current_device(), dtype=torch.int64
        )
    if tensor_send_prev is not None:
        send_prev_shape_tensor = torch.tensor(
            tensor_send_prev.size(), device=torch.cuda.current_device(), dtype=torch.int64
        )
    if tensor_send_next is not None:
        send_next_shape_tensor = torch.tensor(
            tensor_send_next.size(), device=torch.cuda.current_device(), dtype=torch.int64
        )

    if config.use_ring_exchange_p2p:
        torch.distributed.ring_exchange(
            tensor_send_prev=send_prev_shape_tensor,
            tensor_recv_prev=recv_prev_shape_tensor,
            tensor_send_next=send_next_shape_tensor,
            tensor_recv_next=recv_next_shape_tensor,
            group=get_pipeline_model_parallel_group(),
        )
    else:
        ops = []
        if send_prev_shape_tensor is not None:
            send_prev_op = torch.distributed.P2POp(
                torch.distributed.isend,
                send_prev_shape_tensor,
                get_pipeline_model_parallel_prev_rank(),
            )
            ops.append(send_prev_op)
        if recv_prev_shape_tensor is not None:
            recv_prev_op = torch.distributed.P2POp(
                torch.distributed.irecv,
                recv_prev_shape_tensor,
                get_pipeline_model_parallel_prev_rank(),
            )
            ops.append(recv_prev_op)
        if send_next_shape_tensor is not None:
            send_next_op = torch.distributed.P2POp(
                torch.distributed.isend,
                send_next_shape_tensor,
                get_pipeline_model_parallel_next_rank(),
            )
            ops.append(send_next_op)
        if recv_next_shape_tensor is not None:
            recv_next_op = torch.distributed.P2POp(
                torch.distributed.irecv,
                recv_next_shape_tensor,
                get_pipeline_model_parallel_next_rank(),
            )
            ops.append(recv_next_op)
        if len(ops) > 0:
            reqs = torch.distributed.batch_isend_irecv(ops)
            for req in reqs:
                req.wait()

        # To protect against race condition when using batch_isend_irecv().
        # should take this out once the bug with batch_isend_irecv is resolved.
        torch.cuda.synchronize()

    recv_prev_shape = [0, 0, 0]
    if recv_prev_shape_tensor is not None:
        recv_prev_shape = recv_prev_shape_tensor.tolist()

    recv_next_shape = [0, 0, 0]
    if recv_next_shape_tensor is not None:
        recv_next_shape = recv_next_shape_tensor.tolist()

    return recv_prev_shape, recv_next_shape


def _batched_p2p_ops(
    *,
    tensor_send_prev: Optional[torch.Tensor],
    tensor_recv_prev: Optional[torch.Tensor],
    tensor_send_next: Optional[torch.Tensor],
    tensor_recv_next: Optional[torch.Tensor],
    group: torch.distributed.ProcessGroup,
    prev_pipeline_rank: int,
    next_pipeline_rank: int,
):
    ops = []

    if tensor_send_prev is not None:
        send_prev_op = torch.distributed.P2POp(
            torch.distributed.isend, tensor_send_prev, prev_pipeline_rank, group
        )
        ops.append(send_prev_op)
    if tensor_recv_prev is not None:
        recv_prev_op = torch.distributed.P2POp(
            torch.distributed.irecv, tensor_recv_prev, prev_pipeline_rank, group
        )
        ops.append(recv_prev_op)
    if tensor_send_next is not None:
        send_next_op = torch.distributed.P2POp(
            torch.distributed.isend, tensor_send_next, next_pipeline_rank, group
        )
        ops.append(send_next_op)
    if tensor_recv_next is not None:
        recv_next_op = torch.distributed.P2POp(
            torch.distributed.irecv, tensor_recv_next, next_pipeline_rank, group
        )
        ops.append(recv_next_op)
    if len(ops) > 0:
        reqs = torch.distributed.batch_isend_irecv(ops)
    else:
        reqs = []
    return reqs


def _p2p_ops(
    *,
    tensor_send_prev: Optional[torch.Tensor],
    tensor_recv_prev: Optional[torch.Tensor],
    tensor_send_next: Optional[torch.Tensor],
    tensor_recv_next: Optional[torch.Tensor],
    group: torch.distributed.ProcessGroup,
    prev_pipeline_rank: int,
    next_pipeline_rank: int,
):
    reqs = {}
    even_send_odd_recv_group = group
    if (
        get_pipeline_model_parallel_world_size() == 2
        and torch.distributed.get_backend(group) != 'ucc'
    ):
        # Use the global process group for one of the two p2p communications
        # to allow the overlap of the independent communications.
        # Using the global process group is compatible because the pipeline-parallel
        # communications set the source and destination by global rank.
        # The only exception occurs when using the ‘ucc’ backend.
        # Because the global communicator always uses the ‘nccl’ backend,
        # we must ensure the else path is followed for the ‘ucc’ backend.
        even_recv_odd_send_group = torch.distributed.group.WORLD
        print("DEBUG: Using global process group for even_recv_odd_send_group")
    else:
        even_recv_odd_send_group = group
        print("DEBUG: Using the same process group for all communications")

    if get_pipeline_model_parallel_rank() % 2 == 0:
        if tensor_send_next is not None:
            print(f"DEBUG: Sending to next rank {next_pipeline_rank}")
            send_next_req = torch.distributed.isend(
                tensor=tensor_send_next, dst=next_pipeline_rank, group=even_send_odd_recv_group
            )
            reqs["send_next"] = send_next_req
            print("DEBUG: Issued send_next request")

        if tensor_recv_prev is not None:
            print(f"DEBUG: Receiving from previous rank {prev_pipeline_rank}")
            recv_prev_req = torch.distributed.irecv(
                tensor=tensor_recv_prev, src=prev_pipeline_rank, group=even_recv_odd_send_group
            )
            reqs["recv_prev"] = recv_prev_req
            print("DEBUG: Issued recv_prev request")

        if tensor_send_prev is not None:
            print(f"DEBUG: Sending to previous rank {prev_pipeline_rank}")
            send_prev_req = torch.distributed.isend(
                tensor=tensor_send_prev, dst=prev_pipeline_rank, group=even_send_odd_recv_group
            )
            reqs["send_prev"] = send_prev_req
            print("DEBUG: Issued send_prev request")

        if tensor_recv_next is not None:
            print(f"DEBUG: Receiving from next rank {next_pipeline_rank}")
            recv_next_req = torch.distributed.irecv(
                tensor=tensor_recv_next, src=next_pipeline_rank, group=even_recv_odd_send_group
            )
            reqs["recv_next"] = recv_next_req
            print("DEBUG: Issued recv_next request")

    else:
        print("DEBUG: Odd rank communication pattern")
        if tensor_recv_prev is not None:
            print(f"DEBUG: Receiving from previous rank {prev_pipeline_rank}")
            recv_prev_req = torch.distributed.irecv(
                tensor=tensor_recv_prev, src=prev_pipeline_rank, group=even_send_odd_recv_group
            )
            reqs["recv_prev"] = recv_prev_req
            print("DEBUG: Issued recv_prev request")

        if tensor_send_next is not None:
            print(f"DEBUG: Sending to next rank {next_pipeline_rank}")
            send_next_req = torch.distributed.isend(
                tensor=tensor_send_next, dst=next_pipeline_rank, group=even_recv_odd_send_group
            )
            reqs["send_next"] = send_next_req
            print("DEBUG: Issued send_next request")

        if tensor_recv_next is not None:
            print(f"DEBUG: Receiving from next rank {next_pipeline_rank}")
            recv_next_req = torch.distributed.irecv(
                tensor=tensor_recv_next, src=next_pipeline_rank, group=even_send_odd_recv_group
            )
            reqs["recv_next"] = recv_next_req
            print("DEBUG: Issued recv_next request")

        if tensor_send_prev is not None:
            print(f"DEBUG: Sending to previous rank {prev_pipeline_rank}")
            send_prev_req = torch.distributed.isend(
                tensor=tensor_send_prev, dst=prev_pipeline_rank, group=even_recv_odd_send_group
            )
            reqs["send_prev"] = send_prev_req
            print("DEBUG: Issued send_prev request")
            
    print(f"DEBUG: Exiting _p2p_ops function with {len(reqs)} pending requests")
    return reqs

def _communicate(
    *,
    tensor_send_next: Optional[torch.Tensor],
    tensor_send_prev: Optional[torch.Tensor],
    recv_prev: bool,
    recv_next: bool,
    tensor_shape: Shape,
    config: ModelParallelConfig,
    wait_on_reqs: bool = True,
    group: torch.distributed.ProcessGroup = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Communicate tensors between stages. Used as helper method in other
    communication methods that are used in megatron/schedules.py.

    Args:
        tensor_send_next (torch.Tensor, optional):
            Tensor to send to next rank (no tensor sent if None)

        tensor_send_prev (torch.Tensor, optional):
            Tensor to send to prev rank (no tensor sent if None)

        recv_prev (boolean, required):
            whether tensor should be received from previous rank.

        recv_next (boolean, required):
            whether tensor should be received from next rank.

        tensor_shape (List[int] or torch.Size, required):
            shape of tensor to receive (this method assumes that all
            tensors sent and received in a single function call are
            the same shape).

        wait_on_reqs (boolean, optional, default=False):
            For non-batched p2p communication, wait on each request
            before returning.

    Returns:
        tuple containing

        - tensor_recv_prev: torch.Tensor if recv_prev is True, None otherwise.
        - tensor_recv_next: torch.Tensor if recv_next is True, None otherwise.

    """

    tensor_recv_prev_func = None
    tensor_recv_next_func = None

    if not config.variable_seq_lengths:
        recv_prev_shape = tensor_shape
        recv_next_shape = tensor_shape
        print(f"DEBUG: Using fixed tensor shapes: {tensor_shape}")
    else:
        print("DEBUG: Communicating shapes for variable sequence lengths")
        recv_prev_shape, recv_next_shape = _communicate_shapes(
            tensor_send_next, tensor_send_prev, recv_prev, recv_next, config
        )
        print(f"DEBUG: Received shapes - prev: {recv_prev_shape}, next: {recv_next_shape}")

    def create_tensor_recv_prev():
        print(f"DEBUG: Creating tensor_recv_prev with shape {recv_prev_shape}")
        return torch.empty(
            recv_prev_shape,
            requires_grad=True,
            device=torch.cuda.current_device() if "cpu:gloo" != group.name() else torch.device("cpu"),
            dtype=config.pipeline_dtype,
        )

    def create_tensor_recv_next():
        print(f"DEBUG: Creating tensor_recv_next with shape {recv_next_shape}")
        return torch.empty(
            recv_next_shape,
            requires_grad=True,
            device=torch.cuda.current_device() if "cpu:gloo" != group.name() else torch.device("cpu"),
            dtype=config.pipeline_dtype,
        )

    if recv_prev:
        if config.pipeline_dtype is None:
            raise RuntimeError("pipeline_dtype must be provided if recv_prev is True")
        if tensor_shape is None:
            raise RuntimeError(
                "tensor_shape must be specified if recv_prev is True. "
                "Common tensor_shape is (seq_length, micro_batch_size, hidden_size)"
            )
        tensor_recv_prev_func = create_tensor_recv_prev
        print("DEBUG: Will receive from previous rank")

    if recv_next:
        if config.pipeline_dtype is None:
            raise RuntimeError("dtype must be provided if recv_next is True")
        if tensor_shape is None:
            raise RuntimeError(
                "tensor_shape must be specified if recv_next is True. "
                "Common tensor_shape is (seq_length, micro_batch_size, hidden_size)"
            )
        tensor_recv_next_func = create_tensor_recv_next
        print("DEBUG: Will receive from next rank")

    # Send tensors in both the forward and backward directions as appropriate.
    if config.use_ring_exchange_p2p:
        print("DEBUG: Using ring_exchange for p2p communication")

        def _ring_exchange_wrapper(**kwargs):
            print("DEBUG: Executing ring_exchange")
            torch.distributed.ring_exchange(**kwargs)
            return []

        p2p_func = _ring_exchange_wrapper
    elif config.batch_p2p_comm:
        print("DEBUG: Using batched p2p communication")
        assert wait_on_reqs
        p2p_func = _batched_p2p_ops
    else:
        print("DEBUG: Using regular p2p communication")
        p2p_func = _p2p_ops

    # Each rank can now be part of several different pipeline parallel groups
    # (specifically, this can occur when encoder tensor parallelism != decoder
    # tensor parallelism, and hence a rank in the encoder is going to feed
    # several different decoder ranks. We therefore have to receive or send tensors
    # from several groups. For convenience, I wrap everything into lists.
    if config.enable_hetero: # Using the passed 'group' in the case of 'enable_hetero'
        print("DEBUG: Using heterogeneous mode with provided group")
        pp_group = group
        next_rank = get_pipeline_model_parallel_next_rank(group=group)
        prev_rank = get_pipeline_model_parallel_prev_rank(group=group)
    else:
        print("DEBUG: Using default pipeline parallel group")
        pp_group = get_pipeline_model_parallel_group()
        next_rank = get_pipeline_model_parallel_next_rank()
        prev_rank = get_pipeline_model_parallel_prev_rank()
    print("pp_group: ",pp_group)
    print("next_rank: ",next_rank)
    print("prev_rank: ",prev_rank)
    if not isinstance(pp_group, list):
        pp_group = [pp_group]
        assert not isinstance(next_rank, list)
        next_rank = [next_rank]
        assert not isinstance(prev_rank, list)
        prev_rank = [prev_rank]
        print("DEBUG: Converted single group/ranks to lists")

    if config.use_ring_exchange_p2p or config.batch_p2p_comm:
        reqs = []
    else:
        reqs = {}
    tensor_recv_prev_list = []
    tensor_recv_next_list = []

    for group, nr, pr in zip(pp_group, next_rank, prev_rank):
        if tensor_recv_prev_func is not None:
            tensor_recv_prev = tensor_recv_prev_func()
            tensor_recv_prev_list.append(tensor_recv_prev)
            print(f"DEBUG: Created tensor_recv_prev for group {nr} {pr}")
        else:
            tensor_recv_prev = None

        if tensor_recv_next_func is not None:
            tensor_recv_next = tensor_recv_next_func()
            tensor_recv_next_list.append(tensor_recv_next)
            print(f"DEBUG: Created tensor_recv_next for group {nr} {pr}")
        else:
            tensor_recv_next = None

        print("DEBUG: Calling p2p communication function")
        p2p_reqs = p2p_func(
            tensor_send_prev=tensor_send_prev,
            tensor_recv_prev=tensor_recv_prev,
            tensor_send_next=tensor_send_next,
            tensor_recv_next=tensor_recv_next,
            group=group,
            prev_pipeline_rank=pr,
            next_pipeline_rank=nr,
        )
        if isinstance(p2p_reqs, list):
            reqs.extend(p2p_reqs)
            print(f"DEBUG: Added {len(p2p_reqs)} requests to list")
        else:
            reqs.update(p2p_reqs)
            print(f"DEBUG: Added {len(p2p_reqs)} requests to dict")

    if wait_on_reqs and len(reqs) > 0:
        print(f"DEBUG: Waiting on {len(reqs)} requests")
        for req in reqs if isinstance(reqs, list) else reqs.values():
            req.wait()
        reqs = None
        print("DEBUG: All requests completed")

    if (
        (config.batch_p2p_comm and config.batch_p2p_sync)
        # The lists below have a size > 1 only when ETP ≠ DTP,
        # meaning this synchronization is required when ETP ≠ DTP.
        or len(tensor_recv_prev_list) > 1
        or len(tensor_recv_next_list) > 1
    ):
        print("DEBUG: Synchronizing CUDA for batch p2p communication")
        # To protect against race condition when using batch_isend_irecv().
        # User should assert that we have a modern enough PyTorch to not need this
        torch.cuda.synchronize()

    def _handle_tensor_list(x):
        """This basically handles all the cases that we expect to see. Either the list None,
        or it's a singleton (the usual cases, since most ranks only belong to one pipeline group),
        or everything returned is None, or everything returned is not None, and it has to be summed
        together."""
        if len(x) == 0:
            print("DEBUG: Empty tensor list")
            return None
        if len(x) == 1:
            print("DEBUG: Singleton tensor list")
            return x[0]
        if all(xx is None for xx in x):
            print("DEBUG: All None tensor list")
            return None
        # When the encoder's TP size differs from the decoder's TP size
        # (with the constraint `encoder_tp_size <= decoder_tp_size`), each encoder TP rank
        # may receive multiple gradients from corresponding decoder TP ranks.
        # For example, if `ETP=1` and `DTP=2`, then encoder rank 0 will receive gradients
        # from decoder ranks 1 and 2. These received gradients must be averaged.
        print(f"DEBUG: Stacking and summing {len(x)} tensors")
        return torch.stack(x, dim=0).mean(dim=0, dtype=torch.float32).to(x[0].dtype)

    print("DEBUG: Processing tensor_recv_prev_list")
    tensor_recv_prev = _handle_tensor_list(tensor_recv_prev_list)
    print("DEBUG: Processing tensor_recv_next_list")
    tensor_recv_next = _handle_tensor_list(tensor_recv_next_list)

    print("DEBUG: Exiting _communicate function")
    return tensor_recv_prev, tensor_recv_next, reqs

def warm_up_comm_group(config):
    if config.enable_hetero:
        from flagscale.train.hetero.p2p_communication import warm_up_comm_group_hetero
        warm_up_comm_group_hetero(config)

def recv_forward(tensor_shape: Shape, config: ModelParallelConfig) -> torch.Tensor:
    """Receive tensor from previous rank in pipeline (forward receive).

    See _communicate for argument details.
    """
    if config.enable_hetero:
        from flagscale.train.hetero.p2p_communication import recv_forward_hetero
        return recv_forward_hetero(tensor_shape, config)

    if core.parallel_state.is_pipeline_first_stage():
        input_tensor = None
    else:
        if config.timers is not None:
            config.timers('forward-recv', log_level=2).start()
        input_tensor, _, _ = _communicate(
            tensor_send_next=None,
            tensor_send_prev=None,
            recv_prev=True,
            recv_next=False,
            tensor_shape=tensor_shape,
            config=config,
        )
        if config.timers is not None:
            config.timers('forward-recv').stop()
    return input_tensor


def recv_backward(tensor_shape: Shape, config: ModelParallelConfig) -> torch.Tensor:
    """Receive tensor from next rank in pipeline (backward receive).

    See _communicate for argument details.
    """
    if config.enable_hetero:
        from flagscale.train.hetero.p2p_communication import recv_backward_hetero
        return recv_backward_hetero(tensor_shape, config)

    if core.parallel_state.is_pipeline_last_stage():
        output_tensor_grad = None
    else:
        if config.timers is not None:
            config.timers('backward-recv', log_level=2).start()
        _, output_tensor_grad, _ = _communicate(
            tensor_send_next=None,
            tensor_send_prev=None,
            recv_prev=False,
            recv_next=True,
            tensor_shape=tensor_shape,
            config=config,
        )
        if config.timers is not None:
            config.timers('backward-recv').stop()
    return output_tensor_grad


def send_forward(output_tensor: torch.Tensor, config: ModelParallelConfig) -> None:
    """Send tensor to next rank in pipeline (forward send).

    See _communicate for argument details.
    """
    if config.enable_hetero:
        from flagscale.train.hetero.p2p_communication import send_forward_hetero
        send_forward_hetero(output_tensor, config)
        return

    if not core.parallel_state.is_pipeline_last_stage():
        if config.timers is not None:
            config.timers('forward-send', log_level=2).start()
        _communicate(
            tensor_send_next=output_tensor,
            tensor_send_prev=None,
            recv_prev=False,
            recv_next=False,
            tensor_shape=None,
            config=config,
        )
        if config.timers is not None:
            config.timers('forward-send').stop()


def send_backward(input_tensor_grad: torch.Tensor, config: ModelParallelConfig) -> None:
    """Send tensor to previous rank in pipeline (backward send).

    See _communicate for argument details.
    """
    if config.enable_hetero:
        from flagscale.train.hetero.p2p_communication import send_backward_hetero
        send_backward_hetero(input_tensor_grad, config)
        return

    if not core.parallel_state.is_pipeline_first_stage():
        if config.timers is not None:
            config.timers('backward-send', log_level=2).start()
        _communicate(
            tensor_send_next=None,
            tensor_send_prev=input_tensor_grad,
            recv_prev=False,
            recv_next=False,
            tensor_shape=None,
            config=config,
        )
        if config.timers is not None:
            config.timers('backward-send').stop()


def send_forward_recv_backward(
    output_tensor: torch.Tensor, tensor_shape: Shape, config: ModelParallelConfig
) -> torch.Tensor:
    """Batched send and recv with next rank in pipeline.

    See _communicate for argument details.
    """
    if config.enable_hetero:
        from flagscale.train.hetero.p2p_communication import send_forward_recv_backward_hetero
        return send_forward_recv_backward_hetero(output_tensor, tensor_shape, config)

    if core.parallel_state.is_pipeline_last_stage():
        output_tensor_grad = None
    else:
        if config.timers is not None:
            config.timers('forward-send-backward-recv', log_level=2).start()
        _, output_tensor_grad, _ = _communicate(
            tensor_send_next=output_tensor,
            tensor_send_prev=None,
            recv_prev=False,
            recv_next=True,
            tensor_shape=tensor_shape,
            config=config,
        )
        if config.timers is not None:
            config.timers('forward-send-backward-recv').stop()
    return output_tensor_grad


def send_backward_recv_forward(
    input_tensor_grad: torch.Tensor, tensor_shape: Shape, config: ModelParallelConfig
) -> torch.Tensor:
    """Batched send and recv with previous rank in pipeline.

    See _communicate for argument details.
    """
    if config.enable_hetero:
        from flagscale.train.hetero.p2p_communication import send_backward_recv_forward_hetero
        return send_backward_recv_forward_hetero(input_tensor_grad, tensor_shape, config)

    if core.parallel_state.is_pipeline_first_stage():
        input_tensor = None
    else:
        if config.timers is not None:
            config.timers('backward-send-forward-recv', log_level=2).start()
        input_tensor, _, _ = _communicate(
            tensor_send_next=None,
            tensor_send_prev=input_tensor_grad,
            recv_prev=True,
            recv_next=False,
            tensor_shape=tensor_shape,
            config=config,
        )
        if config.timers is not None:
            config.timers('backward-send-forward-recv').stop()
    return input_tensor


def send_forward_recv_forward(
    output_tensor: torch.Tensor,
    recv_prev: bool,
    tensor_shape: Shape,
    config: ModelParallelConfig,
    overlap_p2p_comm: bool = False,
) -> torch.Tensor:
    """Batched recv from previous rank and send to next rank in pipeline.

    See _communicate for argument details.
    """
    if config.timers is not None:
        config.timers('forward-send-forward-recv', log_level=2).start()
    input_tensor, _, wait_handles = _communicate(
        tensor_send_next=output_tensor,
        tensor_send_prev=None,
        recv_prev=recv_prev,
        recv_next=False,
        tensor_shape=tensor_shape,
        wait_on_reqs=(not overlap_p2p_comm),
        config=config,
    )
    if config.timers is not None:
        config.timers('forward-send-forward-recv').stop()
    if overlap_p2p_comm:
        return input_tensor, wait_handles
    return input_tensor


def send_backward_recv_backward(
    input_tensor_grad: torch.Tensor,
    recv_next: bool,
    tensor_shape: Shape,
    config: ModelParallelConfig,
    overlap_p2p_comm: bool = False,
) -> torch.Tensor:
    """Batched recv from next rank and send to previous rank in pipeline.

    See _communicate for argument details.
    """
    if config.timers is not None:
        config.timers('backward-send-backward-recv', log_level=2).start()
    _, output_tensor_grad, wait_handles = _communicate(
        tensor_send_next=None,
        tensor_send_prev=input_tensor_grad,
        recv_prev=False,
        recv_next=recv_next,
        tensor_shape=tensor_shape,
        wait_on_reqs=(not overlap_p2p_comm),
        config=config,
    )
    if config.timers is not None:
        config.timers('backward-send-backward-recv').stop()
    if overlap_p2p_comm:
        return output_tensor_grad, wait_handles
    return output_tensor_grad


def send_forward_backward_recv_forward_backward(
    output_tensor: torch.Tensor,
    input_tensor_grad: torch.Tensor,
    recv_prev: bool,
    recv_next: bool,
    tensor_shape: Shape,
    config: ModelParallelConfig,
) -> torch.Tensor:
    """Batched send and recv with previous and next ranks in pipeline.

    See _communicate for argument details.
    """
    if config.timers is not None:
        config.timers('forward-backward-send-forward-backward-recv', log_level=2).start()
    input_tensor, output_tensor_grad, _ = _communicate(
        tensor_send_next=output_tensor,
        tensor_send_prev=input_tensor_grad,
        recv_prev=recv_prev,
        recv_next=recv_next,
        tensor_shape=tensor_shape,
        config=config,
    )
    if config.timers is not None:
        config.timers('forward-backward-send-forward-backward-recv').stop()
    return input_tensor, output_tensor_grad
