# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

import contextlib
from typing import Callable, Iterator, List, Optional, Union

import torch
from torch.autograd.variable import Variable

#from flagscale.hetero import parallel_state
from megatron.core import parallel_state
from megatron.core.enums import ModelType
from flagscale.hetero import p2p_communication
from megatron.core.transformer.moe.router import MoEAuxLossAutoScaler
from megatron.core.utils import get_attr_wrapped_model, get_model_config, get_model_type
from megatron.core.pipeline_parallel.schedules import (
    forward_backward_pipelining_with_interleaving,
    forward_backward_pipelining_without_interleaving,
    forward_backward_no_pipelining,
    forward_step,
    backward_step,
    check_first_val_step,
    deallocate_output_tensor
    )

# Types
Shape = Union[List[int], torch.Size]

def get_forward_backward_func():
    """Retrieves the appropriate forward_backward function given the
    configuration of parallel_state.

    Returns a function that will perform all of the forward and
    backward passes of the model given the pipeline model parallel
    world size and virtual pipeline model parallel world size in the
    global parallel_state.

    Note that if using sequence parallelism, the sequence length component of
    the tensor shape is updated to original_sequence_length /
    tensor_model_parallel_world_size.

    The function returned takes the following arguments:

    forward_step_func (required): A function that takes a data
        iterator and a model as its arguments and return the model's
        forward output and the loss function. The loss function should
        take one torch.Tensor and return a torch.Tensor of loss and a
        dictionary of string -> torch.Tensor.

        A third argument, checkpoint_activations_microbatch, indicates
        that the activations for this microbatch should be
        checkpointed. A None value for this argument indicates that
        the default from the configuration should be used. This is
        used when the
        num_microbatches_with_partial_activation_checkpoints is used.

        For example:

        def loss_func(loss_mask, output_tensor):
            losses = output_tensor.float()
            loss_mask = loss_mask.view(-1).float()
            loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()

            # Reduce loss for logging.
            averaged_loss = average_losses_across_data_parallel_group([loss])

            return loss, {'lm loss': averaged_loss[0]}

        def forward_step(data_iterator, model):
            data, loss_mask = next(data_iterator)
            output = model(data)
            return output, partial(loss_func, loss_mask)


        forward_backward_func(forward_step_func=forward_step, ...)


    data_iterator (required): an iterator over the data, will be
        passed as is to forward_step_func. Expected to be a list of
        iterators in the case of interleaved pipeline parallelism.

    model (required): the actual model. Expected to be a list of modules in the case of interleaved
        pipeline parallelism. Must be a (potentially wrapped) megatron.core.models.MegatronModule.

    num_microbatches (int, required):
        The number of microbatches to go through

    seq_length (int, required): Sequence length of the current global batch. If this is a dual-stack
        transformer, this is the encoder's sequence length. This is ignored if variable_seq_lengths
        in the config is True. Otherwise, each microbatch in the current global batch size must use
        this sequence length.

    micro_batch_size (int, required): The number of sequences in a microbatch.

    decoder_seq_length (int, optional): The sequence length for the decoder in a dual-stack
        transformer. This is ignored for a single-stack transformer.

    forward_only (optional, default = False): Perform only the forward step

    collect_non_loss_data (optional, bool, default=False): TODO

    first_val_step (bool, optional): Is the first step of the validation phase. Used by
        Transformer Engine modules to only update their fp8 weights only on the first validation step.

    """
    pipeline_model_parallel_size = parallel_state.get_pipeline_model_parallel_world_size()
    if pipeline_model_parallel_size > 1:
        if isinstance(parallel_state.get_pipeline_model_parallel_group(), list):
            assert parallel_state.get_virtual_pipeline_model_parallel_world_size() == None, \
                'vp not supported for hetero tp mode'
            forward_backward_func = forward_backward_pipelining_without_interleaving_hetero
        else:
            if parallel_state.get_virtual_pipeline_model_parallel_world_size() is not None:
                forward_backward_func = forward_backward_pipelining_with_interleaving
            else:
                forward_backward_func = forward_backward_pipelining_without_interleaving
    else:
        forward_backward_func = forward_backward_no_pipelining
    return forward_backward_func

def get_tp_hetero_tensor_shapes(
    *,
    send_recv: bool,
    model_type: ModelType,
    seq_length: int,
    micro_batch_size: int,
    decoder_seq_length: int,
    config,
):
    # send: True, recv: False
    assert model_type == ModelType.encoder_or_decoder, \
        'Only support encoder or decoder model type for tp hetero mode for now!'
    #TODO: cp support
    
    tensor_shapes = []
    tp_size_of_each_pipeline_stage = parallel_state.get_tensor_model_parallel_size_of_each_pipeline_stage()
    pipeline_rank = parallel_state.get_pipeline_model_parallel_rank()
    pipeline_size = parallel_state.get_pipeline_model_parallel_world_size()
    tp_size_of_current_pipline_rank = tp_size_of_each_pipeline_stage[pipeline_rank]    
    
    # Send
    if send_recv:
        tp_size_of_next_pipeline_rank = tp_size_of_each_pipeline_stage[(pipeline_rank + 1) % pipeline_size]
        tp_scale = tp_size_of_current_pipline_rank / tp_size_of_next_pipeline_rank
        if config.sequence_parallel:
            if tp_size_of_current_pipline_rank == tp_size_of_next_pipeline_rank:
                seq_length = seq_length // tp_size_of_current_pipline_rank
                tensor_shapes.append((seq_length, micro_batch_size, config.hidden_size))
            elif tp_size_of_current_pipline_rank > tp_size_of_next_pipeline_rank:
                seq_length = seq_length // tp_size_of_current_pipline_rank
                tensor_shapes.append((seq_length, micro_batch_size, config.hidden_size))
            else:
                seq_length = seq_length // tp_size_of_next_pipeline_rank
                for i in range(tp_size_of_next_pipeline_rank // tp_size_of_current_pipline_rank):
                    tensor_shapes.append((seq_length, micro_batch_size, config.hidden_size))
        else:
            tensor_shapes.append((seq_length, micro_batch_size, config.hidden_size))
    # Recv
    else:
        tp_size_of_prev_pipeline_rank = tp_size_of_each_pipeline_stage[(pipeline_rank - 1) % pipeline_size]
        tp_scale = tp_size_of_prev_pipeline_rank / tp_size_of_current_pipline_rank
        if config.sequence_parallel:
            if tp_size_of_current_pipline_rank == tp_size_of_prev_pipeline_rank:
                seq_length = seq_length // tp_size_of_current_pipline_rank
                tensor_shapes.append((seq_length, micro_batch_size, config.hidden_size))
            elif tp_size_of_current_pipline_rank > tp_size_of_prev_pipeline_rank:
                seq_length = seq_length // tp_size_of_current_pipline_rank
                tensor_shapes.append((seq_length, micro_batch_size, config.hidden_size))
            else:
                seq_length = seq_length // tp_size_of_prev_pipeline_rank
                for i in range(tp_size_of_prev_pipeline_rank // tp_size_of_current_pipline_rank):
                    tensor_shapes.append((seq_length, micro_batch_size, config.hidden_size))
        else:
            tensor_shapes.append((seq_length, micro_batch_size, config.hidden_size))
    
    return tensor_shapes, tp_scale

def tp_hetero_recv_backward(send_tensor_shapes, tp_scale, config):
    output_tensor_grads = []
    if not parallel_state.is_pipeline_last_stage():
        if config.sequence_parallel:
            # match
            if int(tp_scale) == 1:
                output_tensor_grads = p2p_communication.tp_hetero_recv_backward(send_tensor_shapes, 1, config)
            # Fwd small tp -> large tp, Bwd small tp <- large tp
            elif tp_scale < 1:
                tmp_tensors = p2p_communication.tp_hetero_recv_backward(send_tensor_shapes, int(1 / tp_scale), config)
                output_tensor_grads.append(torch.cat(tmp_tensors, dim=0))
            else:
                output_tensor_grads = p2p_communication.tp_hetero_recv_backward(send_tensor_shapes, 1, config)
        else:
            output_tensor_grads = p2p_communication.tp_hetero_recv_backward(send_tensor_shapes, 1, config)
    else:
        output_tensor_grads.append(None)
    return output_tensor_grads
    
def tp_hetero_send_forward(output_tensors, tensor_shapes, tp_scale, config):
    if not parallel_state.is_pipeline_last_stage():
        if isinstance(output_tensors, list):
            output_tensor = output_tensors[0]
        else:
            output_tensor = output_tensors
            
        if config.sequence_parallel:
            # match
            if int(tp_scale) == 1:
                p2p_communication.tp_hetero_send_forward([output_tensor], 1, config)
            # small tp -> large tp
            elif tp_scale < 1:
                split_size = int(1 / tp_scale)
                tmp_tensors = list(torch.chunk(output_tensor, split_size, dim=0))
                p2p_communication.tp_hetero_send_forward(tmp_tensors, split_size, config)
            # large tp -> small tp
            else:
                p2p_communication.tp_hetero_send_forward([output_tensor], 1, config)
        else:
            # match
            if int(tp_scale) == 1:
                p2p_communication.tp_hetero_send_forward([output_tensor], 1, config)
            # small tp -> large tp
            elif tp_scale < 1:
                p2p_communication.tp_hetero_send_forward([output_tensor], int(1 / tp_scale), config)
            # large tp -> small tp
            else:
                tensor_rank = parallel_state.get_tensor_model_parallel_rank()
                if tensor_rank % (int(tp_scale)) == 0:
                    p2p_communication.tp_hetero_send_forward([output_tensor], 1, config)

    
def tp_hetero_recv_forward(recv_tensor_shapes, tp_scale, config):
    input_tensors = []
    if not parallel_state.is_pipeline_first_stage():
        if config.sequence_parallel:
            # match
            if int(tp_scale) == 1:
                input_tensors = p2p_communication.tp_hetero_recv_forward(recv_tensor_shapes, 1, config)
            # small tp -> large tp
            elif tp_scale < 1:
                input_tensors = p2p_communication.tp_hetero_recv_forward(recv_tensor_shapes, 1, config)
            # large tp -> small tp
            else:
                tmp_tensors = p2p_communication.tp_hetero_recv_forward(recv_tensor_shapes, int(tp_scale), config)
                input_tensors.append(torch.cat(tmp_tensors, dim=0))
        else:
            input_tensors = p2p_communication.tp_hetero_recv_forward(recv_tensor_shapes, 1, config)
    else:
        input_tensors.append(None)
    return input_tensors

def tp_hetero_send_backward(input_tensor_grads, tensor_shapes, tp_scale, config):
    if not parallel_state.is_pipeline_first_stage():        
        if isinstance(input_tensor_grads, list):
            input_tensor_grad = input_tensor_grads[0]
        else:
            input_tensor_grad = input_tensor_grads
            
        if config.sequence_parallel:
            # match
            if int(tp_scale) == 1:
                p2p_communication.tp_hetero_send_backward([input_tensor_grad], 1, config)
            # Fwd: small tp -> large tp, Bwd large tp -> small tp
            elif tp_scale < 1:
                p2p_communication.tp_hetero_send_backward([input_tensor_grad], 1, config)
            # Fwd: large tp -> small tp, Bwd small tp -> large tp
            else:
                split_size = int(tp_scale)
                tmp_tensors = list(torch.chunk(input_tensor_grad, split_size, dim=0))
                p2p_communication.tp_hetero_send_backward(tmp_tensors, split_size, config)
        else:
            # match
            if int(tp_scale) == 1:
                p2p_communication.tp_hetero_send_backward([input_tensor_grad], 1, config)
            # Fwd: small tp -> large tp, Bwd large tp -> small tp
            elif tp_scale < 1:
                tensor_rank = parallel_state.get_tensor_model_parallel_rank()
                if  tensor_rank % (int(1 / tp_scale)) == 0:
                    p2p_communication.tp_hetero_send_backward([input_tensor_grad], 1, config)
            # Fwd: large tp -> small tp, Bwd small tp -> large tp
            else:
                p2p_communication.tp_hetero_send_backward([input_tensor_grad], int(tp_scale), config)

def tp_hetero_send_forward_recv_backward(output_tensors, tensor_shapes, tp_scale, config):
    output_tensor_grads = []
    if not parallel_state.is_pipeline_last_stage():
        if isinstance(output_tensors, list):
            output_tensor = output_tensors[0]
        else:
            output_tensor = output_tensors
            
        if config.sequence_parallel:
            # match
            if int(tp_scale) == 1:
                output_tensor_grads = p2p_communication.tp_hetero_send_forward_recv_backward([output_tensor], tensor_shapes, 1, 1, config)
            # small tp -> large_tp
            elif tp_scale < 1:
                split_size = int(1 / tp_scale)
                tmp_tensors = list(torch.chunk(output_tensor, split_size, dim=0))
                recv_tensors = p2p_communication.tp_hetero_send_forward_recv_backward(tmp_tensors, tensor_shapes, split_size, split_size, config)
                output_tensor_grads.append(torch.cat(recv_tensors, dim=0))
            else:
                output_tensor_grads = p2p_communication.tp_hetero_send_forward_recv_backward([output_tensor], tensor_shapes, 1, 1, config)
        else:
            # match
            if int(tp_scale) == 1:
                output_tensor_grads = p2p_communication.tp_hetero_send_forward_recv_backward([output_tensor], tensor_shapes, 1, 1, config)
            # small tp -> large_tp
            elif tp_scale < 1:
                output_tensor_grads = p2p_communication.tp_hetero_send_forward_recv_backward([output_tensor], tensor_shapes, int(1 / tp_scale), 1, config)
            else:
                tensor_rank = parallel_state.get_tensor_model_parallel_rank()
                output_tensor_grads = p2p_communication.tp_hetero_send_forward_recv_backward([output_tensor], tensor_shapes, 
                                                                      1  if tensor_rank % (int(tp_scale)) == 0 else 0, 
                                                                      1, config)
    else:
        output_tensor_grads.append(None)
    return output_tensor_grads

def tp_hetero_send_backward_recv_forward(input_tensor_grads, tensor_shapes, tp_scale, config):
    input_tensors = []
    if not parallel_state.is_pipeline_first_stage():
        if isinstance(input_tensor_grads, list):
            input_tensor_grad = input_tensor_grads[0]
        else:
            input_tensor_grad = input_tensor_grads
            
        if config.sequence_parallel:
            # match
            if int(tp_scale) == 1:
                input_tensors = p2p_communication.tp_hetero_send_backward_recv_forward([input_tensor_grad], tensor_shapes, 1, 1, config)
            # small tp -> large tp
            elif tp_scale < 1:
                input_tensors = p2p_communication.tp_hetero_send_backward_recv_forward([input_tensor_grad], tensor_shapes, 1, 1, config)
            # large tp -> small tp
            else:
                split_size = int(tp_scale)
                tmp_tensors = list(torch.chunk(input_tensor_grad, split_size, dim=0))
                recv_tensors = p2p_communication.tp_hetero_send_backward_recv_forward(tmp_tensors, tensor_shapes, split_size, split_size, config)
                input_tensors.append(torch.cat(recv_tensors, dim=0))
        else:
            # match
            if int(tp_scale) == 1:
                input_tensors = p2p_communication.tp_hetero_send_backward_recv_forward([input_tensor_grad], tensor_shapes, 1, 1, config)
            # small tp -> large tp
            elif tp_scale < 1:
                tensor_rank = parallel_state.get_tensor_model_parallel_rank()
                input_tensors = p2p_communication.tp_hetero_send_backward_recv_forward([input_tensor_grad], tensor_shapes,
                                                                    1 if tensor_rank % (int(1 / tp_scale)) == 0 else 0, 
                                                                    1, config)
            # large tp -> small tp
            else:
                input_tensors = p2p_communication.tp_hetero_send_backward_recv_forward([input_tensor_grad], tensor_shapes, int(tp_scale), 1, config)
    else:
        input_tensors.append(None)
    return input_tensors

def forward_backward_pipelining_without_interleaving_hetero(
    *,
    forward_step_func,
    data_iterator: Union[Iterator, List[Iterator]],
    model: Union[torch.nn.Module, List[torch.nn.Module]],
    num_microbatches: int,
    seq_length: int,
    micro_batch_size: int,
    decoder_seq_length: int = None,
    forward_only: bool = False,
    collect_non_loss_data: bool = False,
    first_val_step: bool = None,
):
    """Run non-interleaved 1F1B schedule, with communication between pipeline
    stages.

    Returns dictionary with losses if the last stage, empty dict otherwise."""

    if isinstance(model, list):
        assert (
            len(model) == 1
        ), "non-interleaved pipeline parallelism does not support model chunking"
        model = model[0]
    if isinstance(data_iterator, list):
        assert (
            len(data_iterator) == 1
        ), "non-pipeline-parallel schedule does not support model chunking"
        data_iterator = data_iterator[0]

    config = get_model_config(model)
    if config.overlap_p2p_comm:
        raise ValueError(
            "Non-interleaved pipeline parallelism does not support overlapping p2p communication"
        )

    if config.timers is not None:
        config.timers('forward-backward', log_level=1).start(barrier=config.barrier_with_L1_time)

    # Disable async grad reductions
    no_sync_func = config.no_sync_func
    if no_sync_func is None:
        no_sync_func = contextlib.nullcontext
    no_sync_context = None

    def disable_grad_sync():
        """Disable asynchronous grad reductions"""
        nonlocal no_sync_context
        if no_sync_context is None:
            no_sync_context = no_sync_func()
            no_sync_context.__enter__()

    def enable_grad_sync():
        """Enable asynchronous grad reductions"""
        nonlocal no_sync_context
        if no_sync_context is not None:
            no_sync_context.__exit__(None, None, None)
            no_sync_context = None

    disable_grad_sync()

    # Compute number of warmup microbatches.
    num_warmup_microbatches = (
        parallel_state.get_pipeline_model_parallel_world_size()
        - parallel_state.get_pipeline_model_parallel_rank()
        - 1
    )
    num_warmup_microbatches = min(num_warmup_microbatches, num_microbatches)
    num_microbatches_remaining = num_microbatches - num_warmup_microbatches

    # Checkpoint the activations of partial Transformer layers in a number of micro-batches
    # within the maximum outstanding micro-batch backpropagations.
    # Micro-batches with the ids less than 'num_microbatches_with_partial_activation_checkpoints'
    # checkpoint partial Transformer layers (or skip checkpointing) and
    # the rest of micro-batches within a window of micro-batches checkpoint
    # all Transformer layers. The window of micro-batches is set by the maximum
    # outstanding backpropagations and becomes smaller at later pipeline stages.
    # Please refer the appendix C in https://arxiv.org/pdf/2205.05198.pdf
    max_outstanding_backprops = None
    if config.num_microbatches_with_partial_activation_checkpoints is not None:
        max_outstanding_backprops = num_warmup_microbatches + 1

    model_type = get_model_type(model)

    rank = parallel_state.get_pipeline_model_parallel_rank()
    
    send_tensor_shapes, send_tp_scale = get_tp_hetero_tensor_shapes(
        send_recv=True,
        model_type=model_type,
        seq_length=seq_length,
        micro_batch_size=micro_batch_size,
        decoder_seq_length=decoder_seq_length,
        config=config,
    )
    
    recv_tensor_shapes, recv_tp_scale = get_tp_hetero_tensor_shapes(
        send_recv=False,
        model_type=model_type,
        seq_length=seq_length,
        micro_batch_size=micro_batch_size,
        decoder_seq_length=decoder_seq_length,
        config=config,
    )
    
    # Input, output tensors only need to be saved when doing backward passes
    input_tensors = None
    output_tensors = None
    total_num_tokens = torch.tensor(0, dtype=torch.int).cuda()
    
    if not forward_only:
        input_tensors = []
        output_tensors = []
    forward_data_store = []

    # Run warmup forward passes.
    for i in range(num_warmup_microbatches):
        # Decide to checkpoint all layers' activations of the current micro-batch
        if max_outstanding_backprops is not None:
            checkpoint_activations_microbatch = (
                i % max_outstanding_backprops
                >= config.num_microbatches_with_partial_activation_checkpoints
            )
        else:
            checkpoint_activations_microbatch = None

        input_tensor = tp_hetero_recv_forward(recv_tensor_shapes, recv_tp_scale, config)
        output_tensor, num_tokens = forward_step(
            forward_step_func,
            data_iterator,
            model,
            num_microbatches,
            input_tensor,
            forward_data_store,
            config,
            collect_non_loss_data,
            checkpoint_activations_microbatch,
            check_first_val_step(first_val_step, forward_only, i == 0),
            current_microbatch=i,
        )
        tp_hetero_send_forward(output_tensor, send_tensor_shapes, send_tp_scale, config)
        total_num_tokens += num_tokens.item()

        if not forward_only:
            input_tensors.append(input_tensor)
            output_tensors.append(output_tensor)
            deallocate_output_tensor(output_tensor[0], config.deallocate_pipeline_outputs)

    # Before running 1F1B, need to receive first forward tensor.
    # If all microbatches are run in warmup / cooldown phase, then no need to
    # receive this tensor here.
    if num_microbatches_remaining > 0:
        input_tensor = tp_hetero_recv_forward(recv_tensor_shapes, recv_tp_scale, config)

    # Run 1F1B in steady state.
    for i in range(num_microbatches_remaining):
        last_iteration = i == (num_microbatches_remaining - 1)

        # Decide to checkpoint all layers' activations of the current micro-batch
        if max_outstanding_backprops is not None:
            checkpoint_activations_microbatch = (
                (i + num_warmup_microbatches) % max_outstanding_backprops
            ) >= config.num_microbatches_with_partial_activation_checkpoints
        else:
            checkpoint_activations_microbatch = None

        output_tensor, num_tokens = forward_step(
            forward_step_func,
            data_iterator,
            model,
            num_microbatches,
            input_tensor,
            forward_data_store,
            config,
            collect_non_loss_data,
            checkpoint_activations_microbatch,
            check_first_val_step(
                first_val_step, forward_only, (i == 0) and (num_warmup_microbatches == 0)
            ),
            current_microbatch=i + num_warmup_microbatches,
        )
        total_num_tokens += num_tokens.item()

        if forward_only:
            tp_hetero_send_forward(output_tensor, send_tensor_shapes, send_tp_scale, config)

            if not last_iteration:
                input_tensor = tp_hetero_recv_forward(recv_tensor_shapes, recv_tp_scale, config)

        else:
            output_tensor_grad = tp_hetero_send_forward_recv_backward(output_tensor, send_tensor_shapes, send_tp_scale, config)

            # Add input_tensor and output_tensor to end of list.
            input_tensors.append(input_tensor)
            output_tensors.append(output_tensor)
            deallocate_output_tensor(output_tensor[0], config.deallocate_pipeline_outputs)

            # Pop input_tensor and output_tensor from the start of the list for
            # the backward pass.
            input_tensor = input_tensors.pop(0)
            output_tensor = output_tensors.pop(0)

            # Enable grad sync for the last microbatch in the batch if the full
            # backward pass completes in the 1F1B stage.
            if num_warmup_microbatches == 0 and last_iteration:
                if config.grad_sync_func is None or rank == 0:
                    enable_grad_sync()

            input_tensor_grad = backward_step(
                input_tensor, output_tensor, output_tensor_grad, model_type, config
            )

            if last_iteration:
                input_tensor = None
                tp_hetero_send_backward(input_tensor_grad, recv_tensor_shapes, recv_tp_scale, config)
            else:
                input_tensor = tp_hetero_send_backward_recv_forward(input_tensor_grad, recv_tensor_shapes, recv_tp_scale, config)

    # Run cooldown backward passes.
    if not forward_only:
        for i in range(num_warmup_microbatches):

            # Enable async grad reduction in the last backward pass
            # Note: If grad sync function is provided, only enable
            # async grad reduction in first pipeline stage. Other
            # pipeline stages do grad reduction during pipeline
            # bubble.
            if i == num_warmup_microbatches - 1:
                if config.grad_sync_func is None or rank == 0:
                    enable_grad_sync()

            input_tensor = input_tensors.pop(0)
            output_tensor = output_tensors.pop(0)

            output_tensor_grad = tp_hetero_recv_backward(send_tensor_shapes, send_tp_scale, config)

            input_tensor_grad = backward_step(
                input_tensor, output_tensor, output_tensor_grad, model_type, config
            )

            tp_hetero_send_backward(input_tensor_grad, recv_tensor_shapes, recv_tp_scale, config)

        # Launch any remaining grad reductions.
        if no_sync_context is not None:
            enable_grad_sync()
            if config.grad_sync_func is not None:
                config.grad_sync_func(model.parameters())

    if config.finalize_model_grads_func is not None and not forward_only:
        # Finalize model grads (perform full grad all-reduce / reduce-scatter for
        # data parallelism, layernorm all-reduce for sequence parallelism, and
        # embedding all-reduce for pipeline parallelism).
        config.finalize_model_grads_func(
            [model], total_num_tokens if config.calculate_per_token_loss else None
        )
        
    if config.timers is not None:
        config.timers('forward-backward').stop()

    return forward_data_store