# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.


import contextlib
from functools import wraps
from typing import Iterator, List, Union

import torch

from megatron.core import parallel_state
from megatron.core.enums import ModelType
from megatron.core.transformer.moe.router import MoEAuxLossAutoScaler
from megatron.core.utils import (
    get_attr_wrapped_model,
    get_model_config,
    get_model_type,
)
from megatron.core.pipeline_parallel.schedules import clear_embedding_activation_buffer, deallocate_output_tensor
from megatron.core import ModelParallelConfig
from megatron.core.pipeline_parallel.p2p_communication import _communicate
from megatron.core.pipeline_parallel.schedules import backward_step, set_current_microbatch, custom_backward, finish_embedding_wgrad_compute
from megatron.core.models.gpt import GPTModel
# from megatron.core.pipeline_parallel.weight_grad_store import WeightGradStore
# from megatron.core.pipeline_parallel.weight_grad_store import weight_grad_store
# from megatron.core.pipeline_parallel.weight_grad_store import WeightGradStore

# Types
Shape = Union[List[int], torch.Size]
# LOSS_BACKWARD_SCALE = torch.tensor(1.0)

_DUALPIPE_CHUNK = None

def set_dualpipe_chunk(chunkid):
    """set_dualpipe_chunk for fp16forward patch"""
    global _DUALPIPE_CHUNK
    _DUALPIPE_CHUNK = chunkid

def get_dualpipe_chunk():
    global _DUALPIPE_CHUNK
    if _DUALPIPE_CHUNK is not None:
        return _DUALPIPE_CHUNK
    else:
        raise AssertionError("_DUALPIPE_CHUNK is None")


def is_dualpipev_last_stgae(model_chunk_id):
    return parallel_state.is_pipeline_first_stage() and model_chunk_id == 1


def send_forward(output_tensor: torch.Tensor, tensor_shape, config: ModelParallelConfig, model_chunk_id, async_op=False) -> None:
    """Send tensor to next rank in pipeline (forward send).

    See _communicate for argument details.
    """
    tensor_send_next, tensor_send_prev = None, None
    if model_chunk_id == 0:
        if parallel_state.is_pipeline_last_stage():
            return None
        tensor_send_next = output_tensor
    else:
        if parallel_state.is_pipeline_first_stage():
            return None
        tensor_send_prev = output_tensor

    if config.timers is not None:
        config.timers('forward-send', log_level=2).start()

    _, _, fwd_wait_handles = _communicate(
        tensor_send_next=tensor_send_next,
        tensor_send_prev=tensor_send_prev,
        recv_prev=False,
        recv_next=False,
        tensor_shape=tensor_shape,
        config=config,
        wait_on_reqs=(not async_op)
    )
    if config.timers is not None:
        config.timers('forward-send').stop()

    return fwd_wait_handles


def send_backward(input_tensor_grad: torch.Tensor, tensor_shape, config: ModelParallelConfig, model_chunk_id, async_op=False) -> None:
    """Send tensor to next rank in pipeline (forward send).

    See _communicate for argument details.
    """

    tensor_send_next, tensor_send_prev = None, None
    if model_chunk_id == 0:
        if parallel_state.is_pipeline_first_stage():
            return None
        tensor_send_prev = input_tensor_grad
    else:
        if parallel_state.is_pipeline_last_stage():
            return None
        tensor_send_next = input_tensor_grad

    if config.timers is not None:
        config.timers('backward-send', log_level=2).start()
    _, _, reqs = _communicate(
        tensor_send_next=tensor_send_next,
        tensor_send_prev=tensor_send_prev,
        recv_prev=False,
        recv_next=False,
        tensor_shape=tensor_shape,
        config=config,
        wait_on_reqs=(not async_op)
    )
    if config.timers is not None:
        config.timers('backward-send').stop()
    return reqs


def recv_forward(tensor_shape: Shape, config: ModelParallelConfig, model_chunk_id, async_op=False) -> torch.Tensor:
    """ Receive tensor from previous rank in pipeline (forward receive).

    See _communicate for argument details.
    """
    print(f"[DEBUG] Entering recv_forward: model_chunk_id={model_chunk_id}, async_op={async_op}, tensor_shape={tensor_shape}")
    
    recv_prev, recv_next = False, False
    if model_chunk_id == 0:
        recv_prev = True
        print(f"[DEBUG] model_chunk_id is 0, setting recv_prev=True")
    else:
        recv_next = True
        print(f"[DEBUG] model_chunk_id is not 0, setting recv_next=True")

    if (parallel_state.is_pipeline_first_stage() and recv_prev) or (parallel_state.is_pipeline_last_stage() and recv_next):
        print('if (parallel_state.is_pipeline_first_stage() and recv_prev) or (parallel_state.is_pipeline_last_stage() and recv_next)')
        fwd_wait_handles = None
        return None, fwd_wait_handles
    else:
        if config.timers is not None:
            config.timers('forward-recv', log_level=2).start()
            print(f"[DEBUG] Started forward-recv timer")
        
        print(f"[DEBUG] Calling _communicate for reception: recv_prev={recv_prev}, recv_next={recv_next}, async_op={not async_op}")
        tensor_recv_prev, tensor_recv_next, fwd_wait_handles = _communicate(
            tensor_send_next=None,
            tensor_send_prev=None,
            recv_prev=recv_prev,
            recv_next=recv_next,
            tensor_shape=tensor_shape,
            config=config,
            wait_on_reqs=(not async_op),
        )
        if config.timers is not None:
            config.timers('forward-recv').stop()
            print(f"[DEBUG] Stopped forward-recv timer")

    if recv_prev:
        print(f"[DEBUG] Returning tensor_recv_prev: {tensor_recv_prev.shape if tensor_recv_prev is not None and hasattr(tensor_recv_prev, 'shape') else 'None'}")
        return tensor_recv_prev, fwd_wait_handles
    else:
        print(f"[DEBUG] Returning tensor_recv_next: {tensor_recv_next.shape if tensor_recv_next is not None and hasattr(tensor_recv_next, 'shape') else 'None'}")
        return tensor_recv_next, fwd_wait_handles


def recv_backward(tensor_shape: Shape, config: ModelParallelConfig, model_chunk_id, async_op=False) -> torch.Tensor:
    """Receive tensor from next rank in pipeline (backward receive).

    See _communicate for argument details.
    """
    recv_prev, recv_next = False, False
    if model_chunk_id == 0:
        recv_next = True
    else:
        recv_prev = True

    if (parallel_state.is_pipeline_first_stage() and recv_prev) or (parallel_state.is_pipeline_last_stage() and recv_next):
        output_tensor_grad = None
        bwd_wait_handles = None
        return output_tensor_grad, bwd_wait_handles
    else:

        if config.timers is not None:
            config.timers('backward-recv', log_level=2).start()
        tensor_recv_prev, tensor_recv_next, bwd_wait_handles = _communicate(
            tensor_send_next=None,
            tensor_send_prev=None,
            recv_prev=recv_prev,
            recv_next=recv_next,
            tensor_shape=tensor_shape,
            config=config,
            wait_on_reqs=(not async_op)
        )
        if config.timers is not None:
            config.timers('backward-recv').stop()

    if recv_prev:
        return tensor_recv_prev, bwd_wait_handles
    else:
        return tensor_recv_next, bwd_wait_handles


def send_forward_recv_forward(
    output_tensor: torch.Tensor,
    tensor_shape: Shape,
    config: ModelParallelConfig,
    model_chunk_id,
    async_op=False
) -> torch.Tensor:
    """Batched recv from previous rank and send to next rank in pipeline.

    See _communicate for argument details.
    """
    print(f"[DEBUG] Entering send_forward_recv_forward with model_chunk_id={model_chunk_id}")
    recv_prev, recv_next = False, False
    tensor_send_next, tensor_send_prev = None, None
    if model_chunk_id == 0:
        if not parallel_state.is_pipeline_last_stage():
            tensor_send_next = output_tensor
            print(f"[DEBUG] model_chunk_id=0: Setting tensor_send_next, pipeline_rank={parallel_state.get_pipeline_model_parallel_rank()}")
        if not parallel_state.is_pipeline_first_stage():
            recv_prev = True
            print(f"[DEBUG] model_chunk_id=0: Setting recv_prev=True, pipeline_rank={parallel_state.get_pipeline_model_parallel_rank()}")
    if model_chunk_id == 1:
        if not parallel_state.is_pipeline_first_stage():
            tensor_send_prev = output_tensor
            print(f"[DEBUG] model_chunk_id=1: Setting tensor_send_prev, pipeline_rank={parallel_state.get_pipeline_model_parallel_rank()}")
        if not parallel_state.is_pipeline_last_stage():
            recv_next = True
            print(f"[DEBUG] model_chunk_id=1: Setting recv_next=True, pipeline_rank={parallel_state.get_pipeline_model_parallel_rank()}")

    print(f"[DEBUG] Before communication: recv_prev={recv_prev}, recv_next={recv_next}")
    if config.timers is not None:
        config.timers('forward-send-forward-recv', log_level=2).start()
    tensor_recv_prev, tensor_recv_next, fwd_wait_handles = _communicate(
        tensor_send_next=tensor_send_next,
        tensor_send_prev=tensor_send_prev,
        recv_prev=recv_prev,
        recv_next=recv_next,
        tensor_shape=tensor_shape,
        wait_on_reqs=(not async_op),
        config=config
    )
    if config.timers is not None:
        config.timers('forward-send-forward-recv').stop()
    print(f"[DEBUG] After communication: tensor_recv_prev={tensor_recv_prev is not None}, tensor_recv_next={tensor_recv_next is not None}")

    if model_chunk_id == 0:
        if not parallel_state.is_pipeline_first_stage():
            print(f"[DEBUG] Returning tensor_recv_prev for model_chunk_id=0, not first stage")
            return tensor_recv_prev, fwd_wait_handles
        else:
            print(f"[DEBUG] Returning None for model_chunk_id=0, first stage")
            return None, fwd_wait_handles
    else:
        if not parallel_state.is_pipeline_last_stage():
            print(f"[DEBUG] Returning tensor_recv_next for model_chunk_id=1, not last stage")
            return tensor_recv_next, fwd_wait_handles
        else:
            print(f"[DEBUG] Returning None for model_chunk_id=1, last stage")
            return None, fwd_wait_handles


def send_forward_recv_slave_forward(
    output_tensor: torch.Tensor,
    tensor_shape: Shape,
    config: ModelParallelConfig,
    model_chunk_id,
    async_op=False,
) -> torch.Tensor:
    """Batched recv from previous rank and send to next rank in pipeline.
    See _communicate for argument details.
    """
    recv_prev, recv_next = False, False
    tensor_send_next, tensor_send_prev = None, None
    if model_chunk_id == 0:
        if parallel_state.is_pipeline_last_stage():
            return None, None
        tensor_send_next = output_tensor
        recv_next = True
    if model_chunk_id == 1:
        if parallel_state.is_pipeline_first_stage():
            return None, None
        tensor_send_prev = output_tensor
        recv_prev = True
    if config.timers is not None:
        config.timers('forward-send-slave-forward-recv', log_level=2).start()
    tensor_recv_prev, tensor_recv_next, fwd_wait_handles = _communicate(
        tensor_send_next=tensor_send_next,
        tensor_send_prev=tensor_send_prev,
        recv_prev=recv_prev,
        recv_next=recv_next,
        tensor_shape=tensor_shape,
        wait_on_reqs=(not async_op),
        config=config,
    )
    if config.timers is not None:
        config.timers('forward-send-slave-forward-recv').stop()

    if model_chunk_id == 0:
        return tensor_recv_next, fwd_wait_handles
    else:
        return tensor_recv_prev, fwd_wait_handles


def generate_dualpipev_schedule(pp_size, num_microbatches):
    from megatron.training import get_args
    
    print("pp_size: ",pp_size)  
    print("num_microbatches: ",num_microbatches)

    num_microbatches = num_microbatches * 2
    num_warmup_stages = [0] * pp_size
    num_interleaved_forward_stages = [0] * pp_size
    num_1b1w1f_stages = [0] * pp_size
    num_overlap_stages = [0] * pp_size
    num_1b1overlap_stages = [0] * pp_size
    num_interleaved_backward_stages = [0] * pp_size
    num_cooldown_stages = [0] * pp_size

    pp_size *= 2
    for i in range(pp_size // 2):
        num_warmup_stages[i] = pp_size - 2 - i * 2

        num_interleaved_forward_stages[i] = i + 1  # 每个单位是一组1f1f

        num_1b1w1f_stages[i] = pp_size // 2 - i - 1

        num_overlap_stages[i] = num_microbatches - pp_size * 2 + i * 2 + 2

        num_1b1overlap_stages[i] = (pp_size // 2 - i - 1) * 2

        num_interleaved_backward_stages[i] = i + 1

        num_cooldown_stages[i] = [i + 1, pp_size - 2 * i - 2, i + 1]

    schedule_all_stages = {
        'warmup': num_warmup_stages,
        'interleaved_forward': num_interleaved_forward_stages,
        '1b1w1f': num_1b1w1f_stages,
        'overlap': num_overlap_stages,
        '1b1overlap': num_1b1overlap_stages,
        'interleaved_backward': num_interleaved_backward_stages,
        'cooldown': num_cooldown_stages
    }

    print("schedule_all_stages: ",schedule_all_stages)

    return schedule_all_stages


def forward_step_no_model_graph(
    forward_step_func,
    model_chunk_id,
    data_iterator,
    model,
    num_microbatches,
    input_tensor,
    forward_data_store,
    config,
    collect_non_loss_data=False,
    checkpoint_activations_microbatch=None,
    is_first_microbatch=False,
    current_microbatch=None,
):
    print(f"[DEBUG] Entering forward_step_no_model_graph: model_chunk_id={model_chunk_id}, is_first_microbatch={is_first_microbatch}, current_microbatch={current_microbatch}")
    
    if config.timers is not None:
        config.timers('forward-compute', log_level=2).start()

    if is_first_microbatch and hasattr(model, 'set_is_first_microbatch'):
        model.set_is_first_microbatch()
        print(f"[DEBUG] Setting is_first_microbatch for model")
    if current_microbatch is not None:
        set_current_microbatch(model, current_microbatch)
        print(f"[DEBUG] Set current_microbatch to {current_microbatch}")

    unwrap_output_tensor = False
    if not isinstance(input_tensor, list):
        input_tensor = [input_tensor]
        unwrap_output_tensor = True
        print(f"[DEBUG] Input tensor wrapped as list, unwrap_output_tensor={unwrap_output_tensor}")

    set_input_tensor = get_attr_wrapped_model(model, "set_input_tensor")
    set_input_tensor(input_tensor)
    print(f"[DEBUG] Input tensor set for model")

    if config.enable_autocast:
        context_manager = torch.autocast("cuda", dtype=config.autocast_dtype)
        print(f"[DEBUG] Using autocast with dtype={config.autocast_dtype}")
    else:
        context_manager = contextlib.nullcontext()
        print(f"[DEBUG] Using nullcontext (autocast disabled)")
    
    with context_manager:
        print(f"[DEBUG] Calling forward_step_func with {'checkpoint_activations_microbatch' if checkpoint_activations_microbatch is not None else 'standard'} mode")
        if checkpoint_activations_microbatch is None:
            output_tensor, loss_func = forward_step_func(data_iterator, model)
        else:
            output_tensor, loss_func = forward_step_func(
                data_iterator, model, checkpoint_activations_microbatch
            )
        print(f"[DEBUG] Forward step completed, output_tensor shape: {output_tensor.shape if hasattr(output_tensor, 'shape') else 'N/A'}")

    num_tokens = torch.tensor(0, dtype=torch.int)
    print(f"[DEBUG] is_dualpipev_last_stgae check: {is_dualpipev_last_stgae(model_chunk_id)}")
    
    if is_dualpipev_last_stgae(model_chunk_id):
        if not collect_non_loss_data:
            print(f"[DEBUG] Processing loss data")
            outputs = loss_func(output_tensor)
            print(f"[DEBUG] Loss function outputs length: {len(outputs)}")
            
            if len(outputs) == 3:
                output_tensor, num_tokens, loss_reduced = outputs
                print(f"[DEBUG] 3-tuple output: num_tokens={num_tokens}, loss_reduced={loss_reduced}")
                if not config.calculate_per_token_loss:
                    output_tensor /= num_tokens
                    output_tensor /= num_microbatches
                    print(f"[DEBUG] Averaged output_tensor by tokens and microbatches")
            else:
                # preserve legacy loss averaging behavior (ie, over the number of microbatches)
                assert len(outputs) == 2
                output_tensor, loss_reduced = outputs
                output_tensor /= num_microbatches
                print(f"[DEBUG] 2-tuple output: loss_reduced={loss_reduced}, divided by num_microbatches={num_microbatches}")
            
            forward_data_store.append(loss_reduced)
            print(f"[DEBUG] Appended loss_reduced to forward_data_store")
        else:
            print(f"[DEBUG] Collecting non-loss data")
            data = loss_func(output_tensor, non_loss_data=True)
            forward_data_store.append(data)
            print(f"[DEBUG] Appended non-loss data to forward_data_store")

    if config.timers is not None:
        config.timers('forward-compute').stop()
        print(f"[DEBUG] Forward-compute timer stopped")

    # Set the loss scale for the auxiliary loss of the MoE layer.
    # Since we use a trick to do backward on the auxiliary loss, we need to set the scale explicitly.
    if hasattr(config, 'num_moe_experts') and config.num_moe_experts is not None:
        # Calculate the loss scale based on the grad_scale_func if available, else default to 1.
        loss_scale = (
            config.grad_scale_func(torch.ones(1, device=output_tensor.device))
            if config.grad_scale_func is not None
            else torch.tensor(1.0)
        )
        # Set the loss scale
        MoEAuxLossAutoScaler.set_loss_scale(loss_scale / num_microbatches)
        print(f"[DEBUG] Set MoE auxiliary loss scale: {loss_scale / num_microbatches}")

    # If T5 model (or other model with encoder and decoder)
    # and in decoder stack, then send encoder_hidden_state
    # downstream as well.
    model_type = get_model_type(model)
    print(f"[DEBUG] Model type: {model_type}")
    
    if (
        parallel_state.is_pipeline_stage_after_split()
        and model_type == ModelType.encoder_and_decoder
    ):
        print(f"[DEBUG] Returning encoder-decoder output: [output_tensor, input_tensor[-1]], num_tokens={num_tokens}")
        return [output_tensor, input_tensor[-1]], num_tokens

    if unwrap_output_tensor:
        print(f"[DEBUG] Returning unwrapped output_tensor, num_tokens={num_tokens}")
        return output_tensor, num_tokens
    
    print(f"[DEBUG] Returning wrapped [output_tensor], num_tokens={num_tokens}")
    return [output_tensor], num_tokens


shared_embedding = None

def set_shared_embedding_from_dual_chunk(model1, model2):
    global shared_embedding
    if shared_embedding is not None:
        return
    if model1.module.module.pre_process:
        shared_embedding = model1.module.module.embedding.word_embeddings.weight
    elif model2.module.module.pre_process:
        shared_embedding = model2.module.module.embedding.word_embeddings.weight


def forward_backward_pipelining_with_cutinhalf(
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
    from megatron.training import get_args
    
    print(f"Starting forward_backward_pipelining_with_cutinhalf function")
    
    args = get_args()
    # args.moe_fb_overlap = True
    args.moe_fb_overlap = False
    args.dualpipe_no_dw_detach = True
    
    set_shared_embedding_from_dual_chunk(model[0], model[1])
    assert (
        isinstance(model, list) and len(model) == 2
    ), 'Dualpipe Schedule only support chunk model for two consecutive chunks'

    assert (
        isinstance(data_iterator, list) and len(data_iterator) == 2
    ), 'Dualpipe Schedule only support two data_iterators'

    config = get_model_config(model[0])
    config.batch_p2p_comm = False

    print(f"Configuration complete, checking embedding module")
    
    # Needed only when gradients are finalized in M-Core
    if config.finalize_model_grads_func is not None and not forward_only:
        embedding_module = clear_embedding_activation_buffer(config, model)

    if config.timers is not None:
        config.timers('forward-backward',
                      log_level=1).start(barrier=config.barrier_with_L1_time)

    # Disable async grad reductions
    no_sync_func = config.no_sync_func
    print(f"in dualpipev, no_sync_func is {no_sync_func}")
    if no_sync_func is None:
        no_sync_func = contextlib.nullcontext
    no_sync_context = None

    if not isinstance(no_sync_func, list):
        no_sync_func = [no_sync_func]

    def disable_grad_sync():
        """Disable asynchronous grad reductions"""
        nonlocal no_sync_context
        if no_sync_context is None:
            for no_sync_f in no_sync_func:
                no_sync_context = no_sync_f()
                no_sync_context.__enter__()

    def enable_grad_sync():
        """Enable asynchronous grad reductions"""
        nonlocal no_sync_context
        if no_sync_context is not None:
            no_sync_context.__exit__(None, None, None)
            no_sync_context = None

    print(f"Disabling gradient synchronization")
    disable_grad_sync()

    # Compute number of steps for each stage
    pp_size = parallel_state.get_pipeline_model_parallel_world_size()
    rank = parallel_state.get_pipeline_model_parallel_rank()
    print(f"Pipeline parallel size: pp_size={pp_size}, rank={rank}")
    
    schedule = generate_dualpipev_schedule(pp_size, num_microbatches)

    model_type = get_model_type(model[0])

    tensor_shape = [seq_length, micro_batch_size, config.hidden_size]
    tensor_shape[0] = tensor_shape[0] // parallel_state.get_context_parallel_world_size()
    if config.sequence_parallel:
        tensor_shape[0] = tensor_shape[0] // parallel_state.get_tensor_model_parallel_world_size()
    
    print(f"Tensor shape: {tensor_shape}")

    total_num_tokens = torch.tensor(0, dtype=torch.int).cuda()
    input_tensors = [[], []]
    output_tensors = [[], []]
    model_graphs = [[], []]
    logits_inputs = []
    forward_data_store = []

    master_chunk_id = 0
    slave_chunk_id = 1

    master_cur_microbatch = 0
    slave_cur_microbatch = num_microbatches
    master_microbatch_max = num_microbatches
    slave_microbatch_max = num_microbatches * 2

    print(f"Setting dualpipe_chunk: {master_chunk_id}")
    set_dualpipe_chunk(master_chunk_id)


    checkpoint_activations_microbatch = None



    print(f"Starting to receive forward data")
    input_tensor = recv_forward(tensor_shape, config, master_chunk_id)[0]
    print(f"Received forward data")

    fwd_wait_handles_warmup = None
    # Run warmup forward passes
    print(f"Starting warmup phase: stage {rank} needs {schedule['warmup'][rank]} warmup iterations")
    for i in range(schedule['warmup'][rank]):
        print(f"Warmup iteration {i+1}/{schedule['warmup'][rank]}")

        print(f"Using normal mode for warmup, starting forward computation")
        output_tensor_warmup, num_tokens = forward_step_no_model_graph(
            forward_step_func,
            master_chunk_id,
            data_iterator[master_chunk_id],
            model[master_chunk_id],
            num_microbatches,
            input_tensor,
            forward_data_store,
            config,
            collect_non_loss_data,
            checkpoint_activations_microbatch,
            is_first_microbatch=(i == 0),
            current_microbatch=master_cur_microbatch
        )
        print(f"Completed warmup forward computation")

        total_num_tokens += num_tokens.item()
        input_tensors[master_chunk_id].append(
            (master_cur_microbatch, input_tensor))
        output_tensors[master_chunk_id].append(output_tensor_warmup)

        master_cur_microbatch += 1

        if i != schedule['warmup'][rank] - 1:
            print(f"Warmup iteration {i+1}: Sending forward data and receiving new data")
            input_tensor, _ = send_forward_recv_forward(
                output_tensor_warmup, tensor_shape, config, master_chunk_id)
            print(f"Warmup iteration {i+1}: Data send and receive completed")

            deallocate_output_tensor(
                output_tensor_warmup, config.deallocate_pipeline_outputs)
            print(f"Warmup iteration {i+1}: Tensor deallocation completed")
        else:
            print(f"Last warmup iteration: Receiving forward data only")
            input_tensor, _ = recv_forward(
                tensor_shape, config, master_chunk_id)
            print(f"Last warmup iteration: Forward data reception completed")
            
            print(f"Last warmup iteration: Sending forward data")
            fwd_wait_handles_warmup = send_forward(
                output_tensor_warmup, tensor_shape, config, master_chunk_id, async_op=True)
            print(f"Last warmup iteration: Forward data send completed")
    
    print(f"Warmup phase completed")
    
    # Run interleaved forward passes for two model chunk
    fwd_wait_handles = None
    fwd_wait_handles_slave_chunk = None
    fwd_wait_handles_send = None
    print(f"Starting interleaved forward phase: stage {rank} needs {schedule['interleaved_forward'][rank]} interleaved forwards")
    for i in range(schedule['interleaved_forward'][rank]):
        print(f"Interleaved forward iteration {i+1}/{schedule['interleaved_forward'][rank]}")

        if fwd_wait_handles is not None:
            print(f"Waiting for forward handles")
            print(f"fwd_wait_handles: {fwd_wait_handles}")
            for req in fwd_wait_handles:
                if type(req) is str:
                    fwd_wait_handles[req].wait()
                else:
                    req.wait()
            fwd_wait_handles = None
            print(f"Forward handles wait completed")

        is_first_microbatch = parallel_state.is_pipeline_last_stage() and (i == 0)
        print(f"Setting master chunk_id: {master_chunk_id}")
        set_dualpipe_chunk(master_chunk_id)


        print(f"Using normal mode for interleaved forward, starting forward computation")
        output_tensor, num_tokens = forward_step_no_model_graph(
            forward_step_func,
            master_chunk_id,
            data_iterator[master_chunk_id],
            model[master_chunk_id],
            num_microbatches,
            input_tensor,
            forward_data_store,
            config,
            collect_non_loss_data,
            checkpoint_activations_microbatch,
            is_first_microbatch=is_first_microbatch,
            current_microbatch=master_cur_microbatch
        )
        print(f"Interleaved forward computation completed")

        total_num_tokens += num_tokens.item()
        input_tensors[master_chunk_id].append(
            (master_cur_microbatch, input_tensor))
        output_tensors[master_chunk_id].append(output_tensor)

        master_cur_microbatch += 1
        print(f"Master microbatch increased to {master_cur_microbatch}")

        if not parallel_state.is_pipeline_last_stage() and fwd_wait_handles_send is not None:
            print(f"Waiting for forward send handles")
            for req in fwd_wait_handles_send:
                if type(req) is str:
                    fwd_wait_handles_send[req].wait()
                else:
                    req.wait()
            deallocate_output_tensor(
                output_tensor_send, config.deallocate_pipeline_outputs)
            fwd_wait_handles_send = None
            print(f"Forward send handles wait completed")

        if parallel_state.is_pipeline_last_stage():
            print(f"Pipeline last stage: Setting slave chunk input tensor")
            # input_tensor_slave_chunk = output_tensor
            # input_tensor_slave_chunk = output_tensor.clone()
            input_tensor_slave_chunk = output_tensor.detach()

            print(f"Pipeline last stage: Receiving forward data")
            input_tensor, fwd_wait_handles = recv_forward(
                tensor_shape, config, master_chunk_id, async_op=True)
            print(f"Pipeline last stage: Forward data reception completed")
        else:
            print(f"Non-pipeline last stage: Receiving slave chunk forward data")
            input_tensor_slave_chunk, _ = recv_forward(
                tensor_shape, config, slave_chunk_id)
            print(f"Non-pipeline last stage: Slave chunk forward data reception completed")

            print(f"Non-pipeline last stage: Receiving master chunk forward data")
            input_tensor, fwd_wait_handles = recv_forward(
                tensor_shape, config, master_chunk_id, async_op=True)
            print(f"Non-pipeline last stage: Master chunk forward data reception completed")

        if fwd_wait_handles_warmup is not None:
            print(f"Waiting for warmup forward handles")
            for req in fwd_wait_handles_warmup:
                if type(req) is str:
                    fwd_wait_handles_warmup[req].wait()
                else:
                    req.wait()
            deallocate_output_tensor(
                output_tensor_warmup, config.deallocate_pipeline_outputs)
            fwd_wait_handles_warmup = None
            print(f"Warmup forward handles wait completed")

        if fwd_wait_handles_slave_chunk is not None:
            print(f"Waiting for slave chunk forward handles")
            for req in fwd_wait_handles_slave_chunk:
                if type(req) is str:
                    fwd_wait_handles_slave_chunk[req].wait()
                else:
                    req.wait()
            deallocate_output_tensor(
                output_tensor_slave_chunk, config.deallocate_pipeline_outputs)
            fwd_wait_handles_slave_chunk = None
            print(f"Slave chunk forward handles wait completed")

        print(f"Setting slave chunk_id: {slave_chunk_id}")
        set_dualpipe_chunk(slave_chunk_id)

        print(f"Using normal mode for slave chunk forward, starting forward computation")
        output_tensor_slave_chunk, num_tokens = forward_step_no_model_graph(
            forward_step_func,
            slave_chunk_id,
            data_iterator[slave_chunk_id],
            model[slave_chunk_id],
            num_microbatches,
            input_tensor_slave_chunk,
            forward_data_store,
            config,
            collect_non_loss_data,
            checkpoint_activations_microbatch,
            current_microbatch=slave_cur_microbatch,
        )
        print(f"Slave chunk forward computation completed")

        input_tensors[slave_chunk_id].append(
            (slave_cur_microbatch, input_tensor_slave_chunk))
        total_num_tokens += num_tokens.item()
        output_tensors[slave_chunk_id].append(output_tensor_slave_chunk)

        slave_cur_microbatch += 1
        print(f"Slave microbatch increased to {slave_cur_microbatch}")

        if i == schedule['interleaved_forward'][rank] - 1:
            print(f"Last interleaved forward iteration, setting firstFB_no_overlp")
            firstFB_no_overlp = False
            firstFB_no_overlp_handle = None
            # last rank not overlap first F&B
            if parallel_state.is_pipeline_last_stage():
                print(f"Pipeline last stage: Setting firstFB_no_overlp=True")
                firstFB_no_overlp = True
                print(f"Pipeline last stage: Receiving backward data")
                output_tensor_grad_bwd, firstFB_no_overlp_handle = recv_backward(
                    tensor_shape, config, slave_chunk_id, async_op=True)
                print(f"Pipeline last stage: Backward data reception completed")
            else:
                print(f"Non-pipeline last stage: Receiving backward data")
                output_tensor_grad_bwd, _ = recv_backward(
                    tensor_shape, config, slave_chunk_id)
                print(f"Non-pipeline last stage: Backward data reception completed")

        print(f"Sending slave chunk forward data")
        fwd_wait_handles_slave_chunk = send_forward(output_tensor_slave_chunk,
                                                    tensor_shape, config, slave_chunk_id, async_op=True)
        print(f"Slave chunk forward data send completed")

        if not parallel_state.is_pipeline_last_stage():
            print(f"Non-pipeline last stage: Sending master chunk forward data")
            output_tensor_send = output_tensor
            fwd_wait_handles_send = send_forward(
                output_tensor_send, tensor_shape, config, master_chunk_id, async_op=True)
            print(f"Non-pipeline last stage: Master chunk forward data send completed")  
        else:
            print(f"pipeline last stage: deallocate_pipeline_outputs")
            print(f"before deallocate, output_tensor is {output_tensor}")
            deallocate_output_tensor(
                output_tensor, config.deallocate_pipeline_outputs)      
            print(f"after deallocate, output_tensor is {output_tensor}") 

    print(f"Interleaved forward phase completed")
    
    if fwd_wait_handles is not None:
        print(f"Waiting for remaining forward handles")
        for req in fwd_wait_handles:
            if type(req) is str:
                fwd_wait_handles[req].wait()
            else:
                req.wait()
        fwd_wait_handles = None
        print(f"Remaining forward handles wait completed")

    # Run 1b1w1f stages for slave chunk
    bwd_wait_handles = None
    print(f"Starting 1b1w1f phase: stage {rank} needs {schedule['1b1w1f'][rank]} 1b1w1f iterations")
    for i in range(schedule['1b1w1f'][rank]):
        print(f"1b1w1f iteration {i+1}/{schedule['1b1w1f'][rank]}")

        # print(f"Starting weight gradient decoupling")
        # WeightGradStore.start_decouple()

        print(f"Using normal mode for 1b1w1f")
        print(f"Getting input and output tensors")
        input_tensor_bwd = input_tensors[slave_chunk_id].pop(0)[1]
        output_tensor_bwd = output_tensors[slave_chunk_id].pop(0)

        print(f"Executing backward step")
        input_tensor_grad = backward_step(
            input_tensor_bwd, output_tensor_bwd, output_tensor_grad_bwd, model_type, config
        )
        print(f"Backward step completed")

        # print(f"Ending weight gradient decoupling")
        # WeightGradStore.end_decouple()

        print(f"Sending backward gradient")
        bwd_wait_handles = send_backward(input_tensor_grad,
                                         tensor_shape, config, slave_chunk_id)
        print(f"Backward gradient send completed")

        if fwd_wait_handles_slave_chunk is not None:
            print(f"Waiting for slave chunk forward handles")
            for req in fwd_wait_handles_slave_chunk:
                if type(req) is str:
                    fwd_wait_handles_slave_chunk[req].wait()
                else:
                    req.wait()
            deallocate_output_tensor(
                output_tensor_slave_chunk, config.deallocate_pipeline_outputs)
            fwd_wait_handles_slave_chunk = None
            print(f"Slave chunk forward handles wait completed")
        if fwd_wait_handles_send is not None:
            print(f"Waiting for send forward handles")
            for req in fwd_wait_handles_send:
                if type(req) is str:
                    fwd_wait_handles_send[req].wait()
                else:
                    req.wait()
            deallocate_output_tensor(
                output_tensor, config.deallocate_pipeline_outputs)
            fwd_wait_handles_send = None
            print(f"Send forward handles wait completed")

        print(f"Receiving forward data")
        input_tensor_slave_chunk, recv_forward_handle = recv_forward(
            tensor_shape, config, slave_chunk_id)
        print(f"Forward data reception completed")

        # print(f"Popping weight gradient")
        # WeightGradStore.pop()
        # print(f"Weight gradient pop completed")

        if recv_forward_handle is not None:
            print(f"Waiting for receive forward handles")
            for req in recv_forward_handle:
                if type(req) is str:
                    recv_forward_handle[req].wait()
                else:
                    req.wait()
            recv_forward_handle = None
            print(f"Receive forward handles wait completed")

        # 1F: Forward pass
        print(f"Setting slave chunk_id: {slave_chunk_id}")
        set_dualpipe_chunk(slave_chunk_id)


        print(f"Using normal mode for slave chunk forward, starting forward computation")
        output_tensor_slave_chunk, num_tokens = forward_step_no_model_graph(
            forward_step_func,
            slave_chunk_id,
            data_iterator[slave_chunk_id],
            model[slave_chunk_id],
            num_microbatches,
            input_tensor_slave_chunk,
            forward_data_store,
            config,
            collect_non_loss_data,
            checkpoint_activations_microbatch,
            current_microbatch=slave_cur_microbatch
        )
        print(f"Slave chunk forward computation completed")

        input_tensors[slave_chunk_id].append(
            (slave_cur_microbatch, input_tensor_slave_chunk))
        total_num_tokens += num_tokens.item()
        output_tensors[slave_chunk_id].append(output_tensor_slave_chunk)

        slave_cur_microbatch += 1
        print(f"Slave microbatch increased to {slave_cur_microbatch}")

        print(f"Receiving backward gradient")
        output_tensor_grad_bwd, _ = recv_backward(
            tensor_shape, config, slave_chunk_id)
        print(f"Backward gradient reception completed")

        print(f"Sending slave chunk forward data")
        fwd_wait_handles_slave_chunk = send_forward(output_tensor_slave_chunk,
                                                    tensor_shape, config, slave_chunk_id, async_op=True)
        print(f"Slave chunk forward data send completed")

    fwd_wait_handles_recv = None
    # Run overlaping f&bw stages
    fwd_model_chunk_id = master_chunk_id
    bwd_model_chunk_id = slave_chunk_id
    print(f"[DEBUG][Overlap] Starting overlapping f&bw stages. Initial fwd_chunk={fwd_model_chunk_id}, bwd_chunk={bwd_model_chunk_id}")
    overlap_loop_range = schedule['overlap'][rank] + schedule['1b1overlap'][rank] + schedule['interleaved_backward'][rank]
    print(f"[DEBUG][Overlap] Loop range: {overlap_loop_range}")
    for iter_num in range(overlap_loop_range):
        print(f"[DEBUG][Overlap][Iter {iter_num+1}/{overlap_loop_range}] Starting iteration. fwd_chunk={fwd_model_chunk_id}, bwd_chunk={bwd_model_chunk_id}")
        only_bwd = False
        if fwd_model_chunk_id == master_chunk_id and master_cur_microbatch == master_microbatch_max:
            only_bwd = True
            print(f"[DEBUG][Overlap][Iter {iter_num+1}] Master forward finished (mb {master_cur_microbatch}/{master_microbatch_max}), switching to only_bwd=True")
        if fwd_model_chunk_id == slave_chunk_id and slave_cur_microbatch == slave_microbatch_max:
            only_bwd = True
            print(f"[DEBUG][Overlap][Iter {iter_num+1}] Slave forward finished (mb {slave_cur_microbatch}/{slave_microbatch_max}), switching to only_bwd=True")


        firstFB_no_overlp = False
        if not only_bwd:
            print(f"[DEBUG][Overlap][Iter {iter_num+1}] Running Forward step for chunk {fwd_model_chunk_id}")
            fwd_microbatch = master_cur_microbatch if fwd_model_chunk_id == master_chunk_id else slave_cur_microbatch
            print(f"[DEBUG][Overlap][Iter {iter_num+1}] Forward microbatch index: {fwd_microbatch}")
            set_dualpipe_chunk(fwd_model_chunk_id)
            print(f"[DEBUG][Overlap][Iter {iter_num+1}] Set dualpipe chunk to {fwd_model_chunk_id}")

            output_tensor, num_tokens = forward_step_no_model_graph(
                forward_step_func,
                fwd_model_chunk_id,
                data_iterator[fwd_model_chunk_id],
                model[fwd_model_chunk_id],
                num_microbatches,
                input_tensor,
                forward_data_store,
                config,
                collect_non_loss_data,
                checkpoint_activations_microbatch,
                current_microbatch=fwd_microbatch
            )
            input_tensors[fwd_model_chunk_id].append(
                (fwd_microbatch, input_tensor))
            total_num_tokens += num_tokens.item()
            output_tensors[fwd_model_chunk_id].append(output_tensor)

            if fwd_model_chunk_id == master_chunk_id:
                master_cur_microbatch += 1
                fwd_send_only = False
                print(f"[DEBUG][Overlap][Iter {iter_num+1}] Increased master_cur_microbatch to {master_cur_microbatch}. fwd_send_only=False")
            else:
                slave_cur_microbatch += 1
                fwd_send_only = (master_cur_microbatch ==
                                    master_microbatch_max)
                print(f"[DEBUG][Overlap][Iter {iter_num+1}] Increased slave_cur_microbatch to {slave_cur_microbatch}. fwd_send_only={fwd_send_only}")

            if fwd_send_only:
                print(f"[DEBUG][Overlap][Iter {iter_num+1}] Forward send only for chunk {fwd_model_chunk_id}")
                fwd_wait_handles = send_forward(
                    output_tensor, tensor_shape, config, fwd_model_chunk_id, async_op=True)
                print(f"[DEBUG][Overlap][Iter {iter_num+1}] Forward send initiated (async)")
            else:
                if parallel_state.is_pipeline_last_stage() and fwd_model_chunk_id == master_chunk_id:
                    # input_tensor = output_tensor
                    input_tensor = output_tensor.detach()
                else:
                    print(f"[DEBUG][Overlap][Iter {iter_num+1}] Send forward and receive slave forward for chunk {fwd_model_chunk_id}")
                    input_tensor, fwd_wait_handles = send_forward_recv_slave_forward(
                        output_tensor, tensor_shape, config, fwd_model_chunk_id, async_op=True)

            if firstFB_no_overlp_handle is not None:
                for req in firstFB_no_overlp_handle:
                    if type(req) is str:
                        firstFB_no_overlp_handle[req].wait()
                    else:
                        req.wait()
                firstFB_no_overlp_handle = None
                print(f"[DEBUG][Overlap][Iter {iter_num+1}] Done waiting on firstFB_no_overlp_handle.")

            if bwd_wait_handles is not None:
                for req in bwd_wait_handles:
                    if type(req) is str:
                        bwd_wait_handles[req].wait()
                    else:
                        req.wait()
                bwd_wait_handles = None
                print(f"[DEBUG][Overlap][Iter {iter_num+1}] Done waiting on bwd_wait_handles.")


            input_tensor_bwd = input_tensors[bwd_model_chunk_id].pop(0)[
                1]
            output_tensor_bwd = output_tensors[bwd_model_chunk_id].pop(
                0)
            print(f"[DEBUG][Overlap][Iter {iter_num+1}] Popped output tensor for backward.")
            
            print(f"input_tensor_bwd is {input_tensor_bwd}")
            print(f"output_tensor_bwd is {output_tensor_bwd}")
            
            input_tensor_grad = backward_step(
                input_tensor_bwd, output_tensor_bwd, output_tensor_grad_bwd, model_type, config
            )
            print(f"[DEBUG][Overlap][Iter {iter_num+1}] Backward step completed for chunk {bwd_model_chunk_id}. Input grad shape: {input_tensor_grad.shape if hasattr(input_tensor_grad, 'shape') else 'N/A'}")

            if parallel_state.is_pipeline_last_stage():
                deallocate_output_tensor(
                    output_tensor, config.deallocate_pipeline_outputs)
                print(f"pipeline last stage, deallocated forward output tensor.")
            
            if fwd_wait_handles is not None:
                for req in fwd_wait_handles:
                    if type(req) is str:
                        fwd_wait_handles[req].wait()
                    else:
                        req.wait()
                fwd_wait_handles = None
                deallocate_output_tensor(
                    output_tensor, config.deallocate_pipeline_outputs)
                print(f"[DEBUG][Overlap][Iter {iter_num+1}] Deallocated forward output tensor.")

            if parallel_state.is_pipeline_last_stage() and fwd_model_chunk_id == master_chunk_id:
                output_tensor_grad_bwd = input_tensor_grad
                print(f"[DEBUG][Overlap][Iter {iter_num+1}] Last stage, master chunk fwd. Setting next output_tensor_grad_bwd directly.")
            else:
                #  send_backward_recv_slave_backward
                output_tensor_grad_bwd, bwd_wait_handles = send_forward_recv_slave_forward(input_tensor_grad,
                                                                                            tensor_shape, config, fwd_model_chunk_id, async_op=True)

            if fwd_wait_handles_slave_chunk is not None:
                for req in fwd_wait_handles_slave_chunk:  # 同步上个阶段最后一个slave前向send
                    if type(req) is str:
                        fwd_wait_handles_slave_chunk[req].wait()
                    else:
                        req.wait()
                deallocate_output_tensor(
                    output_tensor_slave_chunk, config.deallocate_pipeline_outputs)
                print(f"[DEBUG][Overlap][Iter {iter_num+1}] Deallocated slave chunk output tensor.")
                fwd_wait_handles_slave_chunk = None
                print(f"[DEBUG][Overlap][Iter {iter_num+1}] Done waiting on fwd_wait_handles_slave_chunk.")

        # only run backward
        else:
            print(f"[DEBUG][Overlap][Iter {iter_num+1}] Running Backward step ONLY for chunk {bwd_model_chunk_id}")
            if bwd_model_chunk_id == slave_chunk_id and slave_cur_microbatch < slave_microbatch_max:
                # This recv seems out of place if only backward is running.
                # Potentially needed if the *next* iteration will be forward, but logging for clarity.
                print(f"[DEBUG][Overlap][Iter {iter_num+1}] Receiving forward tensor for slave chunk {slave_chunk_id} (potentially for next iter)")
                input_tensor, _ = recv_forward(
                    tensor_shape, config, slave_chunk_id)
                print(f"[DEBUG][Overlap][Iter {iter_num+1}] Received forward tensor shape: {input_tensor.shape if hasattr(input_tensor, 'shape') else 'N/A'}")

            if bwd_wait_handles is not None:
                for req in bwd_wait_handles:
                    if type(req) is str:
                        bwd_wait_handles[req].wait()
                    else:
                        req.wait()
                bwd_wait_handles = None
                print(f"[DEBUG][Overlap][Iter {iter_num+1}] Done waiting on bwd_wait_handles (backward only).")


            input_tensor_bwd = input_tensors[bwd_model_chunk_id].pop(0)[
                1]
            output_tensor_bwd = output_tensors[bwd_model_chunk_id].pop(
                0)
            print(f"[DEBUG][Overlap][Iter {iter_num+1}] Popped output tensor for backward (backward only).")
            input_tensor_grad = backward_step(
                input_tensor_bwd, output_tensor_bwd, output_tensor_grad_bwd, model_type, config
            )
            print(f"[DEBUG][Overlap][Iter {iter_num+1}] Backward step completed for chunk {bwd_model_chunk_id} (backward only). Input grad shape: {input_tensor_grad.shape if hasattr(input_tensor_grad, 'shape') else 'N/A'}")

            if parallel_state.is_pipeline_last_stage() and fwd_model_chunk_id == master_chunk_id:
                # This condition seems related to the *forward* chunk ID, which might be stale here.
                # Logging the condition check.
                print(f"[DEBUG][Overlap][Iter {iter_num+1}] Last stage and fwd_model_chunk_id == master_chunk_id ({fwd_model_chunk_id} == {master_chunk_id}) is True. Setting output_tensor_grad_bwd directly.")
                output_tensor_grad_bwd = input_tensor_grad
            else:
                #  send_backward_recv_slave_backward
                output_tensor_grad_bwd, bwd_wait_handles = send_forward_recv_slave_forward(input_tensor_grad,
                                                                                            tensor_shape, config, fwd_model_chunk_id)

        # swap fwd & bwd chunks
        print(f"[DEBUG][Overlap][Iter {iter_num+1}] Swapping chunks. Before: fwd={fwd_model_chunk_id}, bwd={bwd_model_chunk_id}")
        fwd_model_chunk_id, bwd_model_chunk_id = bwd_model_chunk_id, fwd_model_chunk_id
        print(f"[DEBUG][Overlap][Iter {iter_num+1}] Swapped chunks. After: fwd={fwd_model_chunk_id}, bwd={bwd_model_chunk_id}")
        print(f"[DEBUG][Overlap][Iter {iter_num+1}] End of iteration.")

    print(f"[DEBUG] Overlapping f&bw stages finished.")
    # Run cooldown phases
    merged_input_tensors = []
    merged_output_tensors = []
    print(f"[DEBUG][Cooldown] Merging remaining tensors. Input queues: {[len(q) for q in input_tensors]}, Output queues: {[len(q) for q in output_tensors]}. Current bwd_chunk={bwd_model_chunk_id}")
    while len(input_tensors[0]) > 0 or len(input_tensors[1]) > 0:
        if len(input_tensors[bwd_model_chunk_id]) > 0:
            merged_input_tensors.append(
                input_tensors[bwd_model_chunk_id].pop(0))
            merged_output_tensors.append(
                (output_tensors[bwd_model_chunk_id].pop(0), bwd_model_chunk_id))

        if len(input_tensors[1 - bwd_model_chunk_id]) > 0:
            merged_input_tensors.append(
                input_tensors[1 - bwd_model_chunk_id].pop(0))
            merged_output_tensors.append(
                (output_tensors[1 - bwd_model_chunk_id].pop(0), 1 - bwd_model_chunk_id))

    bwd_wait_handles_recv = None
    print(f"[DEBUG][Cooldown] Starting cooldown loop for {pp_size} iterations.")
    for i in range(pp_size):
        print(f"[DEBUG][Cooldown][Iter {i+1}/{pp_size}] Starting iteration.")
        if bwd_wait_handles is not None:
            for req in bwd_wait_handles:
                if type(req) is str:
                    bwd_wait_handles[req].wait()
                else:
                    req.wait()
            bwd_wait_handles = None
            print(f"[DEBUG][Cooldown][Iter {i+1}] Done waiting on bwd_wait_handles.")
        if bwd_wait_handles_recv is not None:
            for req in bwd_wait_handles_recv:
                if type(req) is str:
                    bwd_wait_handles_recv[req].wait()
                else:
                    req.wait()
            bwd_wait_handles_recv = None
            print(f"[DEBUG][Cooldown][Iter {i+1}] Done waiting on bwd_wait_handles_recv.")

        input_tensor_bwd = merged_input_tensors.pop(0)[1]
        output_tensor_bwd, bwd_model_chunk_id = merged_output_tensors.pop(0)

        # if not args.dualpipe_no_dw_detach:
        #     WeightGradStore.start_decouple()

        input_tensor_grad = backward_step(
            input_tensor_bwd, output_tensor_bwd, output_tensor_grad_bwd, model_type, config
        )
        print(f"[DEBUG][Cooldown][Iter {i+1}] Backward step completed. Input grad shape: {input_tensor_grad.shape if hasattr(input_tensor_grad, 'shape') else 'N/A'}")

        # if not args.dualpipe_no_dw_detach:
        #     WeightGradStore.end_decouple()

        if i == pp_size - 1:
            bwd_wait_handles = send_backward(input_tensor_grad,
                                             tensor_shape, config, bwd_model_chunk_id, async_op=True)
        elif i >= schedule['cooldown'][rank][0] - 1:
            bwd_wait_handles = send_backward(input_tensor_grad,
                                             tensor_shape, config, bwd_model_chunk_id, async_op=True)
            output_tensor_grad_bwd, bwd_wait_handles_recv = recv_backward(
                tensor_shape, config, bwd_model_chunk_id, async_op=True)
        else:
            if parallel_state.is_pipeline_last_stage() and (1 - bwd_model_chunk_id) == master_chunk_id:
                output_tensor_grad_bwd = input_tensor_grad
            else:
                #  send_backward_recv_slave_backward
                output_tensor_grad_bwd, bwd_wait_handles = send_forward_recv_slave_forward(input_tensor_grad,
                                                                                           tensor_shape, config, 1 - bwd_model_chunk_id)

        # WeightGradStore.flush_chunk_grad()
        # if i >= schedule['cooldown'][rank][0] - 1:
        #     WeightGradStore.pop_single()
        print(f"[DEBUG][Cooldown][Iter {i+1}] End of iteration.")

    # for _ in range(schedule['cooldown'][rank][2] - 1):
    #     WeightGradStore.pop_single()

    # assert WeightGradStore.weight_grad_queue.empty()
    print(f"[DEBUG][Cooldown] Cooldown loop finished.")

    if bwd_wait_handles is not None:
        for req in bwd_wait_handles:
            if type(req) is str:
                bwd_wait_handles[req].wait()
            else:
                req.wait()
        bwd_wait_handles = None
        print(f"[DEBUG][Post-Cooldown] Done waiting on final bwd_wait_handles.")

    if config.finalize_model_grads_func is not None and not forward_only:
        print(f"[DEBUG] Finalizing gradients...")
        # If defer_embedding_wgrad_compute is enabled we need to do the
        # weight gradient GEMM's here.
        finish_embedding_wgrad_compute(config, embedding_module)
        print(f"[DEBUG] Finished embedding wgrad compute (if applicable).")

        # Finalize model grads (perform full grad all-reduce / reduce-scatter for
        # data parallelism, layernorm all-reduce for sequence parallelism, and
        # embedding all-reduce for pipeline parallelism).
        config.finalize_model_grads_func(
            model, total_num_tokens if config.calculate_per_token_loss else None
        )
        print(f"[DEBUG] Model grads finalized.")
    else:
        print(f"[DEBUG] Skipping gradient finalization (forward_only={forward_only} or finalize_func is None).")

    print(f"[DEBUG] Returning forward_data_store (length: {len(forward_data_store)})")

    if config.timers is not None:
        config.timers('forward-backward',
                      log_level=1).stop()
    return forward_data_store
