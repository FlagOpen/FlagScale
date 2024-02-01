import contextlib
from typing import Iterator, List, Union

import torch
from torch.nn.parallel.distributed import DistributedDataParallel as torchDDP
import megatron
from megatron.core import parallel_state
from megatron.core.enums import ModelType
from megatron.core.pipeline_parallel.schedules import custom_backward, forward_step
from megatron.core.utils import get_model_config, get_model_type

from . import FLAG_SUPPORT_INF_NAN


def clear_npu_overflow_flag():
    float_status = torch.zeros(8).npu()
    result = torch.npu_clear_float_status(float_status)


def get_npu_overflow_flag():
    float_status = torch.zeros(8).npu()
    result = torch.npu_get_float_status(float_status)
    if float_status.cpu()[0] != 0:
        return True
    else:
        return False


def set_npu_overflow_flag():
    torch.tensor([65504]).half().npu() + 100


def backward_step(input_tensor, output_tensor, output_tensor_grad, model_type, config):
    """Backward step through passed-in output tensor.

    If last stage, output_tensor_grad is None, otherwise gradient of loss
    with respect to stage's output tensor.

    Returns gradient of loss with respect to input tensor (None if first
    stage)."""

    # NOTE: This code currently can handle at most one skip connection. It
    # needs to be modified slightly to support arbitrary numbers of skip
    # connections.

    if config.timers is not None:
        config.timers('backward-compute', log_level=2).start()

    # Retain the grad on the input_tensor.
    unwrap_input_tensor_grad = False
    if not isinstance(input_tensor, list):
        input_tensor = [input_tensor]
        unwrap_input_tensor_grad = True
    for x in input_tensor:
        if x is not None:
            x.retain_grad()

    if not isinstance(output_tensor, list):
        output_tensor = [output_tensor]
    if not isinstance(output_tensor_grad, list):
        output_tensor_grad = [output_tensor_grad]

    # Backward pass.
    if not FLAG_SUPPORT_INF_NAN:
        clear_npu_overflow_flag()
    if output_tensor_grad[0] is None and config.grad_scale_func is not None:
        output_tensor[0] = config.grad_scale_func(output_tensor[0])

    if config.deallocate_pipeline_outputs:
        custom_backward(output_tensor[0], output_tensor_grad[0])
    else:
        torch.autograd.backward(output_tensor[0], grad_tensors=output_tensor_grad[0])

    # Collect the grad of the input_tensor.
    input_tensor_grad = [None]
    if input_tensor is not None:
        input_tensor_grad = []
        for x in input_tensor:
            if x is None:
                input_tensor_grad.append(None)
            else:
                input_tensor_grad.append(x.grad)

    # Handle single skip connection if it exists (encoder_hidden_state in
    # model with encoder and decoder).
    if (
        parallel_state.get_pipeline_model_parallel_world_size() > 1
        and parallel_state.is_pipeline_stage_after_split()
        and model_type == ModelType.encoder_and_decoder
    ):
        if output_tensor_grad[1] is not None:
            input_tensor_grad[-1].add_(output_tensor_grad[1])
    if unwrap_input_tensor_grad:
        input_tensor_grad = input_tensor_grad[0]

    if config.timers is not None:
        config.timers('backward-compute').stop()

    return input_tensor_grad


def forward_backward_no_pipelining(
    *,
    forward_step_func,
    data_iterator: Union[Iterator, List[Iterator]],
    model: Union[torch.nn.Module, List[torch.nn.Module]],
    num_microbatches: int,
    seq_length: int,  # unused
    micro_batch_size: int,  # unused
    decoder_seq_length: int = None,  # unused
    forward_only: bool = False,
    collect_non_loss_data: bool = False,
):
    """Run forward and backward passes with no pipeline parallelism
    (no inter-stage communication).

    Returns dictionary with losses.


    See get_forward_backward_func() for argument details
    """

    if isinstance(model, list):
        assert len(model) == 1, "non-pipeline-parallel schedule does not support model chunking"
        model = model[0]
    if isinstance(data_iterator, list):
        assert (
            len(data_iterator) == 1
        ), "non-pipeline-parallel schedule does not support model chunking"
        data_iterator = data_iterator[0]

    config = get_model_config(model)

    no_sync_func = config.no_sync_func
    if no_sync_func is None and isinstance(model, torchDDP):
        no_sync_func = model.no_sync
    if no_sync_func is None:
        no_sync_func = contextlib.nullcontext

    model_type = get_model_type(model)

    forward_data_store = []
    input_tensor, output_tensor_grad = None, None
    overflow_flag_all = False
    with no_sync_func():
        for i in range(num_microbatches - 1):
            output_tensor = forward_step(
                forward_step_func,
                data_iterator,
                model,
                num_microbatches,
                input_tensor,
                forward_data_store,
                config,
                collect_non_loss_data,
            )
            if not forward_only:
                backward_step(input_tensor, output_tensor, output_tensor_grad, model_type, config)

            if not FLAG_SUPPORT_INF_NAN:
                overflow_flag = get_npu_overflow_flag()
                overflow_flag_all = overflow_flag or overflow_flag_all

    # Run computation for last microbatch out of context handler (want to
    # synchronize gradients).
    output_tensor = forward_step(
        forward_step_func,
        data_iterator,
        model,
        num_microbatches,
        input_tensor,
        forward_data_store,
        config,
        collect_non_loss_data,
    )

    if not forward_only:
        backward_step(input_tensor, output_tensor, output_tensor_grad, model_type, config)

    if not FLAG_SUPPORT_INF_NAN:
        overflow_flag = get_npu_overflow_flag()
        overflow_flag_all = overflow_flag or overflow_flag_all
        if overflow_flag_all:
            set_npu_overflow_flag()

    return forward_data_store


def deallocate_output_tensor(out, deallocate_pipeline_outputs=False):
    if (out is None) or (not deallocate_pipeline_outputs):
        return
    assert isinstance(out, torch.Tensor), "expected Tensor, found %s." % type(out).__name__
    assert out._base is None, "counter-productive to free a view of another tensor."
    with torch.no_grad():
        out.set_(torch.empty((1,), device=out.device, dtype=out.dtype))


megatron.core.pipeline_parallel.schedules.backward_step = backward_step
megatron.core.pipeline_parallel.schedules.forward_backward_no_pipelining = forward_backward_no_pipelining
megatron.core.pipeline_parallel.schedules.deallocate_output_tensor = deallocate_output_tensor
