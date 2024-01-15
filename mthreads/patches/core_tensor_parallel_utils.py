import torch

import megatron
from megatron.core import parallel_state

def split_tensor_into_1d_equal_chunks(tensor, new_buffer=False):
    """ Break a tensor into equal 1D chunks across tensor parallel ranks.

        Returns a Tensor or View with this rank's portion of the data.

        Arguments:
            tensor: The tensor to split

        Keyword Arguments:
            new_buffer (bool): If True, returns a new Tensor.
                               If False, returns a view into the existing Tensor.
                               Default is False

    """
    partition_size = torch.numel(tensor) // parallel_state.get_tensor_model_parallel_world_size()
    start_index = partition_size * parallel_state.get_tensor_model_parallel_rank()
    end_index = start_index + partition_size
    if new_buffer:
        data = torch.empty(
            partition_size,
            dtype=tensor.dtype,
            device='musa:{}'.format(torch.musa.current_device()),
            requires_grad=False,
        )
        data.copy_(tensor.view(-1)[start_index:end_index])
    else:
        data = tensor.view(-1)[start_index:end_index]
    return data


def gather_split_1d_tensor(tensor):
    """ Opposite of split_tensor_into_1d_equal_chunks. Gather values from tensor
        model parallel ranks.

        Returns a new Tensor with the gathered data.

        Arguments:
            tensor: A Tensor or view of this rank's portion of the data.
    """
    numel_gathered = torch.numel(tensor) * parallel_state.get_tensor_model_parallel_world_size()
    gathered = torch.empty(
        numel_gathered, dtype=tensor.dtype, device='musa:{}'.format(torch.musa.current_device()), requires_grad=False
    )
    # TODO: This API is experimental in pytorch (as of Feb 2022) and
    # this might break in future pytorch releases. We chose this API
    # as opposed to torch.distributed.all_gather for efficiency reasons.
    # This API calls directly NCCL all-gather versus the former does
    # internal copies and can potentially cause slow down.
    torch.distributed._all_gather_base(
        gathered, tensor, group=parallel_state.get_tensor_model_parallel_group()
    )
    return gathered

# megatron.core.tensor_parallel.utils.split_tensor_into_1d_equal_chunks = split_tensor_into_1d_equal_chunks
# megatron.core.tensor_parallel.utils.gather_split_1d_tensor = gather_split_1d_tensor
