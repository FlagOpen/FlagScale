import torch
import megatron


class MemoryBuffer:

    def __init__(self, numel, numel_padded, dtype):
        self.numel = numel
        self.numel_padded = numel_padded
        self.dtype = dtype
        self.data = torch.zeros(self.numel_padded,
                                dtype=self.dtype,
                                device='musa:{}'.format(torch.musa.current_device()),
                                requires_grad=False)

    def zero(self):
        """Reset the buffer to zero."""
        self.data.zero_()


    def get(self, shape, start_index):
        """Return a tensor with the input `shape` as a view into the
        1-D data starting at `start_index`."""
        end_index = start_index + shape.numel()
        assert end_index <= self.numel, \
            'requested tensor is out of the buffer range.'
        buffer_tensor = self.data[start_index:end_index]
        buffer_tensor = buffer_tensor.view(shape)
        return buffer_tensor
    

# megatron.model.distributed.MemoryBuffer = MemoryBuffer
