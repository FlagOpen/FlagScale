import contextlib

import torch
from torch import _C
from torch.cuda import _lazy_call
from torch.utils.checkpoint import detach_variable
from torch.cuda import device as device_ctx_manager

import megatron

from megatron.core.tensor_parallel.random import (
    _MODEL_PARALLEL_RNG_TRACKER_NAME,
    get_cuda_rng_tracker
)
from megatron.core.utils import safely_set_viewless_tensor_data
from .core_tensor_parallel_utils import split_tensor_into_1d_equal_chunks, gather_split_1d_tensor

def _set_cuda_rng_state(new_state, device=-1):
    """Sets the random number generator state of the current GPU.

    Argumentss:
        new_state (torch.ByteTensor): The desired state
    This function is adapted from PyTorch repo (torch.cuda.set_rng_state)
    with a single change: the input state is not cloned. Cloning caused
    major performance issues for +4 GPU cases.
    """
    if hasattr(_C, '_cuda_setRNGState') and callable(_C._cuda_setRNGState):
        # older PyTorch
        def cb():
            with device_ctx_manager(device):
                _C._cuda_setRNGState(new_state)

    else:
        # newer PyTorch
        if device == -1:
            device = torch.device('musa')
        elif isinstance(device, str):
            device = torch.device(device)
        elif isinstance(device, int):
            device = torch.device('musa', device)

        def cb():
            idx = device.index
            if idx is None:
                idx = torch.musa.current_device()
            default_generator = torch.cuda.default_generators[idx]
            default_generator.set_state(new_state)

    _lazy_call(cb)


def CudaRNGStatesTracker_add(self, name, seed):
    """Track the rng state."""
    # Check seed is not already used.
    if seed in self.seeds_:
        raise Exception('seed {} already exists'.format(seed))
    self.seeds_.add(seed)
    # Check that state is not already defined.
    if name in self.states_:
        raise Exception('cuda rng state {} already exists'.format(name))
    # Get the current rng state.
    orig_rng_state = torch.musa.get_rng_state()
    # Set the new state and store it.
    torch.musa.manual_seed(seed)
    self.states_[name] = torch.musa.get_rng_state()
    # Reset rng state to what it was.
    _set_cuda_rng_state(orig_rng_state)


@contextlib.contextmanager
def CudaRNGStatesTracker_fork(self, name=_MODEL_PARALLEL_RNG_TRACKER_NAME):
    """Fork the cuda rng state, perform operations, and exit with
    the original state."""
    # Check if we have added the state
    if name not in self.states_:
        raise Exception('cuda rng state {} is not added'.format(name))
    # Store current rng state.
    orig_cuda_rng_state = torch.musa.get_rng_state()
    # Set rng state to the desired one
    _set_cuda_rng_state(self.states_[name])
    # Do the stuff we wanted to do.
    try:
        yield
    finally:
        # Update the current rng state for later use.
        self.states_[name] = torch.musa.get_rng_state()
        # And set the state to the original state we started with.
        _set_cuda_rng_state(orig_cuda_rng_state)


class CheckpointFunction(torch.autograd.Function):
    """This function is adapted from torch.utils.checkpoint with
       two main changes:
           1) torch.cuda.set_rng_state is replaced with `_set_cuda_rng_state`
           2) the states in the model parallel tracker are also properly
              tracked/set/reset.
    """

    @staticmethod
    def forward(ctx, run_function, distribute_saved_activations, *args):
        ctx.run_function = run_function
        ctx.distribute_saved_activations = distribute_saved_activations

        # Copy the rng states.
        ctx.fwd_cpu_rng_state = torch.get_rng_state()
        ctx.fwd_cuda_rng_state = torch.musa.get_rng_state()
        ctx.fwd_cuda_rng_state_tracker = get_cuda_rng_tracker().get_states()

        with torch.no_grad():
            outputs = run_function(*args)

        # Divide hidden states across model parallel group and only keep
        # the chunk corresponding to the current rank.
        if distribute_saved_activations:
            ctx.input_0_shape = args[0].data.shape
            safely_set_viewless_tensor_data(
                args[0], split_tensor_into_1d_equal_chunks(args[0].data, new_buffer=True)
            )

        # Store everything.
        ctx.save_for_backward(*args)

        return outputs

    @staticmethod
    def backward(ctx, *args):
        if not torch.autograd._is_checkpoint_valid():
            raise RuntimeError(
                "Checkpointing is not compatible with .grad(), "
                "please use .backward() if possible"
            )
        inputs = ctx.saved_tensors
        if ctx.distribute_saved_activations:
            safely_set_viewless_tensor_data(
                inputs[0], gather_split_1d_tensor(inputs[0].data).view(ctx.input_0_shape)
            )

        # Store the current states.
        bwd_cpu_rng_state = torch.get_rng_state()
        bwd_cuda_rng_state = torch.musa.get_rng_state()
        bwd_cuda_rng_state_tracker = get_cuda_rng_tracker().get_states()

        # Set the states to what it used to be before the forward pass.
        torch.set_rng_state(ctx.fwd_cpu_rng_state)
        _set_cuda_rng_state(ctx.fwd_cuda_rng_state)
        get_cuda_rng_tracker().set_states(ctx.fwd_cuda_rng_state_tracker)

        # Compute the forward pass.
        detached_inputs = detach_variable(inputs)
        with torch.enable_grad():
            outputs = ctx.run_function(*detached_inputs)

        # Set the states back to what it was at the start of this function.
        torch.set_rng_state(bwd_cpu_rng_state)
        _set_cuda_rng_state(bwd_cuda_rng_state)
        get_cuda_rng_tracker().set_states(bwd_cuda_rng_state_tracker)

        if isinstance(outputs, torch.Tensor):
            outputs = (outputs,)
        torch.autograd.backward(outputs, args)
        grads = tuple(inp.grad if isinstance(inp, torch.Tensor) else inp for inp in detached_inputs)
        return (None, None) + grads

megatron.core.tensor_parallel.random._set_cuda_rng_state = _set_cuda_rng_state
megatron.core.tensor_parallel.random.CudaRNGStatesTracker.add = CudaRNGStatesTracker_add
megatron.core.tensor_parallel.random.CudaRNGStatesTracker.fork = CudaRNGStatesTracker_fork
megatron.core.tensor_parallel.random.CheckpointFunction = CheckpointFunction