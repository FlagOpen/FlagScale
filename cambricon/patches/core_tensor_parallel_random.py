import contextlib
import torch
from torch import _C
import torch_mlu
from torch.mlu import _lazy_call
from torch.mlu import device as device_ctx_manager
import megatron
from megatron.core.parallel_state import (
    get_global_memory_buffer,
    get_tensor_model_parallel_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from megatron.core.tensor_parallel.random import CudaRNGStatesTracker

# Default name for the model parallel rng tracker.
_MODEL_PARALLEL_RNG_TRACKER_NAME = 'model-parallel-rng'
# RNG tracker object.
_CUDA_RNG_STATE_TRACKER = CudaRNGStatesTracker()


def _set_cuda_rng_state(new_state, device=-1):
    """Sets the random number generator state of the current GPU.

    Argumentss:
        new_state (torch.ByteTensor): The desired state
    This function is adapted from PyTorch repo (torch.mlu.set_rng_state)
    with a single change: the input state is not cloned. Cloning caused
    major performance issues for +4 GPU cases.
    """
    if hasattr(_C, '_mlu_setRNGState') and callable(_C._mlu_setRNGState):
        # older PyTorch
        def cb():
            with device_ctx_manager(device):
                _C._mlu_setRNGState(new_state)

    else:
        if device == -1:
            device = torch.device('mlu')
        elif isinstance(device, str):
            device = torch.device(device)
        elif isinstance(device, int):
            device = torch.device('mlu', device)

        def cb():
            idx = device.index
            if idx is None:
                idx = torch.mlu.current_device()
            default_generator = torch.mlu.default_generators[idx]
            default_generator.set_state(new_state)

        _lazy_call(cb)

def add(self, name, seed):
    """Track the rng state."""
    # Check seed is not already used.
    if seed in self.seeds_:
        raise Exception('seed {} already exists'.format(seed))
    self.seeds_.add(seed)
    # Check that state is not already defined.
    if name in self.states_:
        raise Exception('mlu rng state {} already exists'.format(name))
    # Get the current rng state.
    orig_rng_state = torch.mlu.get_rng_state()
    # Set the new state and store it.
    torch.mlu.manual_seed(seed)
    self.states_[name] = torch.mlu.get_rng_state()
    # Reset rng state to what it was.
    _set_cuda_rng_state(orig_rng_state)

@contextlib.contextmanager
def fork(self, name=_MODEL_PARALLEL_RNG_TRACKER_NAME):
    """Fork the mlu rng state, perform operations, and exit with
    the original state."""
    # Check if we have added the state
    if name not in self.states_:
        raise Exception('mlu rng state {} is not added'.format(name))
    # Store current rng state.
    orig_mlu_rng_state = torch.mlu.get_rng_state()
    # Set rng state to the desired one
    _set_cuda_rng_state(self.states_[name])
    # Do the stuff we wanted to do.
    try:
        yield
    finally:
        # Update the current rng state for later use.
        self.states_[name] = torch.mlu.get_rng_state()
        # And set the state to the original state we started with.
        _set_cuda_rng_state(orig_mlu_rng_state)

def CheckpointFunctionBackward(ctx, *args):
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
    bwd_mlu_rng_state = torch.mlu.get_rng_state()
    bwd_mlu_rng_state_tracker = get_cuda_rng_tracker().get_states()

    # Set the states to what it used to be before the forward pass.
    torch.set_rng_state(ctx.fwd_cpu_rng_state)
    _set_cuda_rng_state(ctx.fwd_mlu_rng_state)
    get_cuda_rng_tracker().set_states(ctx.fwd_mlu_rng_state_tracker)

    # Compute the forward pass.
    detached_inputs = detach_variable(inputs)
    with torch.enable_grad():
        outputs = ctx.run_function(*detached_inputs)

    # Set the states back to what it was at the start of this function.
    torch.set_rng_state(bwd_cpu_rng_state)
    _set_cuda_rng_state(bwd_mlu_rng_state)
    get_cuda_rng_tracker().set_states(bwd_mlu_rng_state_tracker)

    if isinstance(outputs, torch.Tensor):
        outputs = (outputs,)
    torch.autograd.backward(outputs, args)
    grads = tuple(inp.grad if isinstance(inp, torch.Tensor) else inp for inp in detached_inputs)
    return (None, None) + grads

megatron.core.tensor_parallel.random._set_cuda_rng_state = _set_cuda_rng_state
megatron.core.tensor_parallel.random.CudaRNGStatesTracker.add = add 
megatron.core.tensor_parallel.random.CudaRNGStatesTracker.fork = fork 
megatron.core.tensor_parallel.random.CheckpointFunction.backward = CheckpointFunctionBackward 
