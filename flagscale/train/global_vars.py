import torch

from flagscale.train.hetero.parallel_context import ParallelContext

_GLOBAL_EXTRA_VALID_DATASETS = None
_GLOBAL_EXATRA_INPUT_TENSOR = None
_GLOBAL_PARALLEL_CONTEXT = None


def _ensure_var_is_initialized(var, name):
    """Make sure the input variable is not None."""
    # Refer to the same function from megatron/megatron/training/global_vars.py
    assert var is not None, '{} is not initialized.'.format(name)


def _ensure_var_is_not_initialized(var, name):
    """Make sure the input variable is not None."""
    # Refer to the same function from megatron/megatron/training/global_vars.py
    assert var is None, '{} is already initialized.'.format(name)


def get_extra_valid_datasets():
    """Return extra_valid datasets."""""
    return _GLOBAL_EXTRA_VALID_DATASETS


def set_extra_valid_datasets(extra_valid_datasets):
    """Set extra_valid datasets."""""
    global _GLOBAL_EXTRA_VALID_DATASETS
    _GLOBAL_EXTRA_VALID_DATASETS = extra_valid_datasets


def get_extra_input_tensor():
    return _GLOBAL_EXATRA_INPUT_TENSOR


def set_extra_input_tensor(input_tensor : torch.Tensor):
    global _GLOBAL_EXATRA_INPUT_TENSOR
    _GLOBAL_EXATRA_INPUT_TENSOR = input_tensor


def get_parallel_context():
    """Return heterogenous parallel context."""
    _ensure_var_is_initialized(_GLOBAL_PARALLEL_CONTEXT, 'parallel context')
    return _GLOBAL_PARALLEL_CONTEXT


def set_parallel_context(args):
    """Initialize heterogenous parallel context."""
    global _GLOBAL_PARALLEL_CONTEXT
    _ensure_var_is_not_initialized(_GLOBAL_PARALLEL_CONTEXT, 'parallel context')
    _GLOBAL_PARALLEL_CONTEXT = ParallelContext(args)
