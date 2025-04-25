import torch

from flagscale.train.hetero.parallel_context import ParallelContext
from flagscale.train.spiky_loss import SpikyLossDetector

_GLOBAL_EXTRA_VALID_DATASETS = None
_GLOBAL_PARALLEL_CONTEXT = None
_GLOBAL_SPIKY_LOSS_DETECTOR = None


def _ensure_var_is_initialized(var, name):
    """Make sure the input variable is not None."""
    # Refer to the same function from megatron/megatron/training/global_vars.py
    assert var is not None, "{} is not initialized.".format(name)


def _ensure_var_is_not_initialized(var, name):
    """Make sure the input variable is not None."""
    # Refer to the same function from megatron/megatron/training/global_vars.py
    assert var is None, "{} is already initialized.".format(name)


def get_extra_valid_datasets():
    """Return extra_valid datasets.""" ""
    return _GLOBAL_EXTRA_VALID_DATASETS


def set_extra_valid_datasets(extra_valid_datasets):
    """Set extra_valid datasets.""" ""
    global _GLOBAL_EXTRA_VALID_DATASETS
    _GLOBAL_EXTRA_VALID_DATASETS = extra_valid_datasets


def get_parallel_context():
    """Return heterogenous parallel context."""
    return _GLOBAL_PARALLEL_CONTEXT


def set_parallel_context(args):
    """Initialize heterogenous parallel context."""
    global _GLOBAL_PARALLEL_CONTEXT
    _ensure_var_is_not_initialized(_GLOBAL_PARALLEL_CONTEXT, "parallel context")
    _GLOBAL_PARALLEL_CONTEXT = ParallelContext(args)


def get_spiky_loss_detector():
    """Return spiky loss detector."""
    _ensure_var_is_initialized(_GLOBAL_SPIKY_LOSS_DETECTOR, "spiky loss detector")
    return _GLOBAL_SPIKY_LOSS_DETECTOR


def set_get_spiky_loss_detector(args):
    """Initialize spiky loss detector."""
    global _GLOBAL_SPIKY_LOSS_DETECTOR
    _ensure_var_is_not_initialized(_GLOBAL_SPIKY_LOSS_DETECTOR, "spiky loss detector")
    _GLOBAL_SPIKY_LOSS_DETECTOR = SpikyLossDetector(args.spiky_loss_threshold)
