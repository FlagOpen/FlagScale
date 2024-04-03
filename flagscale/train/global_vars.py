import torch

_GLOBAL_EXTRA_VALID_DATASETS = None
_GLOBAL_EXATRA_INPUT_TENSOR = None


def get_extra_valid_datasets():
    """Return extra_valid datasets."""""
    return _GLOBAL_EXTRA_VALID_DATASETS


def set_extra_valid_datasets(extra_valid_datasets):
    """Initialize heterogenous context."""""
    global _GLOBAL_EXTRA_VALID_DATASETS
    _GLOBAL_EXTRA_VALID_DATASETS = extra_valid_datasets


def get_extra_input_tensor():
    return _GLOBAL_EXATRA_INPUT_TENSOR


def set_extra_input_tensor(input_tensor : torch.Tensor):
    global _GLOBAL_EXATRA_INPUT_TENSOR
    _GLOBAL_EXATRA_INPUT_TENSOR = input_tensor

