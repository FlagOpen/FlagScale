import torch
from contextlib import contextmanager


@contextmanager
def suspend_nn_inits():
    """
    see https://github.com/huggingface/transformers/issues/26258
    """
    skip = lambda *args, **kwargs: None
    saved_inits = torch.nn.init.kaiming_uniform_, torch.nn.init.uniform_, torch.nn.init.normal_, torch.nn.init.xavier_uniform_  # saving
    torch.nn.init.kaiming_uniform_ = torch.nn.init.uniform_ = torch.nn.init.normal_  = torch.nn.init.xavier_uniform_ = skip  # replacing
    try:
        yield
    finally:
        torch.nn.init.kaiming_uniform_, torch.nn.init.uniform_, torch.nn.init.normal_, torch.nn.init.xavier_uniform_ = saved_inits  # restoring
