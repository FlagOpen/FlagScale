from abc import ABC, abstractmethod
from typing import Optional, Set

import torch
import torch.nn as nn
from torch.nn import Module
try:
    import transformers
except ImportError as err:
    transformers = None
    transformers_err = err

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

__all__ = ["BlockCompressionWrapper"]

class BlockCompressionWrapper(Module, ABC):
    def __init__(self, name, layer):
        super(BlockCompressionWrapper, self).__init__()
        if transformers is None:
            raise transformers_err

        self.name = name
        self.layer = layer

        self.dev = next(self.layer.parameters()).device
        if hasattr(self.layer, "_hf_hook") and self.layer._hf_hook.offload:
            self.dev = self.layer._hf_hook.execution_device

        # This need to be a buffer so its preserved between forward passes
        self.register_buffer(
            "nsamples", torch.zeros(1, dtype=torch.int32, device=self.dev)
        )

    def forward(self, *args, **kwargs):
        """
        Run a forward pass of the wrapped layer
        """
        return self.layer(*args, **kwargs)

    def free(self):
        """
        Free buffers used for compression
        """
        delattr(self, "nsamples")

    @abstractmethod
    def add_batch(self, *args, **kwargs):
        """
        Add a batch of layer input and output data to the layer statistics calculation
        """
        raise NotImplementedError("Child class must implement `add_batch`")

    @abstractmethod
    def compress(self, *args, **kwargs):
        """
        Run pruning on the layer up to the target sparsity
        """
        raise NotImplementedError("Child class must implement `compress`")

    def state_dict(self, destination=None, prefix="", keep_vars=False, **kwargs):
        """
        Pass request to wrapped layer, so compression wrapper does not appear in
        the state_dict
        """
        return self.layer.state_dict(
            destination=destination, prefix=prefix, keep_vars=keep_vars, **kwargs
        )

    def load_state_dict(self, state_dict, strict=True):
        """
        Pass request to wrapped layer, so compression wrapper does not appear in
        the state_dict
        """
        return self.layer.load_state_dict(state_dict, strict=strict)

    def named_modules(
        self,
        memo: Optional[Set["Module"]] = None,
        prefix: str = "",
        remove_duplicate: bool = True,
    ):
        """
        Pass request to wrapped layer, so compression wrapper does not appear in
        the module list
        """
        return self.layer.named_modules(
            memo=memo, prefix=prefix, remove_duplicate=remove_duplicate
        )