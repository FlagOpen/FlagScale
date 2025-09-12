from typing import Iterable

from torch import nn

from .base_adapter import BaseAdapter


class GenericDiffusersAdapter(BaseAdapter):
    """
    A generic adapter for common Diffusers models.
    """

    def backbone(self) -> nn.Module | Iterable[nn.Module]:
        if hasattr(self._model, "unet"):
            assert isinstance(
                self._model.unet, nn.Module
            ), f"UNet should be a `nn.Module`, but got {type(self._model.unet)}"
            return self._model.unet
        if hasattr(self._model, "transformer"):
            assert isinstance(
                self._model.transformer, nn.Module
            ), f"transformer should be a `nn.Module`, but got {type(self._model.transformer)}"
            return self._model.transformer
        # TODO(yupu): add support for other models
        raise ValueError(f"Model {self._model} has no unet or transformer.")
