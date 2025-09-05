from typing import Optional

from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from torch import nn

from .base_adapter import BaseAdapter
from .generic_adapter import GenericDiffusersAdapter

__all__ = ["BaseAdapter", "GenericDiffusersAdapter", "create_adapter"]

_ADAPTERS: dict[str, type[BaseAdapter]] = {"generic": GenericDiffusersAdapter}


def _get_adapter_class(name: Optional[str]) -> type[BaseAdapter]:
    """Get the adapter class type for the given name.

    Args:
        name: The name of the adapter to get. If None, `GenericDiffusersAdapter` will be returned.

    Returns:
        The adapter class type for the given name.
    """

    if name is None:
        return GenericDiffusersAdapter
    if name in _ADAPTERS:
        return _ADAPTERS[name]
    else:
        raise ValueError(f"Adapter {name} is not supported")


def create_adapter(
    name: Optional[str], model_or_pipeline: nn.Module | DiffusionPipeline
) -> BaseAdapter:
    """Create an adapter for the given model or pipeline.

    Args:
        name: The name of the adapter to create. If None, a `GenericDiffusersAdapter` will be created.
        model_or_pipeline: The model or pipeline to create an adapter for.

    Returns:
        An adapter for the given model or pipeline.
    """

    adapter_class = _get_adapter_class(name)
    return adapter_class(model_or_pipeline)
