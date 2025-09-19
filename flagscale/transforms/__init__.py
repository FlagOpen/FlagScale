from typing import Dict, List, Type

from omegaconf import DictConfig

from .log_io_transformation import LogIOTransformation
from .state_scope_transformation import StateScopeTransformation
from .taylorseer_transformation import TaylorSeerTransformation
from .timestep_tracker_transformation import TimestepTrackerTransformation
from .transformation import Transformation

# Registry of supported Transformation classes by their class names.
_TRANSFORMATION_REGISTRY: Dict[str, Type[Transformation]] = {
    "LogIOTransformation": LogIOTransformation,
    "StateScopeTransformation": StateScopeTransformation,
    "TimestepTrackerTransformation": TimestepTrackerTransformation,
    "TaylorSeerTransformation": TaylorSeerTransformation,
}

__all__ = ["create_transformations_from_config", "create_default_transformations"]


def create_transformations_from_config(cfg: DictConfig) -> List[Transformation]:
    """Instantiate transformations from the configuration

    Args:
        cfg: The configuration

    Returns:
        A list of instantiated transformations
    """

    instances: List[Transformation] = []

    for name, kwargs in cfg.items():
        cls = _TRANSFORMATION_REGISTRY.get(name)
        if cls is None:
            raise KeyError(
                f"Unknown transformation class '{name}'. Available: {sorted(_TRANSFORMATION_REGISTRY.keys())}"
            )
        try:
            if kwargs is None:
                kwargs = {}
            inst = cls(**kwargs)
        except TypeError as e:
            raise TypeError(
                f"Failed to instantiate transformation '{name}' with kwargs {kwargs}: {e}"
            ) from e
        instances.append(inst)

    return instances


def create_default_transformations() -> List[Transformation]:
    return [TimestepTrackerTransformation()]
