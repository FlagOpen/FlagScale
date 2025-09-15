from typing import Dict, List, Type

from omegaconf import DictConfig

from .infer.log_io import LogIOTransform
from .infer.state_context_transform import StateContextTransform
from .transform import Transform
from .transform_manager import TransformManager

# Registry of supported Transform classes by their class names.
TRANSFORM_REGISTRY: Dict[str, Type[Transform]] = {
    "LogIOTransform": LogIOTransform,
    "StateContextTransform": StateContextTransform,
}

__all__ = ["create_transforms_from_config", "TransformManager"]


def create_transforms_from_config(cfg: DictConfig) -> List[Transform]:
    """Instantiate transforms from the configuration

    Args:
        cfg: The configuration

    Returns:
        A list of instantiated transforms
    """

    instances: List[Transform] = []

    for name, kwargs in cfg.items():
        cls = TRANSFORM_REGISTRY.get(name)
        if cls is None:
            raise KeyError(
                f"Unknown transform class '{name}'. Available: {sorted(TRANSFORM_REGISTRY.keys())}"
            )
        # TODO(yupu): Maybe we should ignore unknown kwargs?
        try:
            print(f"Creating transform {name} with kwargs {kwargs}")
            inst = cls(**kwargs)
        except TypeError as e:
            raise TypeError(
                f"Failed to instantiate transform '{name}' with kwargs {kwargs}: {e}"
            ) from e
        instances.append(inst)

    return instances
