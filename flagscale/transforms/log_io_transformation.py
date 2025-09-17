from typing import Any, Dict, Tuple

import torch

from torch import nn

from flagscale.runner.utils import logger
from flagscale.transforms.hook import ModelHook, ModuleHookRegistry
from flagscale.transforms.transformation import Transformation


class LogIOHook(ModelHook):
    """A simple hook that logs the input shapes of a module.
    Only used for debugging. Will be removed in the future.
    """

    def __init__(self, log_level: str = "info") -> None:
        """Initialize the hook.

        Args:
            log_level: The log level to use. It must be a valid logger level.
        """

        super().__init__()
        if not hasattr(logger, log_level) or not callable(getattr(logger, log_level)):
            raise ValueError(f"Invalid log level: {log_level}")
        self._logger_func = getattr(logger, log_level)

    def pre_forward(self, module: nn.Module, *args, **kwargs) -> Tuple[Tuple[Any], Dict[str, Any]]:
        def shape_of(x: torch.Tensor) -> str:
            return getattr(x, "shape", type(x).__name__)

        self._logger_func(
            f"[LogIOHook] {module.__class__.__name__} input shapes: "
            f"{tuple(shape_of(a) for a in args)}"
        )
        return args, kwargs


class LogIOTransformation(Transformation):
    """A transform that logs the input shapes of a module. Just to showcase the transform API."""

    def __init__(self, log_level: str = "info") -> None:
        """Initialize the transform.

        Args:
            log_level: The log level to use. It must be a valid logger level.
        """

        super().__init__()

        self._log_level = log_level

    def apply(self, model: nn.Module) -> bool:
        reg = ModuleHookRegistry.get_or_create_registry(model)
        hook = LogIOHook(log_level=self._log_level)
        reg.register_hook(hook, "log_io")
        return True
