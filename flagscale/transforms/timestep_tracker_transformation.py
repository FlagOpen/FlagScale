import inspect

from typing import Union

import torch
import torch.nn as nn

from flagscale.inference.runtime_context import current_ctx
from flagscale.transforms.hook import ModelHook, ModuleHookRegistry
from flagscale.transforms.transformation import Transformation


# TODO(yupu): Check if this hook is general and reliable
class TimestepTrackerHook(ModelHook):
    def __init__(self):
        super().__init__()
        self._sig = None

    def pre_forward(self, module: nn.Module, *args, **kwargs):
        ts: Union[torch.Tensor, float, int] = kwargs.get("timestep")
        if not ts:
            if not self._sig:
                self._sig = inspect.signature(module.__class__.forward)
            bound = self._sig.bind_partial(module, *args, **kwargs)
            ts = bound.arguments.get("timestep")
        if not ts:
            raise ValueError("Failed to get `timestep` from model inputs")

        ctx = current_ctx()
        if ctx:
            ctx.update_timestep(ts)
        else:
            raise ValueError("No context found")

        return args, kwargs


class TimestepTrackerTransformation(Transformation):
    def __init__(self):
        super().__init__()

    def apply(self, model: nn.Module) -> bool:
        reg = ModuleHookRegistry.get_or_create_registry(model)
        hook = TimestepTrackerHook()
        reg.register_hook(hook, "timestep_tracker")
        return True
