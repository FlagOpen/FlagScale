from typing import Any, Dict, Tuple

import torch.nn as nn

from flagscale.inference.runtime_context import current_ctx
from flagscale.transforms.hook import ModelHook, ModuleHookRegistry
from flagscale.transforms.transformation import Transformation


class StateScopeHook(ModelHook):
    """A hook that sets the state scope for the current module."""

    def __init__(self) -> None:
        super().__init__()

    def pre_forward(self, module: nn.Module, *args, **kwargs) -> Tuple[Tuple[Any], Dict[str, Any]]:
        """Get the state context from the runtime context and set it for the current module recursively.
        For this hook to work properly, the transform must be applied to the root module and
        the state scope must be set in the runtime context (by some other transform).
        """

        ctx = current_ctx()
        if ctx:
            state_scope = ctx.state_scope
            if state_scope:
                ModuleHookRegistry.get_or_create_registry(module).set_state_scope(state_scope)
        return args, kwargs

    def post_forward(self, module: nn.Module, output: Any) -> Any:
        """Reset the state scope for the current module."""

        ctx = current_ctx()
        if ctx:
            ModuleHookRegistry.get_or_create_registry(module).set_state_scope(None)

        return output


class StateScopeTransformation(Transformation):
    """A transform that sets the state scope."""

    def __init__(self) -> None:
        super().__init__()

    def apply(self, model: nn.Module) -> bool:
        reg = ModuleHookRegistry.get_or_create_registry(model)
        hook = StateScopeHook()
        reg.register_hook(hook, "state_scope")
        return True
