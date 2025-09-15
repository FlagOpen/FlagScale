from typing import Any, Dict, Tuple

import torch.nn as nn

from flagscale.inference.runtime_context import current_ctx
from flagscale.transforms.hook import ModelHook, ModuleHookRegistry
from flagscale.transforms.transformation import Transformation


class StateContextHook(ModelHook):
    """A hook that sets the state context for the current module."""

    def __init__(self) -> None:
        super().__init__()

    def pre_forward(self, module: nn.Module, *args, **kwargs) -> Tuple[Tuple[Any], Dict[str, Any]]:
        """Get the state context from the runtime context and set it for the current module recursively.
        For this hook to work properly, the transform must be applied to the root module and
        the state context must be set in the runtime context (by some other transform).
        """

        ctx = current_ctx()
        if ctx:
            state_ctx = ctx.state_ctx
            print(f"state_ctx: {state_ctx}")
            if state_ctx:
                ModuleHookRegistry.get_or_create_registry(module).set_state_context(state_ctx)
        return args, kwargs

    def post_forward(self, module: nn.Module, output: Any) -> Any:
        """Reset the state context for the current module."""

        ctx = current_ctx()
        if ctx:
            ModuleHookRegistry.get_or_create_registry(module).set_state_context(None)

        return output


class StateContextTransformation(Transformation):
    """A transform that sets the state context."""

    def __init__(self) -> None:
        super().__init__()

    def apply(self, model: nn.Module) -> bool:
        reg = ModuleHookRegistry.get_or_create_registry(model)
        hook = StateContextHook()
        reg.register_hook(hook, "state_context")
        return True
