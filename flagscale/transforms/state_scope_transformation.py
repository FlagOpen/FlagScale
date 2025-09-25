from typing import Any, Dict, Tuple

import torch.nn as nn

from flagscale.inference.runtime_context import current_ctx
from flagscale.runner.utils import logger
from flagscale.transforms.hook import ModelHook, ModuleHookRegistry
from flagscale.transforms.transformation import Transformation


# TODO(yupu): should always applied last
class StateScopeHook(ModelHook):
    """A hook that sets the state scope for the current module."""

    def pre_forward(self, module: nn.Module, *args, **kwargs) -> Tuple[Tuple[Any], Dict[str, Any]]:
        """Get the state context from the runtime context and set it for the current module recursively.
        For this hook to work properly, the transform must be applied to the root module and
        the state scope must be set in the field `engine_config.state_scopes`.
        """

        ctx = current_ctx()
        if ctx:
            state_scope = ctx.state_scope
            if state_scope:
                ModuleHookRegistry.get_or_create_registry(module).set_state_scope(state_scope)
            else:
                logger.warning(
                    "No `state_scope` found in the runtime context. "
                    "Please set the `state_scopes` in the `engine_config`."
                    " e.g. `state_scopes: ['uncond', 'cond']`"
                )
        return args, kwargs

    def post_forward(self, module: nn.Module, output: Any) -> Any:
        """Reset the state scope for the current module."""

        ctx = current_ctx()
        if ctx:
            ModuleHookRegistry.get_or_create_registry(module).set_state_scope(None)

        return output


class StateScopeTransformation(Transformation):
    """A transform that sets the state scope."""

    def apply(self, model: nn.Module) -> bool:
        """
        Register a hook to set the state scope for the current module recursively.

        Args:
            model: The root module to apply the transformation to.

        Returns:
            True if the transformation is applied successfully.
        """

        reg = ModuleHookRegistry.get_or_create_registry(model)
        hook = StateScopeHook()
        reg.register_hook(hook, "state_scope")
        return True
