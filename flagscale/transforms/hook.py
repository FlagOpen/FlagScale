# Modified from https://github.com/huggingface/diffusers/blob/4a7556eaecc9872dea50ce161301edfa6392693c/src/diffusers/hooks/hooks.py

import functools

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from torch import nn


# TODO(yupu): Implement state and context
class ModelHook:
    """
    Hook applied to a module.
    """

    # TODO(yupu): Check if this is needed, should we put all state inside the `RuntimeContext`?
    # _is_stateful: bool = False

    def __init__(self) -> None:
        self.fn_ref: "HookFunctionReference" = None

    def on_attach(self, module: nn.Module) -> nn.Module:
        """
        Called when the hook is attached to a module.
        """
        return module

    def on_detach(self, module: nn.Module) -> nn.Module:
        """
        Called when the hook is detached from a module.
        """
        return module

    # TODO(yupu): See if we need to explicitly pass a `RuntimeContext`
    def pre_forward(self, module: nn.Module, *args, **kwargs) -> Tuple[Tuple[Any], Dict[str, Any]]:
        """
        Called before the module's forward pass.
        """
        return args, kwargs

    def post_forward(self, module: nn.Module, output: Any) -> Any:
        """
        Called after the module's forward pass.
        """
        return output


@dataclass
class HookFunctionReference:
    """
    Mutable references for spliceable chains
    """

    pre_forward: Optional[Callable[..., Any]] = None
    post_forward: Optional[Callable[..., Any]] = None
    forward: Optional[Callable[..., Any]] = None
    original_forward: Optional[Callable[..., Any]] = None


class ModuleHookRegistry:
    """Per-module registry. Attaching a hook immediately wraps module's forward"""

    def __init__(self, module_ref: nn.Module) -> None:
        # A reference to the module
        self._module_ref = module_ref
        # name -> hook
        self._hooks: Dict[str, ModelHook] = {}
        # The order of hooks registered
        self._order: List[str] = []
        # A ordered list of applied forward functions for each hook
        self._fn_refs: List[HookFunctionReference] = []

    @classmethod
    def get_or_create_registry(cls, module: nn.Module) -> "ModuleHookRegistry":
        """
        Get or create a registry for the module.

        Args:
            module: The module to get or create a registry for.

        Returns:
            The registry for the module.
        """
        reg = ModuleHookRegistry.get_registry_if_present(module)
        if reg is None:
            reg = cls(module)
            setattr(module, "_flagscale_hooks", reg)
        return reg

    @classmethod
    def get_registry_if_present(cls, module: nn.Module) -> Optional["ModuleHookRegistry"]:
        """Return existing registry if present, without creating a new one."""
        reg = vars(module).get("_flagscale_hooks")
        return reg if isinstance(reg, ModuleHookRegistry) else None

    def register_hook(self, hook: ModelHook, name: str) -> None:
        """Register a hook on the module.

        Args:
            hook: The hook to register.
            name: The name of the hook, which should be unique within the registry.
        """
        if name in self._hooks:
            raise ValueError(
                f"Hook with name {name} already exists in the registry. Please use a different name."
            )

        # Let hook adjust module if needed
        self._module_ref = hook.on_attach(self._module_ref)

        fn_ref = HookFunctionReference(
            pre_forward=hook.pre_forward,
            post_forward=hook.post_forward,
            forward=self._module_ref.forward,
        )

        # TODO(yupu): This implicitly requires a `custom_forward` method to be defined in the hook. Make it explicit.
        # If hook provides a custom forward, use it instead and keep the original forward
        if hasattr(hook, "custom_forward"):
            fn_ref.original_forward = self._module_ref.forward
            custom = functools.update_wrapper(
                functools.partial(hook.custom_forward, self._module_ref), hook.custom_forward
            )
            fn_ref.forward = custom

        # Build the wrapped forward for this hook
        def make_wrapped(fn_ref: HookFunctionReference):
            def wrapped(module: nn.Module, *args, **kwargs):
                args, kwargs = fn_ref.pre_forward(module, *args, **kwargs)
                output = fn_ref.forward(*args, **kwargs)
                return fn_ref.post_forward(module, output)

            return wrapped

        rewritten = make_wrapped(fn_ref)
        new_forward = functools.update_wrapper(
            functools.partial(rewritten, self._module_ref), rewritten
        )
        self._module_ref.forward = new_forward

        # Track ordering and links
        hook.fn_ref = fn_ref
        setattr(fn_ref, "_name", name)
        self._hooks[name] = hook
        self._order.append(name)
        self._fn_refs.append(fn_ref)

    def get_hook(self, name: str) -> Optional[ModelHook]:
        """Get a hook by name.

        Args:
            name: The name of the hook.

        Returns:
            The hook, or None if the hook is not found.
        """
        return self._hooks.get(name, None)

    def remove_hook(self, name: str, recursive: bool = True) -> None:
        """Remove a hook by name.

        Args:
            name: The name of the hook.
            recursive: Whether to remove the hook recursively.
        """
        if name not in self._hooks:
            return

        hook = self._hooks[name]
        idx = self._order.index(name)
        fn_ref = self._fn_refs[idx]

        # Determine which forward to restore at this splice point
        old_forward = (
            fn_ref.original_forward if fn_ref.original_forward is not None else fn_ref.forward
        )

        # Splice: if removing the last link, restore module.forward,
        # otherwise point the next link's forward to the restored function
        if idx == len(self._fn_refs) - 1:
            self._module_ref.forward = old_forward
        else:
            self._fn_refs[idx + 1].forward = old_forward

        # Allow hook to clean up and possibly adjust module
        self._module_ref = hook.on_detach(self._module_ref)

        # Drop bookkeeping
        self._hooks.pop(name, None)
        self._order.pop(idx)
        self._fn_refs.pop(idx)

        if recursive:
            for module_name, module in self._module_ref.named_modules():
                if module_name == "":
                    continue
                reg = ModuleHookRegistry.get_registry_if_present(module)
                if reg is not None:
                    reg.remove_hook(name, recursive=False)

    # TODO(yupu): State may be consumed by the hook, but by definition, it should be a `Transform`'s attribute.
    # TODO(yupu): Is it possible in reality to have multiple contexts for different hooks at the same time?

    def __repr__(self) -> str:
        registry_repr = ""
        for i, hook_name in enumerate(self._order):
            if self._hooks[hook_name].__class__.__repr__ is not object.__repr__:
                hook_repr = self._hooks[hook_name].__repr__()
            else:
                hook_repr = self._hooks[hook_name].__class__.__name__
            registry_repr += f"  ({i}) {hook_name} - {hook_repr}"
            if i < len(self._order) - 1:
                registry_repr += "\n"
        return f"ModuleHookRegistry(\n{registry_repr}\n)"
