from contextlib import contextmanager

import torch.nn as nn

from flagscale.transforms.hooks import ModuleHookRegistry


def set_state_context(root: nn.Module, name: str | None) -> None:
    for _, mod in root.named_modules():
        reg = getattr(mod, "_flagscale_hooks", None)
        if isinstance(reg, ModuleHookRegistry):
            reg.set_state_context(name)


# TODO(yupu): We could also reigster a context state switcher as a hook.
@contextmanager
def state_context(root: nn.Module, name: str | None):
    set_state_context(root, name)
    try:
        yield
    finally:
        # TODO(yupu): restore a previous name?
        set_state_context(root, None)
