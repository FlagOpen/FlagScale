import unittest

from typing import Any, Dict, Tuple

import torch

from torch import nn

from flagscale.inference.runtime_context import RuntimeContext
from flagscale.transforms.context_state_store import ContextStateStore
from flagscale.transforms.hook import ModelHook, ModuleHookRegistry
from flagscale.transforms.state_context_transformation import (
    StateContextHook,
    StateContextTransformation,
)


class DummyPipeline:
    def __init__(self) -> None:
        self.unet = nn.Sequential(nn.Linear(2, 2))


class DummyHook(ModelHook):
    def __init__(self) -> None:
        super().__init__()

    def pre_forward(self, module: nn.Module, *args, **kwargs) -> Tuple[Tuple[Any], Dict[str, Any]]:
        if self._stateful:
            self._stateful[0].get_or_create_state()
        return args, kwargs


class TestStateContextTransform(unittest.TestCase):
    def test_apply_registers_hook_on_backbone(self):
        pipeline = DummyPipeline()
        transform = StateContextTransformation()

        applied = transform.apply(pipeline.unet)
        self.assertTrue(applied)

        reg = ModuleHookRegistry.get_registry_if_present(pipeline.unet)
        self.assertIsNotNone(reg)
        self.assertIsInstance(reg.get_hook("state_context"), StateContextHook)

    def test_hook_sets_and_resets_state_context_during_forward(self):
        pipeline = DummyPipeline()
        backbone = pipeline.unet

        reg = ModuleHookRegistry.get_or_create_registry(backbone)
        store = ContextStateStore(dict)

        # To make it work, we need to register the dummy hook first
        hook = DummyHook()
        reg.register_hook(hook, "dummy")
        hook.register_stateful(store)

        transform = StateContextTransformation()
        transform.apply(backbone)

        x = torch.zeros(1, 2)
        ctx = RuntimeContext()
        ctx.state_ctx_provider = lambda: "ctxA"
        with ctx.session():
            _ = backbone(x)

        self.assertIn("ctxA", store._state_by_context)
