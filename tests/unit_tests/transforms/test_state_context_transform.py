import unittest

import torch

from torch import nn

from flagscale.engine.runtime_context import RuntimeContext
from flagscale.models.adapters.generic_adapter import GenericDiffusersAdapter
from flagscale.transforms.hook import ModuleHookRegistry
from flagscale.transforms.infer.state_context_transform import (
    StateContextHook,
    StateContextTransform,
)
from flagscale.transforms.state_store import ContextStateStore


class DummyPipeline:
    def __init__(self) -> None:
        self.unet = nn.Sequential(nn.Linear(2, 2))


class TestStateContextTransform(unittest.TestCase):
    def test_apply_registers_hook_on_backbone(self):
        adapter = GenericDiffusersAdapter(DummyPipeline())
        transform = StateContextTransform()

        applied = transform.apply(adapter)
        self.assertTrue(applied)

        reg = ModuleHookRegistry.get_registry_if_present(adapter.backbone())
        self.assertIsNotNone(reg)
        self.assertIsInstance(reg.get_hook("state_context"), StateContextHook)

    def test_hook_sets_and_resets_state_context_during_forward(self):
        adapter = GenericDiffusersAdapter(DummyPipeline())
        transform = StateContextTransform()
        transform.apply(adapter)

        backbone = adapter.backbone()
        reg = ModuleHookRegistry.get_registry_if_present(backbone)

        # Add an observing hook with a ContextStateStore to capture set_state_context calls
        store = ContextStateStore(dict)  # any state type, we only observe set calls
        observe_hook = StateContextHook()
        # Piggyback the store into the stateful list of the registered hook
        # so that set_state_context triggers ContextStateStore.set(...)
        reg.get_hook("state_context").register_stateful(store)

        # Build a simple input and run forward inside a runtime context providing a state ctx
        x = torch.zeros(1, 2)
        ctx = RuntimeContext()
        ctx.state_ctx_provider = lambda: "ctxA"
        with ctx.session():
            _ = backbone(x)

        # After forward, the hook should have called set_state_context with ctxA and then None
        # Validate via the store having created states for the two contexts
        # First call (ctxA) -> state created; second call (None) should not create a state
        self.assertIn("ctxA", store._state_by_context)

        # No context named None should be tracked
        self.assertNotIn(None, store._state_by_context)
