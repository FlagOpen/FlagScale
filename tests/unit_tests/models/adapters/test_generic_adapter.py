import unittest

from torch import nn

from flagscale.models.adapters.generic_adapter import GenericDiffusersAdapter


class TestGenericDiffusersAdapter(unittest.TestCase):
    def test_backbone_returns_unet_when_present_and_correct_type(self):
        class DummyPipeline:
            def __init__(self) -> None:
                self.unet = nn.Module()

        pipeline = DummyPipeline()
        adapter = GenericDiffusersAdapter(pipeline)
        backbone = adapter.backbone()
        self.assertIs(backbone, pipeline.unet)

    def test_backbone_raises_value_error_when_unet_missing(self):
        class NoUnet:
            pass

        adapter = GenericDiffusersAdapter(NoUnet())
        with self.assertRaises(ValueError):
            adapter.backbone()

    def test_backbone_asserts_when_unet_not_module(self):
        class BadUnet:
            def __init__(self) -> None:
                self.unet = "not a module"

        adapter = GenericDiffusersAdapter(BadUnet())
        with self.assertRaises(AssertionError):
            adapter.backbone()
