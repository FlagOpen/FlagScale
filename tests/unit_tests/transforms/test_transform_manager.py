import unittest

import torch
import torch.nn as nn

from flagscale.transforms.transform import Transform, TransformSpec
from flagscale.transforms.transform_manager import TransformManager


class TestModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class TestModel2(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 10)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.linear(x))


class TestTransform1(Transform):
    """To test the pre_compile phase"""

    def spec(self):
        return TransformSpec(name="TestTransform1", phase="pre_compile", priority=10)

    def apply(self, model: nn.Module) -> bool:
        return True


class TestTransform2(Transform):
    """To test the `requires` constraint and `supports` constraint"""

    def spec(self):
        return TransformSpec(
            name="TestTransform2", phase="pre_compile", priority=0, requires=["TestTransform1"]
        )

    def supports(self, model: nn.Module) -> bool:
        if not isinstance(model, TestModel):
            return False
        else:
            return True

    def apply(self, model: nn.Module) -> bool:
        return True


class TestTransform3(Transform):
    """To test the `compile` phase and `preflight` constraint"""

    def spec(self):
        return TransformSpec(name="TestTransform3", phase="compile", priority=0)

    def preflight(self) -> bool:
        return True

    def apply(self, model: nn.Module) -> bool:
        return True


class TestTransform4(Transform):
    """To test the `post_compile` phase and `forbids` constraint"""

    def spec(self):
        return TransformSpec(
            name="TestTransform4", phase="post_compile", priority=0, forbids=["TestTransform2"]
        )

    def apply(self, model: nn.Module) -> bool:
        return True


class TestTransform5(Transform):
    """To test the `post_compile` phase and `after` constraint"""

    def spec(self):
        return TransformSpec(
            name="TestTransform5", phase="post_compile", priority=0, after=["TestTransform4"]
        )

    def apply(self, model: nn.Module) -> bool:
        return True


class TestTransform6(Transform):
    """To test the `post_compile` phase and `before` constraint"""

    def spec(self):
        return TransformSpec(
            name="TestTransform6",
            phase="post_compile",
            priority=0,
            before=["TestTransform5"],
            after=["TestTransform4"],
        )

    def apply(self, model: nn.Module) -> bool:
        return True


class TestTransform7(Transform):
    """To test the `post_compile` phase and `before` constraint and `priority` constraint"""

    def spec(self):
        return TransformSpec(
            name="TestTransform7", phase="post_compile", priority=100, before=["TestTransform6"]
        )

    def apply(self, model: nn.Module) -> bool:
        return True


class TestTransform8(Transform):
    """To test the wrong phase name"""

    def spec(self):
        return TransformSpec(
            name="TestTransform8", phase="adolescence", priority=0, after=["TestTransform7"]
        )

    def apply(self, model: nn.Module) -> bool:
        return True


class TestTransform9(Transform):
    """To test the `preflight` constraint"""

    def spec(self):
        return TransformSpec(name="TestTransform9", phase="post_compile", priority=0)

    def preflight(self) -> bool:
        return False

    def apply(self, model: nn.Module) -> bool:
        return True


class TransformManagerTests(unittest.TestCase):
    def setUp(self):
        self.transforms1 = TestTransform1()
        self.transforms2 = TestTransform2()
        self.transforms3 = TestTransform3()
        self.transforms4 = TestTransform4()
        self.transforms5 = TestTransform5()
        self.transforms6 = TestTransform6()
        self.transforms7 = TestTransform7()
        self.transforms8 = TestTransform8()
        self.transforms9 = TestTransform9()

        self.model1 = TestModel()
        self.model2 = TestModel2()

    def test_phase(self):
        manager = TransformManager([self.transforms1, self.transforms3, self.transforms4])
        expected_result = "DryRun plan(\nPre-compile transforms:\n  TestTransform1\nCompile transforms:\n  TestTransform3\nPost-compile transforms:\n  TestTransform4\n)"
        self.assertEqual(manager.apply(self.model1, dry_run=True), expected_result)

    def test_supports(self):
        manager = TransformManager([self.transforms2])
        with self.assertRaises(ValueError) as cm:
            manager.apply(self.model2, dry_run=True)
        self.assertIn("Transform TestTransform2 not supported for this model.", str(cm.exception))

    def test_priority(self):
        manager = TransformManager(
            [self.transforms1, self.transforms2, self.transforms6, self.transforms7]
        )
        expected_result = "DryRun plan(\nPre-compile transforms:\n  TestTransform1\n  TestTransform2\nCompile transforms:\nPost-compile transforms:\n  TestTransform7\n  TestTransform6\n)"
        self.assertEqual(manager.apply(self.model1, dry_run=True), expected_result)

    def test_preflight(self):
        manager = TransformManager([self.transforms9])
        with self.assertRaises(ValueError) as cm:
            manager.apply(self.model1, dry_run=True)
        self.assertIn("Transform TestTransform9 not supported.", str(cm.exception))

    def test_before_after(self):
        manager = TransformManager(
            [self.transforms4, self.transforms5, self.transforms6, self.transforms7]
        )
        expected_result = "DryRun plan(\nPre-compile transforms:\nCompile transforms:\nPost-compile transforms:\n  TestTransform7\n  TestTransform4\n  TestTransform6\n  TestTransform5\n)"
        self.assertEqual(manager.apply(self.model2, dry_run=True), expected_result)

    def test_forbids(self):
        manager = TransformManager([self.transforms1, self.transforms4, self.transforms2])
        with self.assertRaises(ValueError) as cm:
            manager.apply(self.model1, dry_run=True)
        self.assertIn(
            "Transform TestTransform4 and Transform TestTransform2 cannot be applied together",
            str(cm.exception),
        )

    def test_requires(self):
        manager = TransformManager([self.transforms2, self.transforms3])
        with self.assertRaises(ValueError) as cm:
            manager.apply(self.model1, dry_run=True)
        self.assertIn(
            "Transform TestTransform2 requires missing transforms: ['TestTransform1']",
            str(cm.exception),
        )

    def test_wrong_phase(self):
        manager = TransformManager([self.transforms8])
        with self.assertRaises(ValueError) as cm:
            manager.apply(self.model1, dry_run=True)
        self.assertIn("Unknown phase: adolescence", str(cm.exception))
