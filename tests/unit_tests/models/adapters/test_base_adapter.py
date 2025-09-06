import unittest

from unittest.mock import patch

from flagscale.models.adapters.base_adapter import BaseAdapter


class _MyAdapter(BaseAdapter):
    def backbone(self):
        return object()


class TestBaseAdapter(unittest.TestCase):
    def test_register_and_get_capability(self):
        adapter = _MyAdapter(model_or_pipeline=object())
        self.assertFalse(adapter.has_capability("inc"))

        def inc(x):
            return x + 1

        adapter.register_capability("inc", inc)
        self.assertTrue(adapter.has_capability("inc"))
        cap = adapter.get_capability("inc")
        self.assertIs(cap, inc)
        self.assertEqual(cap(9), 10)
        self.assertTrue(adapter.save())

    def test_register_capability_duplicate_warns_and_no_override(self):
        with patch("flagscale.models.adapters.base_adapter.logger") as mock_logger:
            adapter = _MyAdapter(model_or_pipeline=object())

            def f1():
                return "first"

            def f2():
                return "second"

            adapter.register_capability("cap", f1)
            adapter.register_capability("cap", f2)

            self.assertTrue(adapter.has_capability("cap"))
            self.assertIs(adapter.get_capability("cap"), f1)
            mock_logger.warning.assert_called_once_with("Capability cap is already registered.")
