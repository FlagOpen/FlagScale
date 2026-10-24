# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Inventory completeness and honest capability declarations."""

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "contract", ROOT / ".github/scripts/ci/flagcicd_contract.py"
)
contract = importlib.util.module_from_spec(spec)
spec.loader.exec_module(contract)


def test_repository_inventory_and_capability_gaps():
    _, tiers, inventory = contract.load_contract(ROOT)
    assert tiers["levels"]["L2"]["gaps"]
    assert tiers["levels"]["L4"]["readiness"] == "partial"
    entries = {entry["path"]: entry for entry in inventory["tests"]}
    assert entries["tests/unit_tests/test_basic.py"]["kind"] == "placeholder"
    assert entries["tests/unit_tests/test_utilities.py"]["kind"] == "helper"


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "wrong-group"])
def test_invalid_inventory_fails(tmp_path, monkeypatch, mutation):
    directory = tmp_path / ".flagcicd"
    directory.mkdir()
    for path in (ROOT / ".flagcicd").glob("*"):
        (directory / path.name).write_bytes(path.read_bytes())
    inventory = json.loads((directory / "test-inventory.json").read_text())
    tracked = contract.tracked_tests(ROOT)
    for name in tracked:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    if mutation == "missing":
        inventory["tests"].pop()
    elif mutation == "duplicate":
        inventory["tests"].append(inventory["tests"][0])
    else:
        inventory["tests"][0]["group"] = "does-not-exist"
    (directory / "test-inventory.json").write_text(json.dumps(inventory))
    monkeypatch.setattr(contract, "tracked_tests", lambda _: tracked)
    with pytest.raises(ValueError):
        contract.load_contract(tmp_path)
