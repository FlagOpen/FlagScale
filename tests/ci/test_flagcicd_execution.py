# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Exercise real pytest evidence and planned-target validation without devices."""

import copy
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / ".github/scripts/ci"
sys.path.insert(0, str(SCRIPTS))
import flagcicd_plan as planner
import flagcicd_unit as unit


def selection():
    plan = planner.build_plan(
        ROOT,
        profile="pr",
        mode="enforce",
        labels=[],
        changed=["tests/unit_tests/checkpoint/test_qwen35_converter.py"],
    )
    return {
        "plan_hash": plan["plan_hash"],
        "source_sha": plan["source_sha"],
        "mode": "affected",
        "targets": plan["slices"]["ascend"]["unit_targets"],
    }


def test_unit_target_identity_and_full_compatibility():
    data = selection()
    assert unit.targets(ROOT, data, "unit-910c") == [
        "tests/unit_tests/checkpoint/test_qwen35_converter.py"
    ]
    data.update(mode="full", targets={"unit-910c": []})
    assert unit.targets(ROOT, data, "unit-910c") == []


@pytest.mark.parametrize("mutation", ["empty", "stale", "escape", "option", "missing", "duplicate"])
def test_invalid_selection_fails(mutation):
    data = selection()
    path = data["targets"]["unit-910c"][0]
    if mutation == "stale":
        data["source_sha"] = "0" * 40
    else:
        data["targets"]["unit-910c"] = {
            "empty": [],
            "escape": ["tests/unit_tests/../../pyproject.toml.py"],
            "option": ["-k expression"],
            "missing": ["tests/unit_tests/absent.py"],
            "duplicate": [path, path],
        }[mutation]
    with pytest.raises(ValueError):
        unit.targets(ROOT, data, "unit-910c")


@pytest.mark.parametrize(
    "body,passed",
    [
        ("def test_ok(): assert True", True),
        ("import pytest\ndef test_skip(): pytest.skip('no device')", False),
        ("def test_bad(): assert False", False),
        ("# no tests", False),
        ("raise RuntimeError('collection failed')", False),
    ],
)
def test_real_pytest_plugin_never_passes_empty_or_all_skipped_runs(tmp_path, body, passed):
    (tmp_path / "test_sample.py").write_text(body)
    report = tmp_path / "reports"
    env = {
        **os.environ,
        "PYTHONPATH": str(SCRIPTS),
        "FLAGSCALE_CI_JUNIT_DIR": str(report),
        "LOCAL_RANK": "3",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
    }
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-p", "flagcicd_pytest", "-p", "no:cacheprovider", "-q"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert (result.returncode == 0) == passed, result.stdout + result.stderr
    tree = ET.parse(report / "junit-rank-3.xml")
    assert tree.getroot().tag == "testsuites"


def test_matrix_slices_and_gate_reject_missing_planned_train():
    plan = planner.build_plan(ROOT, profile="nightly", labels=[], changed=None)
    needs = {
        name: {"result": "success"}
        for name in (
            "checkout_and_config",
            "cli_validation",
            "unit_tests",
            "functional_tests_train",
            "functional_tests_benchmark",
        )
    }
    needs.update(
        {
            name: {"result": "skipped"}
            for name in (
                "functional_tests_hetero_train",
                "functional_tests_inference",
                "functional_tests_serve",
            )
        }
    )
    assert not planner.check_platform(plan, "ascend", needs)
    for state in ("skipped", "failure", "cancelled"):
        broken = copy.deepcopy(needs)
        broken["functional_tests_train"]["result"] = state
        assert "functional_tests_train" in planner.check_platform(plan, "ascend", broken)


def test_optional_workflow_inputs_preserve_legacy_callers_and_secrets():
    def workflow(name):
        return yaml.load(
            (ROOT / f".github/workflows/{name}.yml").read_text(), Loader=yaml.BaseLoader
        )

    pipeline = workflow("_platform_pipeline")
    common = workflow("all_tests_common")
    unit_workflow = workflow("unit_tests_common")
    assert pipeline["on"]["workflow_call"]["inputs"]["test_plan"]["default"] == ""
    assert common["on"]["workflow_call"]["inputs"]["test_plan"]["default"] == ""
    assert unit_workflow["on"]["workflow_call"]["inputs"]["unit_selection"]["default"] == ""
    assert "test_plan" not in pipeline["jobs"]["prepare"]["with"]
    assert pipeline["jobs"]["test"]["with"]["test_plan"] == "${{ inputs.test_plan }}"
    assert common["jobs"]["functional_tests_inference"]["if"].startswith("false")
    assert common["jobs"]["functional_tests_serve"]["if"].startswith("false")
    assert "functional_tests_rl" not in common["jobs"]
