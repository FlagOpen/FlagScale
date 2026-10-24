# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Routing safety, full-regression equivalence and planned-job completion."""

import copy
import json
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / ".github/scripts/ci"))
import flagcicd_plan as planner


def plan(changed=None, profile="pr", mode="enforce", labels=None):
    return planner.build_plan(
        ROOT, profile=profile, changed=changed, mode=mode, labels=labels or []
    )


@pytest.mark.parametrize("profile", ["branch", "nightly"])
def test_full_plan_matches_every_enabled_configured_case(profile):
    result = plan(profile=profile)
    assert len(result["platforms"]) == 8
    for platform, selected in result["slices"].items():
        config = yaml.safe_load(
            (ROOT / f"tests/test_utils/config/platforms/{platform}.yaml").read_text()
        )
        expected = {
            (device, task, model, case)
            for device in config["device_types"]
            for task in planner.ACTIVE_TASKS
            for model, cases in config[device]["tests"]["functional"].get(task, {}).items()
            for case in cases
        }
        actual = {
            tuple(case[key] for key in ("device", "task", "model", "case"))
            for case in selected["cases"]
        }
        assert actual == expected
        assert selected["unit_mode"] == "full"
        assert selected["disabled_tasks"].keys() == {"inference", "serve", "rl"}


def test_training_level_follows_actual_topology_not_ten_iteration_name():
    result = plan(profile="nightly")
    assert result["capability_readiness"]["L4"] == "partial"
    assert result["slices"]["ascend"]["unit_execution_level"] == "L3"
    ascend = next(case for case in result["slices"]["ascend"]["cases"] if case["task"] == "train")
    hygon = next(case for case in result["slices"]["hygon"]["cases"] if case["task"] == "train")
    assert ascend["level"] == "L3"
    assert hygon["level"] == "L2"
    assert all(
        case["level"] == "L4"
        for selected in result["slices"].values()
        for case in selected["cases"]
        if case["task"] == "benchmark"
    )


def test_unit_change_selects_group_on_all_platforms_without_functional_cases():
    result = plan(["tests/unit_tests/checkpoint/test_qwen35_converter.py"])
    assert len(result["platforms"]) == 8
    for selected in result["slices"].values():
        assert selected["unit_mode"] == "affected"
        assert selected["proposed_unit_groups"] == ["unit-checkpoint"]
        assert not selected["cases"]
        assert all(
            targets == ["tests/unit_tests/checkpoint/test_qwen35_converter.py"]
            for targets in selected["unit_targets"].values()
        )


def test_shadow_proposes_subset_but_executes_full():
    result = plan(["tests/unit_tests/checkpoint/test_qwen35_converter.py"], mode="shadow")
    for selected in result["slices"].values():
        assert selected["proposed_unit_groups"] == ["unit-checkpoint"]
        assert selected["unit_mode"] == "full"
        assert selected["cases"]


@pytest.mark.parametrize(
    "changed",
    [
        None,
        [],
        ["flagscale/runner/runner_train.py"],
        [".github/workflows/ci.yml"],
        ["requirements/common.txt"],
        ["new-unknown-file"],
        ["tests/conftest.py"],
        ["tests/unit_tests/test_utilities.py"],
    ],
)
def test_unknown_shared_or_missing_diff_keeps_full_tests(changed):
    result = plan(changed)
    assert all(selected["unit_mode"] == "full" for selected in result["slices"].values())
    assert len(result["platforms"]) == 8


def test_platform_labels_remain_additive_and_request_full_tests():
    result = plan(["docker/cuda/Dockerfile.train"], labels=["platform:ascend"])
    assert result["platforms"] == ["ascend", "cuda"]
    assert all(selected["unit_mode"] == "full" for selected in result["slices"].values())


def test_runtime_examples_override_old_documentation_skip():
    result = plan(["examples/qwen35/conf/train/4b.yaml"])
    assert len(result["platforms"]) == 8
    assert all(
        selected["proposed_unit_groups"] == ["unit-checkpoint"] and selected["cases"]
        for selected in result["slices"].values()
    )
    assert plan(["README.md"])["platforms"] == []


def test_mixed_unknown_change_cannot_narrow_groups():
    result = plan(["tools/checkpoint/qwen35/qwen35/io.py", "flagscale/train/train.py"])
    assert all(selected["unit_mode"] == "full" for selected in result["slices"].values())


def test_existing_exclusions_intersect_affected_targets():
    result = plan(["tests/unit_tests/models/kerv/test_configs.py"])
    targets = result["slices"]["ascend"]["unit_targets"]["unit-910c"]
    assert "tests/unit_tests/models/kerv/test_configs.py" in targets
    assert "tests/unit_tests/models/kerv/test_ops.py" not in targets


def test_plan_identity_and_wrong_source_rejected():
    result = plan(profile="nightly")
    planner.verify_plan(ROOT, result)
    other = copy.deepcopy(result)
    other["source_sha"] = "a" * 40
    with pytest.raises(ValueError, match="hash"):
        planner.verify_plan(ROOT, other)
    other["plan_hash"] = planner.fingerprint(
        {key: value for key, value in other.items() if key != "plan_hash"}
    )
    with pytest.raises(ValueError, match="source SHA"):
        planner.verify_plan(ROOT, other)


def test_gate_rejects_missing_failed_cancelled_and_unexpected_skipped_jobs():
    result = plan(profile="nightly")
    selected = result["slices"]["ascend"]
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
    assert selected["cases"] and not planner.check_platform(result, "ascend", needs)
    for state in ("failure", "cancelled", "skipped", None):
        broken = copy.deepcopy(needs)
        broken["unit_tests"] = {"result": state}
        assert "unit_tests" in planner.check_platform(result, "ascend", broken)


def test_affected_empty_functional_matrices_are_explicit():
    result = plan(["tests/unit_tests/checkpoint/test_qwen35_converter.py"])
    outputs = planner.outputs(result, "ascend")
    assert json.loads(outputs["train_test_matrix"]) == []
    assert json.loads(outputs["unit_selection"])["mode"] == "affected"


@pytest.mark.parametrize("path", ["../escape.py", "/tmp/test.py", "-x", "a/../x", "a b.py"])
def test_unsafe_paths_fail_before_command_construction(path):
    with pytest.raises(ValueError, match="path"):
        plan([path])
