# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""CI controller tests deliberately require no FlagScale/torch imports."""

import importlib.util
import json
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / ".github/scripts/ci"


def load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


matrix = load("generate_matrix")
gate = load("check_results")
retry = load("retry_policy")
PLATFORMS = matrix.load_platforms(ROOT / ".github/configs")


@pytest.mark.parametrize("platform", sorted(PLATFORMS))
@pytest.mark.parametrize(
    "path",
    [
        "docker/{0}/Dockerfile.train",
        "requirements/{0}/train.txt",
        "tools/install/{0}/install_train.sh",
        ".github/configs/{0}.yml",
        ".github/scripts/set_env_{0}.sh",
        "tests/test_utils/config/platforms/{0}.yaml",
    ],
)
def test_platform_owned_change(platform, path):
    assert matrix.select_platforms(PLATFORMS, "pull_request", [path.format(platform)], [])[0] == [
        platform
    ]


@pytest.mark.parametrize(
    "files",
    [
        ["flagscale/train/foo.py"],
        ["tests/test_utils/runners/run_unit_tests.sh"],
        ["requirements/common.txt"],
        [".github/workflows/ci.yml"],
        ["tools/install/utils/pkg_utils.sh"],
        ["docker/ascend/Dockerfile.train", "flagscale/runner/foo.py"],
        ["new-unclassified-file"],
    ],
)
def test_shared_changes_cannot_be_narrowed_by_labels(files):
    assert matrix.select_platforms(PLATFORMS, "pull_request", files, ["platform:ascend"])[
        0
    ] == sorted(PLATFORMS)


def test_platform_union():
    assert matrix.select_platforms(
        PLATFORMS, "pull_request", ["docker/ascend/a", "docker/cuda/b"], []
    )[0] == ["ascend", "cuda"]


def test_labels_and_documentation():
    files = ["docker/ascend/a"]
    assert matrix.select_platforms(
        PLATFORMS, "pull_request", files, ["platform:cuda", "platform:ppu"]
    )[0] == ["cuda", "ppu"]
    assert matrix.select_platforms(PLATFORMS, "pull_request", files, ["platform:all"])[0] == sorted(
        PLATFORMS
    )
    assert (
        matrix.select_platforms(PLATFORMS, "pull_request", ["docs/setup.txt", "README.md"], [])[0]
        == []
    )
    assert matrix.select_platforms(PLATFORMS, "pull_request", ["README.md"], ["platform:ascend"])[
        0
    ] == ["ascend"]
    with pytest.raises(ValueError, match="Unknown platform"):
        matrix.select_platforms(PLATFORMS, "pull_request", files, ["platform:typo"])


@pytest.mark.parametrize("event", ["push", "schedule", "workflow_dispatch"])
def test_non_pr_is_full(event):
    assert matrix.select_platforms(PLATFORMS, event, ["README.md"], ["platform:ascend"])[
        0
    ] == sorted(PLATFORMS)


@pytest.mark.parametrize("changed", [None, []])
def test_missing_diff_is_conservative(changed):
    assert matrix.select_platforms(PLATFORMS, "pull_request", changed, [])[0] == sorted(PLATFORMS)


def test_missing_watch_and_invalid_configs(tmp_path):
    assert matrix.select_platforms({"a": [], "b": ["b/"]}, "pull_request", ["b/file"], [])[0] == [
        "a",
        "b",
    ]
    (tmp_path / "a.yml").write_text("paths_watch: string")
    with pytest.raises(ValueError):
        matrix.load_platforms(tmp_path)
    (tmp_path / "a.yml").write_text("paths_watch: [")
    with pytest.raises(yaml.YAMLError):
        matrix.load_platforms(tmp_path)
    with pytest.raises(ValueError):
        matrix.load_platforms(tmp_path / "missing")


def test_path_boundaries_and_globs():
    assert matrix.matches("docker/ascend/file", "docker/ascend/")
    assert not matrix.matches("mydocker/ascend/file", "docker/ascend/")
    assert matrix.matches("a/b.py", "a/*.py")
    assert not matrix.matches(".github/configs/cuda.yml.backup", ".github/configs/cuda.yml")


@pytest.mark.parametrize("name", ["lint", "generate-matrix", "platform-pipeline"])
@pytest.mark.parametrize("result", ["failure", "cancelled", "skipped"])
def test_gate_cannot_hide_upstream_failure(name, result):
    needs = {key: {"result": "success"} for key in ["lint", "generate-matrix", "platform-pipeline"]}
    needs[name]["result"] = result
    assert name in gate.failures(needs, ["ascend"])


def test_documentation_gate():
    needs = {key: {"result": "success"} for key in ["lint", "generate-matrix", "platform-pipeline"]}
    needs["platform-pipeline"]["result"] = "skipped"
    assert not gate.failures(needs, [])
    assert gate.failures(needs, ["ascend"])


@pytest.mark.parametrize(
    "log,expected",
    [
        ("pip ReadTimeout while downloading wheel", True),
        ("git: Could not resolve host github.com", True),
        ("Timeout waiting for distributed workers", False),
        ("AssertionError: bad value; ConnectionResetError", False),
        ("FAILED tests/test_foo.py::test_bad ReadTimeout", False),
        ("No matching distribution found ReadTimeout", False),
        ("CUDA out of memory", False),
        ("", False),
    ],
)
def test_retry_is_conservative(log, expected):
    jobs = [
        {"id": 1, "name": "Platform pipeline (ascend) / test / unit", "conclusion": "failure"},
        {"id": 2, "name": "CI Complete", "conclusion": "failure"},
    ]
    assert retry.retryable(jobs, {1: log}) is expected


def test_no_retry_for_lint_mixed_failures_and_missing_logs():
    jobs = [{"id": 1, "name": "Lint", "conclusion": "failure"}]
    assert not retry.retryable(jobs, {1: "ReadTimeout"})
    jobs[0]["name"] = "unit"
    jobs.append({"id": 2, "name": "train", "conclusion": "failure"})
    assert not retry.retryable(jobs, {1: "ReadTimeout", 2: "AssertionError"})
    assert not retry.retryable(jobs, {})
    assert not retry.retryable([{"id": 2, "name": "CI Complete", "conclusion": "failure"}], {})


def test_diff_includes_both_sides_of_renames(tmp_path):
    def git(*args):
        return subprocess.check_output(["git", "-C", str(tmp_path), *args], text=True).strip()

    git("init", "-q")
    git("config", "user.email", "ci@example.invalid")
    git("config", "user.name", "CI test")
    (tmp_path / "shared.py").write_text("content")
    git("add", ".")
    git("commit", "-qm", "base")
    base = git("rev-parse", "HEAD")
    (tmp_path / "shared.py").rename(tmp_path / "ascend.py")
    git("add", ".")
    git("commit", "-qm", "rename")
    import os

    previous = Path.cwd()
    try:
        os.chdir(tmp_path)
        assert sorted(matrix.changed_files(base, "HEAD")) == ["ascend.py", "shared.py"]
        assert matrix.changed_files("missing-ref", "HEAD") is None
    finally:
        os.chdir(previous)


def test_workflow_source_identity_propagates_to_every_child():
    # BaseLoader preserves the YAML 'on' key rather than treating it as boolean.
    directory = ROOT / ".github/workflows"
    pending = ["_platform_pipeline.yml"]
    seen = set()
    while pending:
        name = pending.pop()
        if name in seen:
            continue
        seen.add(name)
        workflow = yaml.load((directory / name).read_text(), Loader=yaml.BaseLoader)
        assert {"source_repository", "source_sha"} <= workflow["on"]["workflow_call"][
            "inputs"
        ].keys()
        for job in workflow["jobs"].values():
            if job.get("uses", "").startswith("./.github/workflows/"):
                assert job["with"]["source_sha"] == "${{ inputs.source_sha }}"
                assert job["with"]["source_repository"] == "${{ inputs.source_repository }}"
                pending.append(Path(job["uses"]).name)
            for step in job.get("steps", []):
                if step.get("uses") == "actions/checkout@v4":
                    checkout = step.get("with", {})
                    # Other repositories (Megatron/TE) keep their own dependency refs.
                    if "needs.resolve.outputs." not in checkout.get("repository", ""):
                        assert "inputs.source_sha" in checkout.get("ref", ""), (name, step["name"])
                        assert "inputs.source_repository" in checkout.get("repository", "")
    assert "functional_tests_cli.yml" in seen
    assert "prepare_dependencies.yml" in seen


def test_matrix_cli_event_payload(tmp_path):
    event = tmp_path / "event.json"
    event.write_text(json.dumps({"pull_request": {"labels": [{"name": "platform:ppu"}]}}))
    import sys

    result = subprocess.check_output(
        [
            sys.executable,
            str(SCRIPTS / "generate_matrix.py"),
            "--event-name",
            "pull_request",
            "--event-path",
            str(event),
            "--changed-files",
            "docker/ascend/a",
        ],
        cwd=ROOT,
        text=True,
    )
    assert 'platforms=["ppu"]' in result
