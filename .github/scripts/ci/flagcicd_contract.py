#!/usr/bin/env python3
# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""FlagScale adapter validation; no product or accelerator imports."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import yaml


def fingerprint(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def tracked_tests(root: Path) -> set[str]:
    files = (
        subprocess.check_output(["git", "ls-files", "-z", "tests"], cwd=root).decode().split("\0")
    )
    return {
        name
        for name in files
        if (name.endswith(".py") and Path(name).name.startswith("test_"))
        or name.endswith(".test.cjs")
    }


def load_contract(root: Path) -> tuple[dict, dict, dict]:
    project = yaml.safe_load((root / ".flagcicd/project.yaml").read_text())
    if project.get("schema_version") != "flagcicd.project/v1":
        raise ValueError("Unsupported project schema")
    if (
        set(project)
        != {"schema_version", "repository", "platform_config_glob", "adapter", "profiles"}
        or project["repository"] != "FlagScale"
    ):
        raise ValueError("Unsupported project fields or repository")
    # Adapter locations are fixed for v1; source configuration cannot execute hooks.
    expected = {
        "tiers": ".flagcicd/test-tiers.json",
        "inventory": ".flagcicd/test-inventory.json",
        "test_platform_config_glob": "tests/test_utils/config/platforms/*.yaml",
    }
    if project.get("adapter") != expected:
        raise ValueError("Unsupported adapter paths")
    if project.get("platform_config_glob") != ".github/configs/*.yml":
        raise ValueError("Unsupported platform configuration path")
    if project.get("profiles") != {
        "pr": {"selection": "affected", "rollout": "shadow"},
        "branch": {"selection": "all"},
        "nightly": {"selection": "all"},
    }:
        raise ValueError("Invalid profile policy")
    tiers = json.loads((root / expected["tiers"]).read_text())
    inventory = json.loads((root / expected["inventory"]).read_text())
    if tiers.get("schema_version") != 1 or inventory.get("schema_version") != 1:
        raise ValueError("Unsupported adapter schema")
    if set(tiers["levels"]) != {f"L{i}" for i in range(5)}:
        raise ValueError("Define all L0-L4 capability levels")
    paths = [entry["path"] for entry in inventory["tests"]]
    if len(paths) != len(set(paths)) or set(paths) != tracked_tests(root):
        raise ValueError(
            "Test inventory differs from tracked test files; classify additions/deletions"
        )
    allowed_groups = set(tiers["unit_groups"]) | {"ci-control", "functional-validator"}
    for entry in inventory["tests"]:
        if entry["group"] not in allowed_groups or entry["kind"] not in {
            "automated",
            "helper",
            "placeholder",
        }:
            raise ValueError(f"Invalid inventory entry: {entry}")
        if not (root / entry["path"]).is_file():
            raise ValueError(f"Missing test file: {entry['path']}")
        if entry["group"] in tiers["unit_groups"] and not entry["path"].startswith(
            tiers["unit_groups"][entry["group"]]
        ):
            raise ValueError(f"Unit file outside declared group: {entry['path']}")
    for route in tiers["source_routes"]:
        if (
            not route["paths"]
            or not route["groups"]
            or not set(route["groups"]) <= set(tiers["unit_groups"])
        ):
            raise ValueError("Invalid source route")
    return project, tiers, inventory


if __name__ == "__main__":
    project, tiers, inventory = load_contract(Path.cwd())
    print(f"Validated L0-L4 contract and {len(inventory['tests'])} tracked test entries")
