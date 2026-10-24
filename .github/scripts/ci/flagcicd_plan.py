#!/usr/bin/env python3
# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Plan existing FlagScale tests without importing the product or changing runners."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from pathlib import Path

import yaml
from flagcicd_contract import fingerprint, load_contract
from generate_matrix import changed_files, documentation, load_platforms, matches, select_platforms

ACTIVE_TASKS = ("train", "hetero_train", "benchmark")
SCHEMA = "flagcicd.plan/v1"


def safe_path(value: str) -> bool:
    return (
        bool(value)
        and not value.startswith(("/", "-"))
        and not any(part in {"", ".", ".."} for part in value.split("/"))
        and not any(character.isspace() for character in value)
        and "\\" not in value
    )


def unit_files(root: Path, inventory: dict, config: dict, groups: list[str] | None) -> list[str]:
    """Intersect selected inventory groups with existing platform support."""
    include = config.get("include", "*")
    include = [include] if isinstance(include, str) else include
    exclude = config.get("exclude", [])
    if not isinstance(include, list) or not include or not isinstance(exclude, list):
        raise ValueError("Invalid platform unit include/exclude")
    result = []
    for entry in inventory["tests"]:
        path = entry["path"]
        if not path.startswith("tests/unit_tests/"):
            continue
        if groups is not None and (entry["group"] not in groups or entry["kind"] != "automated"):
            continue
        if not any(
            pattern == "*"
            or path == pattern
            or matches(path, pattern.rstrip("/") + "/")
            or matches(path, pattern)
            for pattern in include
        ):
            continue
        if any(
            "::" not in pattern and (path == pattern or path.startswith(pattern.rstrip("/") + "/"))
            for pattern in exclude
        ):
            continue
        if not safe_path(path) or not (root / path).is_file():
            raise ValueError(f"Invalid unit test target: {path}")
        result.append(path)
    return sorted(result)


def functional_cases(root: Path, platform: str, config: dict, tasks: tuple[str, ...]) -> list[dict]:
    result = []
    for device in config["device_types"]:
        for task in tasks:
            for model, cases in config[device]["tests"].get("functional", {}).get(task, {}).items():
                for case in cases:
                    for value in (device, task, model, case):
                        if not re.fullmatch(r"[A-Za-z0-9_.-]+", value):
                            raise ValueError("Unsafe functional case identifier")
                    path = f"tests/functional_tests/{task}/{model}/conf/{case}.yaml"
                    case_config = yaml.safe_load((root / path).read_text())
                    runner = case_config.get("experiment", {}).get("runner", {})
                    count = runner.get("nproc_per_node")
                    nodes = runner.get("nnodes")
                    # Heterogeneous launchers and Hydra expressions cannot be reduced
                    # safely to a single-device claim. Preserve their L3 envelope.
                    single = type(count) is int and count == 1 and type(nodes) is int and nodes == 1
                    level = (
                        "L4"
                        if task == "benchmark"
                        else "L2"
                        if single and task == "train"
                        else "L3"
                    )
                    result.append(
                        {
                            "device": device,
                            "task": task,
                            "model": model,
                            "case": case,
                            "level": level,
                            "exclusive": True,
                            "config": path,
                            "platform": platform,
                        }
                    )
    return result


def config_identity(root: Path, platforms: list[str]) -> str:
    paths = [
        root / ".flagcicd" / name
        for name in ("project.yaml", "test-tiers.json", "test-inventory.json")
    ]
    for name in platforms:
        paths.extend(
            [
                root / f".github/configs/{name}.yml",
                root / f"tests/test_utils/config/platforms/{name}.yaml",
            ]
        )
    return fingerprint({str(path.relative_to(root)): path.read_text() for path in sorted(paths)})


def build_plan(
    root: Path,
    *,
    profile: str,
    changed: list[str] | None,
    labels: list[str],
    mode: str = "shadow",
    source_sha: str = "",
    control_sha: str = "",
    base_sha: str = "",
) -> dict:
    if profile not in {"pr", "branch", "nightly"} or mode not in {"shadow", "enforce"}:
        raise ValueError("Invalid profile or rollout mode")
    _, tiers, inventory = load_contract(root)
    source_sha = (
        source_sha
        or subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    )
    control_sha = control_sha or source_sha
    for value in (source_sha, control_sha):
        if not re.fullmatch(r"[0-9a-f]{40}", value):
            raise ValueError("Immutable source/control SHA required")
    if changed is not None and any(not safe_path(path) for path in changed):
        raise ValueError("Invalid changed path")
    platform_configs = load_platforms(root / ".github/configs")
    # Runtime model examples have executable consumers; they are not docs-only.
    routed = (
        [
            "runtime-example-config"
            if path.startswith("examples/") and not path.endswith(".md")
            else path
            for path in changed
        ]
        if changed is not None
        else None
    )
    platforms, platform_reason = select_platforms(
        platform_configs, "pull_request" if profile == "pr" else "schedule", routed, labels
    )
    code = [
        path
        for path in changed or []
        if not documentation(path) or path.startswith("examples/") and not path.endswith(".md")
    ]
    groups = set()
    full = profile != "pr" or not changed or bool(labels)
    reasons = [platform_reason]
    unit_only = bool(code)
    entries = {entry["path"]: entry for entry in inventory["tests"]}
    for path in code:
        entry = entries.get(path)
        if entry and path.startswith("tests/unit_tests/") and entry["kind"] == "automated":
            groups.add(entry["group"])
            continue
        unit_only = False
        routes = [
            route
            for route in tiers["source_routes"]
            if any(matches(path, pattern) for pattern in route["paths"])
        ]
        if routes:
            for route in routes:
                groups.update(route["groups"])
                reasons.append(route["reason"])
        else:
            full = True
            reasons.append(f"Conservative full fallback: {path}")
    if not code:
        full = True  # labels on docs still request real tests
    slices = {}
    for platform in platforms:
        config = yaml.safe_load(
            (root / f"tests/test_utils/config/platforms/{platform}.yaml").read_text()
        )
        devices = config["device_types"]
        if not isinstance(devices, list) or not devices or len(devices) != len(set(devices)):
            raise ValueError(f"Invalid device inventory for {platform}")
        proposal = {}
        platform_full = full
        for device in devices:
            unit = config[device]["tests"]["unit"]
            # Keep the legacy full target for full mode, including its discovery behavior.
            selected = unit_files(root, inventory, unit, None if full else sorted(groups))
            # Excluded/missing groups must not silently turn into a green empty unit job.
            if not selected:
                platform_full = True
                reasons.append(
                    f"No supported affected unit tests on {platform}/{device}: full fallback"
                )
            proposal[device] = selected
        proposed_tasks = ACTIVE_TASKS if platform_full or not unit_only else ()
        all_cases = functional_cases(root, platform, config, ACTIVE_TASKS)
        effective_full = platform_full or mode == "shadow" or profile != "pr"
        slices[platform] = {
            "devices": devices,
            "unit_execution_level": "L3",
            "unit_level_reason": "Unsplit legacy torchrun envelope; individual unit semantics are not inferred from directory names.",
            "unit_mode": "full" if effective_full else "affected",
            "unit_targets": {
                device: [] if effective_full else proposal[device] for device in devices
            },
            "proposed_unit_groups": sorted(tiers["unit_groups"] if platform_full else groups),
            "proposed_tasks": list(proposed_tasks),
            "cases": all_cases
            if effective_full
            else [case for case in all_cases if case["task"] in proposed_tasks],
            "disabled_tasks": tiers["disabled_tasks"],
        }
    plan = {
        "schema_version": SCHEMA,
        "profile": profile,
        "mode": mode,
        "selection": "affected" if profile == "pr" else "all",
        "base_checks": [
            {"job": "lint", "level": "L0", "scope": "static-and-configuration"},
            {"job": "lint", "level": "L1", "scope": "hardware-free-ci-control-tests"},
        ],
        "capability_readiness": {
            level: definition["readiness"] for level, definition in tiers["levels"].items()
        },
        "source_sha": source_sha,
        "control_sha": control_sha,
        "base_sha": base_sha,
        "diff_fingerprint": fingerprint(changed),
        "platforms": platforms,
        "config_fingerprint": config_identity(root, platforms),
        "reasons": sorted(set(reasons)),
        "slices": slices,
    }
    plan["plan_hash"] = fingerprint(plan)
    return plan


def verify_plan(root: Path, plan: dict) -> None:
    payload = {key: value for key, value in plan.items() if key != "plan_hash"}
    if plan.get("schema_version") != SCHEMA or fingerprint(payload) != plan.get("plan_hash"):
        raise ValueError("Invalid plan schema or hash")
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    if actual != plan["source_sha"]:
        raise ValueError("Plan source SHA does not match checkout")
    if config_identity(root, plan["platforms"]) != plan["config_fingerprint"]:
        raise ValueError("Plan configuration does not match checkout")


def outputs(plan: dict, platform: str) -> dict[str, str]:
    selected = plan["slices"][platform]
    result = {
        "unit_selection": json.dumps(
            {
                "plan_hash": plan["plan_hash"],
                "source_sha": plan["source_sha"],
                "mode": selected["unit_mode"],
                "targets": selected["unit_targets"],
            },
            separators=(",", ":"),
        )
    }
    for task in ACTIVE_TASKS:
        result[f"{task}_test_matrix"] = json.dumps(
            [
                {key: case[key] for key in ("device", "task", "model", "case")}
                for case in selected["cases"]
                if case["task"] == task
            ],
            separators=(",", ":"),
        )
    return result


def check_platform(plan: dict, platform: str, needs: dict) -> list[str]:
    selected = plan["slices"][platform]
    expected = {
        "checkout_and_config": "success",
        "cli_validation": "success",
        "unit_tests": "success",
    }
    for task in ACTIVE_TASKS:
        expected[f"functional_tests_{task}"] = (
            "success" if any(case["task"] == task for case in selected["cases"]) else "skipped"
        )
    errors = [
        name for name, result in expected.items() if needs.get(name, {}).get("result") != result
    ]
    for name in ("functional_tests_inference", "functional_tests_serve"):
        if needs.get(name, {}).get("result") != "skipped":
            errors.append(name)
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["plan", "slice", "check"])
    parser.add_argument("--profile", choices=["pr", "branch", "nightly"], default="pr")
    parser.add_argument("--mode", choices=["shadow", "enforce"], default="shadow")
    parser.add_argument("--base-ref", default="origin/main")
    parser.add_argument("--head-ref", default="HEAD")
    parser.add_argument("--control-sha", default="")
    parser.add_argument("--event-path", type=Path)
    parser.add_argument("--file", type=Path, default=Path("plan.json"))
    parser.add_argument("--platform", default="")
    args = parser.parse_args()
    root = Path.cwd()
    if args.command == "plan":
        event = json.loads(args.event_path.read_text()) if args.event_path else {}
        labels = [label["name"] for label in event.get("pull_request", {}).get("labels", [])]
        source = subprocess.check_output(["git", "rev-parse", args.head_ref], text=True).strip()
        plan = build_plan(
            root,
            profile=args.profile,
            mode=args.mode,
            source_sha=source,
            control_sha=args.control_sha,
            base_sha=event.get("pull_request", {}).get("base", {}).get("sha", ""),
            changed=changed_files(args.base_ref, args.head_ref) if args.profile == "pr" else None,
            labels=labels,
        )
        args.file.write_text(json.dumps(plan, indent=2) + "\n")
        result = {
            "platforms": json.dumps(plan["platforms"]),
            "plan": json.dumps(plan, separators=(",", ":")),
            "reason": "; ".join(plan["reasons"]),
        }
        if summary := os.environ.get("GITHUB_STEP_SUMMARY"):
            with open(summary, "a") as stream:
                stream.write(f"Plan `{plan['plan_hash']}`: {plan['profile']} / {plan['mode']}\n\n")
                stream.write("```json\n" + json.dumps(plan, indent=2) + "\n```\n")
    else:
        plan = json.loads(args.file.read_text())
        verify_plan(root, plan)
        if args.command == "check":
            errors = check_platform(plan, args.platform, json.loads(os.environ["JOB_RESULTS"]))
            if errors:
                raise SystemExit(f"Required planned jobs did not finish: {errors}")
            result = {}
        else:
            result = outputs(plan, args.platform)
    for key, value in result.items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()
