#!/usr/bin/env python3
# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Retry only known transient failures; never infer flakiness from a generic timeout."""

import argparse
import json
import pathlib
import re

TRANSIENT = re.compile(
    r"ReadTimeout|ConnectTimeout|ConnectionResetError|Temporary failure in name resolution|"
    r"Could not resolve host|Connection timed out|Connection reset by peer|"
    r"TLS handshake timeout|HTTP[^\n]*(?:502 Bad Gateway|503 Service Unavailable)",
    re.IGNORECASE,
)
DETERMINISTIC = re.compile(
    r"AssertionError|SyntaxError|IndentationError|FAILED .*test_|"
    r"ModuleNotFoundError|No matching distribution found|CompilationError|"
    r"error: command .* failed|CUDA out of memory|device-side assert",
    re.IGNORECASE,
)


def retryable(jobs: list[dict], logs: dict[int, str]) -> bool:
    failed = [job for job in jobs if job["conclusion"] == "failure"]
    if not failed or any(
        job["conclusion"] in ("cancelled", "timed_out", "action_required") for job in jobs
    ):
        return False
    for job in failed:
        name = job["name"].lower()
        if name in ("ci complete", "all_tests") or name.endswith("/ all_tests_complete"):
            continue
        if name in ("lint", "generate platform matrix") or "checkout_and_config" in name:
            return False
        log = logs.get(job["id"], "")
        if not TRANSIENT.search(log) or DETERMINISTIC.search(log):
            return False
    # An aggregate failure alone is not evidence of a transient problem.
    return any(TRANSIENT.search(logs.get(job["id"], "")) for job in failed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=pathlib.Path, required=True)
    parser.add_argument("--logs-dir", type=pathlib.Path, required=True)
    args = parser.parse_args()
    jobs = json.loads(args.jobs.read_text())
    logs = {
        job["id"]: path.read_text(errors="replace")
        for job in jobs
        if (path := args.logs_dir / f"{job['id']}.log").exists()
    }
    print(f"retry={'true' if retryable(jobs, logs) else 'false'}")
