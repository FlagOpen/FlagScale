# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Evidence for planned unit runs; one JUnit file per torchrun rank."""

import os
from pathlib import Path

import pytest

_passed = 0


@pytest.hookimpl(tryfirst=True)
def pytest_configure(config):
    directory = Path(os.environ["FLAGSCALE_CI_JUNIT_DIR"])
    directory.mkdir(parents=True, exist_ok=True)
    rank = os.environ.get("LOCAL_RANK", "0")
    if not rank.isdecimal():
        raise pytest.UsageError("Invalid LOCAL_RANK for unit report")
    config.option.xmlpath = str(directory / f"junit-rank-{rank}.xml")


def pytest_runtest_logreport(report):
    global _passed
    if report.when == "call" and report.passed:
        _passed += 1


def pytest_sessionfinish(session, exitstatus):
    if exitstatus == 0 and _passed == 0:
        session.exitstatus = pytest.ExitCode.TESTS_FAILED
