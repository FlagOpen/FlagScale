#!/usr/bin/env python3
# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
CI platform matrix generator with changed-file filtering.

For schedule / push / workflow_dispatch: returns all configured platforms.
For pull_request: intersects the changed-file list against per-platform
path patterns; falls back to all platforms when core infrastructure files
change.

Usage (called from a GitHub Actions step):
    python .github/scripts/ci/generate_matrix.py \
        --event-name  pull_request \
        --base-ref    main \
        --configs-dir .github/configs \
        >> "$GITHUB_OUTPUT"

Output written to GITHUB_OUTPUT:
    platforms=["cuda","ascend","hygon"]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys

# ---------------------------------------------------------------------------
# Files / directory prefixes whose change triggers all platforms regardless of
# per-platform filtering.  Keep this list minimal — it is the "everything
# breaks" set.
# ---------------------------------------------------------------------------
CORE_TRIGGERS: tuple[str, ...] = (
    ".github/configs/",
    ".github/workflows/",
    ".github/scripts/ci/",
    "setup.py",
    "setup.cfg",
    "pyproject.toml",
    "CMakeLists.txt",
    "requirements",
)

# Config file stems that are metadata, not platform names.
_NON_PLATFORM_STEMS: frozenset[str] = frozenset(("template", "image_sources", "platforms"))


def _load_platforms(configs_dir: pathlib.Path) -> list[str]:
    """Return sorted platform names derived from *.yml files in configs_dir."""
    return sorted(p.stem for p in configs_dir.glob("*.yml") if p.stem not in _NON_PLATFORM_STEMS)


def _changed_files(base_ref: str) -> list[str]:
    """Return files changed between HEAD and origin/<base_ref>."""
    cmd = ["git", "diff", "--name-only", f"origin/{base_ref}...HEAD"]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return [f for f in result.stdout.splitlines() if f.strip()]
    except subprocess.CalledProcessError as exc:
        print(
            f"::warning::git diff failed ({exc}); returning empty list",
            file=sys.stderr,
        )
        return []


def _touches_core(changed: list[str]) -> bool:
    """Return True if any changed file is a core-trigger path."""
    for f in changed:
        for trigger in CORE_TRIGGERS:
            if f.startswith(trigger) or trigger in f:
                return True
    return False


def _platform_is_affected(
    platform: str,
    changed: list[str],
    configs_dir: pathlib.Path,
) -> bool:
    """
    Return True if the platform should run given the changed file list.

    Resolution order:
    1. If the platform config file itself changed → include.
    2. If the config declares ``paths_watch`` → match against changed list.
    3. If no ``paths_watch`` is declared → always include (conservative).
    """
    # Rule 1: config file itself changed.
    if f".github/configs/{platform}.yml" in changed:
        return True

    # Rule 2: config-declared watch paths.
    config_path = configs_dir / f"{platform}.yml"
    if config_path.exists():
        try:
            import yaml  # optional; installed by callers that need it

            with config_path.open() as fh:
                cfg = yaml.safe_load(fh) or {}
            watch = cfg.get("paths_watch", [])
            if watch:
                return any(any(f.startswith(pat) or pat in f for pat in watch) for f in changed)
        except Exception:  # yaml missing or parse error
            pass

    # Rule 3: conservative fallback — no filter defined.
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event-name", required=True, help="GitHub event name")
    parser.add_argument(
        "--base-ref",
        default="main",
        help="Base branch for git diff (PR base)",
    )
    parser.add_argument(
        "--configs-dir",
        default=".github/configs",
        help="Directory containing platform *.yml configs",
    )
    parser.add_argument(
        "--changed-files",
        default="",
        help="Newline-separated list of changed files (overrides git diff)",
    )
    args = parser.parse_args()

    configs_dir = pathlib.Path(args.configs_dir)
    if not configs_dir.is_dir():
        print(
            f"::error::configs directory not found: {configs_dir}",
            file=sys.stderr,
        )
        sys.exit(1)

    all_platforms = _load_platforms(configs_dir)
    if not all_platforms:
        print(
            f"::error::No platform configs found in {configs_dir}",
            file=sys.stderr,
        )
        sys.exit(1)

    # Non-PR events: always run the full matrix.
    if args.event_name != "pull_request":
        print(f"platforms={json.dumps(all_platforms)}")
        return

    # PR: compute reduced matrix from changed files.
    if args.changed_files.strip():
        changed = [f for f in args.changed_files.splitlines() if f.strip()]
    else:
        changed = _changed_files(args.base_ref)

    if not changed:
        # No diff info → safe to run everything.
        print(f"platforms={json.dumps(all_platforms)}")
        return

    if _touches_core(changed):
        print(f"platforms={json.dumps(all_platforms)}")
        return

    selected = [p for p in all_platforms if _platform_is_affected(p, changed, configs_dir)]
    # Guarantee at least one platform runs even if filtering produces empty set.
    print(f"platforms={json.dumps(selected or all_platforms)}")


if __name__ == "__main__":
    main()
