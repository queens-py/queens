#!/usr/bin/env python3
#
# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (c) 2024-2025, QUEENS contributors.
#
# This file is part of QUEENS.
#
# QUEENS is free software: you can redistribute it and/or modify it under the terms of the GNU
# Lesser General Public License as published by the Free Software Foundation, either version 3 of
# the License, or (at your option) any later version. QUEENS is distributed in the hope that it will
# be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details. You
# should have received a copy of the GNU Lesser General Public License along with QUEENS. If not,
# see <https://www.gnu.org/licenses/>.
#
"""Diff dependency-related pyproject.toml declarations between two git refs."""

import argparse
import difflib
import json
import subprocess
import sys
import tomllib
from typing import Any


def _load_pyproject_sections(ref: str, path: str) -> dict[str, Any]:
    """Load relevant dependency declarations from a git ref."""
    content = subprocess.check_output(["git", "show", f"{ref}:{path}"], text=False)
    parsed = tomllib.loads(content.decode("utf-8"))
    return {
        "project.dependencies": parsed.get("project", {}).get("dependencies", []),
        "dependency-groups": parsed.get("dependency-groups", {}),
        "project.optional-dependencies": parsed.get("project", {}).get("optional-dependencies", {}),
        "tool.pixi.workspace": parsed.get("tool", {}).get("pixi", {}).get("workspace", {}),
        "tool.pixi.dependencies": parsed.get("tool", {}).get("pixi", {}).get("dependencies", {}),
        "tool.pixi.pypi-dependencies": parsed.get("tool", {})
        .get("pixi", {})
        .get("pypi-dependencies", {}),
        "tool.pixi.feature": parsed.get("tool", {}).get("pixi", {}).get("feature", {}),
    }


def main() -> int:
    """Run the diff and return a status code."""
    parser = argparse.ArgumentParser(
        description="Compare dependency-related pyproject.toml declarations between two refs."
    )
    parser.add_argument("--base-ref", required=True, help="Base git ref")
    parser.add_argument("--head-ref", default="HEAD", help="Head git ref")
    parser.add_argument("--path", default="pyproject.toml", help="Path to pyproject.toml")
    args = parser.parse_args()

    try:
        base_content = _load_pyproject_sections(args.base_ref, args.path)
        head_content = _load_pyproject_sections(args.head_ref, args.path)
    except subprocess.CalledProcessError as error:
        print(error.stderr.decode("utf-8") if error.stderr else str(error), file=sys.stderr)
        return 2

    base_json = json.dumps(base_content, indent=2, sort_keys=True).splitlines()
    head_json = json.dumps(head_content, indent=2, sort_keys=True).splitlines()
    diff = list(
        difflib.unified_diff(
            base_json,
            head_json,
            fromfile=f"{args.base_ref}:{args.path}",
            tofile=f"{args.head_ref}:{args.path}",
            lineterm="",
        )
    )

    if not diff:
        return 0

    print("\n".join(diff))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
