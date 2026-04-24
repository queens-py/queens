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
"""Check consistency between PEP-style and pixi dependency declarations."""

import argparse
import difflib
import re
import sys
import tomllib
from pathlib import Path
from typing import Any

META_EXTRA_PATTERN = re.compile(r"^queens\[(?P<extra>[A-Za-z0-9_.-]+)\]$")
PACKAGE_NAME_PATTERN = re.compile(r"^\s*([A-Za-z0-9_.-]+)")


def _load_pyproject(path: str) -> dict[str, Any]:
    """Load pyproject.toml."""
    with Path(path).open("rb") as file:
        return tomllib.load(file)


def _pixi_dependency_to_pep(name: str, spec: Any) -> str:
    """Convert a pixi dependency entry to a PEP-style requirement string."""
    if isinstance(spec, str):
        return name if spec in {"", "*"} else f"{name}{spec}"

    if isinstance(spec, dict):
        if "git" in spec:
            git_url = spec["git"]
            if not str(git_url).startswith("git+"):
                git_url = f"git+{git_url}"
            return f"{name} @ {git_url}"
        if "version" in spec:
            version_spec = spec["version"]
            return name if version_spec in {"", "*"} else f"{name}{version_spec}"
        if "path" in spec:
            editable = bool(spec.get("editable", False))
            if editable:
                return f"{name} @ editable:{spec['path']}"
            return f"{name} @ path:{spec['path']}"

    raise ValueError(f"Unsupported pixi dependency format for '{name}': {spec!r}")


def _combine_pixi_dependencies(
    dependencies: dict[str, Any] | None, pypi_dependencies: dict[str, Any] | None
) -> list[tuple[str, Any]]:
    """Combine conda and PyPI pixi dependencies (preserve order)."""
    combined: list[tuple[str, Any]] = []
    if dependencies:
        combined.extend(dependencies.items())
    if pypi_dependencies:
        combined.extend(pypi_dependencies.items())
    return combined


def _diff_lines(expected: list[str], actual: list[str], name: str) -> str:
    """Create a readable unified diff for two lists."""
    return "\n".join(
        difflib.unified_diff(
            expected,
            actual,
            fromfile=f"{name} (PEP)",
            tofile=f"{name} (pixi)",
            lineterm="",
        )
    )


def _extract_requirement_name(requirement: str) -> str:
    """Extract the package name from a requirement string."""
    match = PACKAGE_NAME_PATTERN.match(requirement)
    if match is None:
        raise ValueError(f"Could not extract package name from requirement {requirement!r}")
    return match.group(1)


def _compare_dependency_lists(
    name: str,
    pep_dependencies: list[str],
    pixi_dependencies: list[tuple[str, Any]],
    allowed_version_mismatches: set[str] | None = None,
) -> list[str]:
    """Compare PEP dependency list to pixi dependency list."""
    allowed_version_mismatches = allowed_version_mismatches or set()
    actual = [
        _pixi_dependency_to_pep(dep_name, dep_spec) for dep_name, dep_spec in pixi_dependencies
    ]

    if len(pep_dependencies) != len(actual):
        return [f"Dependency mismatch for '{name}':\n{_diff_lines(pep_dependencies, actual, name)}"]

    for index, (pep_requirement, pixi_requirement) in enumerate(
        zip(pep_dependencies, actual, strict=True)
    ):
        pep_name = _extract_requirement_name(pep_requirement)
        pixi_name = _extract_requirement_name(pixi_requirement)
        if pep_name != pixi_name:
            return [
                f"Dependency order/name mismatch for '{name}' at position {index}: "
                f"{pep_name!r} != {pixi_name!r}\n{_diff_lines(pep_dependencies, actual, name)}"
            ]
        if pep_requirement != pixi_requirement and pep_name not in allowed_version_mismatches:
            return [
                f"Dependency mismatch for '{name}':\n{_diff_lines(pep_dependencies, actual, name)}"
            ]

    return []


def _validate_base_dependencies(
    pyproject: dict[str, Any], allowed_version_mismatches: set[str] | None = None
) -> list[str]:
    """Validate matching base dependencies between PEP and pixi."""
    error_messages: list[str] = []

    project = pyproject.get("project", {})
    tool_pixi = pyproject.get("tool", {}).get("pixi", {})
    pixi_dependencies = tool_pixi.get("dependencies", {})
    pixi_pypi_dependencies = tool_pixi.get("pypi-dependencies", {})

    ordered_conda = list(pixi_dependencies.items())

    if not ordered_conda:
        return ["[tool.pixi.dependencies] is empty; expected at least python and pip."]

    if ordered_conda[0][0] != "python":
        error_messages.append("The first entry in [tool.pixi.dependencies] must be 'python'.")
    else:
        requires_python = project.get("requires-python")
        if ordered_conda[0][1] != requires_python:
            error_messages.append(
                "Mismatch between project.requires-python and tool.pixi.dependencies.python: "
                f"{requires_python!r} != {ordered_conda[0][1]!r}"
            )

    if len(ordered_conda) < 2 or ordered_conda[1][0] != "pip":
        error_messages.append("The second entry in [tool.pixi.dependencies] must be 'pip'.")

    combined = _combine_pixi_dependencies(pixi_dependencies, pixi_pypi_dependencies)
    if not combined:
        return error_messages

    stripped_combined = [
        (name, spec) for name, spec in combined if name not in {"python", "pip", "queens"}
    ]
    pep_dependencies = project.get("dependencies", [])
    error_messages.extend(
        _compare_dependency_lists(
            "project.dependencies",
            pep_dependencies,
            stripped_combined,
            allowed_version_mismatches,
        )
    )

    return error_messages


def _is_meta_optional_dependency(requirements: list[str]) -> bool:
    """Return whether an optional dependency group is a composed meta-extra."""
    return bool(requirements) and all(
        META_EXTRA_PATTERN.fullmatch(requirement) for requirement in requirements
    )


def _validate_meta_optional_dependency(
    name: str,
    requirements: list[str],
    optional_dependencies: dict[str, list[str]],
    pixi_features: dict[str, Any],
) -> list[str]:
    """Validate a composed optional dependency such as 'all'."""
    error_messages: list[str] = []
    referenced_extras = [
        match.group("extra")
        for requirement in requirements
        if (match := META_EXTRA_PATTERN.fullmatch(requirement))
    ]

    for referenced_extra in referenced_extras:
        if referenced_extra not in optional_dependencies:
            error_messages.append(
                f"Meta optional dependency '{name}' references unknown extra '{referenced_extra}'."
            )
        if referenced_extra not in pixi_features:
            error_messages.append(
                f"Meta optional dependency '{name}' references '{referenced_extra}', but no "
                f"[tool.pixi.feature.{referenced_extra}] exists."
            )

    return error_messages


def _validate_feature_groups(
    pyproject: dict[str, Any], allowed_version_mismatches: set[str] | None = None
) -> list[str]:
    """Validate dependency groups and optional dependencies against pixi.

    For each dependency group and each optional-depedencies group a pixi
    feature with identical requirements should exist.
    """
    error_messages: list[str] = []
    pep_names: list[str] = []

    dependency_groups = pyproject.get("dependency-groups", {})
    optional_dependencies = pyproject.get("project", {}).get("optional-dependencies", {})
    pixi_features = pyproject.get("tool", {}).get("pixi", {}).get("feature", {})

    for name, pep_dependencies in dependency_groups.items():
        pep_names.append(name)
        feature = pixi_features.get(name)
        if feature is None:
            error_messages.append(
                f"Missing [tool.pixi.feature.{name}] for dependency group '{name}'."
            )
            continue
        pixi_dependencies_combined = _combine_pixi_dependencies(
            feature.get("dependencies"),
            feature.get("pypi-dependencies"),
        )
        error_messages.extend(
            _compare_dependency_lists(
                f"dependency-groups.{name}",
                pep_dependencies,
                pixi_dependencies_combined,
                allowed_version_mismatches,
            )
        )

    for name, pep_dependencies in optional_dependencies.items():
        if _is_meta_optional_dependency(pep_dependencies):
            error_messages.extend(
                _validate_meta_optional_dependency(
                    name,
                    pep_dependencies,
                    optional_dependencies,
                    pixi_features,
                )
            )
            continue

        pep_names.append(name)
        feature = pixi_features.get(name)
        if feature is None:
            error_messages.append(
                f"Missing [tool.pixi.feature.{name}] for optional dependency group '{name}'."
            )
            continue
        pixi_dependencies_combined = _combine_pixi_dependencies(
            feature.get("dependencies"),
            feature.get("pypi-dependencies"),
        )
        error_messages.extend(
            _compare_dependency_lists(
                f"project.optional-dependencies.{name}",
                pep_dependencies,
                pixi_dependencies_combined,
                allowed_version_mismatches,
            )
        )

    # for each feature there should either be a dependency group or an optional dependency group
    # except for some special features:
    # 1. the base feature is covered by the project.dependencies
    special_exception_features: list[str] = ["base"]
    for feature_name in pixi_features.keys():
        if feature_name in pep_names or feature_name in special_exception_features:
            continue
        error_messages.append(
            f"Missing either a pep dependency group or optional dependency '{feature_name}' "
            f"for the pixi feature [tool.pixi.feature.{feature_name}]."
            ""
        )

    return error_messages


def main() -> int:
    """Validate pyproject.toml integrity."""
    parser = argparse.ArgumentParser(
        description="Check consistency between PEP-style and pixi dependency declarations."
    )
    parser.add_argument("--path", default="pyproject.toml", help="Path to pyproject.toml")
    parser.add_argument(
        "--allow-version-mismatch",
        action="append",
        default=None,
        help=(
            "Package name for which version mismatches are allowed as long as the package name "
            "and position are identical between the PEP and pixi declarations."
        ),
    )
    args = parser.parse_args()

    try:
        pyproject = _load_pyproject(args.path)
    except (OSError, tomllib.TOMLDecodeError) as error:
        print(f"Failed to load {args.path}: {error}", file=sys.stderr)
        return 2

    allowed_version_mismatches = set(args.allow_version_mismatch or [])

    error_messages = []
    error_messages.extend(_validate_base_dependencies(pyproject, allowed_version_mismatches))
    error_messages.extend(_validate_feature_groups(pyproject, allowed_version_mismatches))

    if not error_messages:
        print(
            "Dependency declarations in pyproject.toml are consistent "
            "between PEP and pixi sections."
        )
        return 0

    print("Dependency integrity check failed:\n", file=sys.stderr)
    for error_message in error_messages:
        print(f"- {error_message}\n", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
