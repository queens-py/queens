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
"""Utility methods used by the tutorial tests."""

from collections import Counter
from itertools import chain
from pathlib import Path

import pytest

from queens.utils.path import relative_path_from_root


def inject_notebook_execution_context(tb, notebook_dir):
    """Inject the notebook directory as Python path and working directory.

    Args:
        tb (testbook): testbook object for inserting code into the notebook
        notebook_dir (str | Path): Directory containing the notebook

    Returns:
        None
    """
    tb.inject(
        f"""
        import os
        import sys
        from pathlib import Path
        notebook_dir = Path({str(notebook_dir)!r})
        repo_root = next(
            path for path in (notebook_dir, *notebook_dir.parents)
            if (path / "pyproject.toml").exists()
        )
        for path in (notebook_dir, repo_root):
            if str(path) not in sys.path:
                sys.path.insert(0, str(path))
        os.chdir(notebook_dir)
        """,
        before=0,
    )


def inject_mock_base_dir(tb, tmp_path):
    """Inject a mock base directory for testing notebooks.

    Args:
        tb (testbook): testbook object for inserting code into the notebook
        tmp_path: pytest tmp_path fixture for experiment directory

    Returns:
        None
    """
    tb.inject(
        f"""
        from unittest.mock import MagicMock
        from pathlib import Path
        import queens.utils.config_directories
        mock_base_dir = Path('{tmp_path}')
        queens.utils.config_directories.base_directory = MagicMock(return_value=mock_base_dir)
        """,
        before=0,
    )


TUTORIAL_NOTEBOOKS_BY_MARKER = {
    "tutorial_tests": [
        relative_path_from_root("tutorials/1_grid_iterator_rosenbrock.ipynb").as_posix(),
        relative_path_from_root(
            "tutorials/2_uncertainty_propagation_and_quantification.ipynb"
        ).as_posix(),
    ],
    "tutorial_tests_fourc": [
        relative_path_from_root(
            "tutorials/3_orchestrating_4c_simulations/3_orchestrating_4c_simulations.ipynb"
        ).as_posix(),
        relative_path_from_root(
            "tutorials/4_quantifying_uncertainty_due_to_heterogeneous_material_fields/"
            "4_quantifying_uncertainty_due_to_heterogeneous_material_fields.ipynb"
        ).as_posix(),
    ],
    "tutorial_tests_remote": [
        relative_path_from_root(
            "tutorials/5_grid_iterator_4c_remote/5_grid_iterator_4c_remote.ipynb"
        ).as_posix(),
    ],
}
ALL_TUTORIAL_NOTEBOOKS = tuple(
    sorted(path.as_posix() for path in relative_path_from_root(Path("tutorials")).rglob("*.ipynb"))
)


def _format_paths(paths):
    """Format paths for collection error messages."""
    return "\n".join(f"  - {path}" for path in sorted(paths))


def _duplicate_paths(paths):
    """Return paths that occur more than once."""
    return sorted(path for path, count in Counter(paths).items() if count > 1)


def _validate_tutorial_notebook_markers():
    """Ensure every tutorial notebook has at least one explicit marker."""
    discovered_notebooks = set(ALL_TUTORIAL_NOTEBOOKS)
    marked_notebooks = set(chain.from_iterable(TUTORIAL_NOTEBOOKS_BY_MARKER.values()))
    missing_notebooks = discovered_notebooks - marked_notebooks
    stale_notebooks = marked_notebooks - discovered_notebooks
    base_notebooks = set(TUTORIAL_NOTEBOOKS_BY_MARKER["tutorial_tests"])
    specialized_notebooks = set(
        chain.from_iterable(
            notebook_paths
            for marker_name, notebook_paths in TUTORIAL_NOTEBOOKS_BY_MARKER.items()
            if marker_name != "tutorial_tests"
        )
    )
    overlapping_base_notebooks = base_notebooks & specialized_notebooks

    duplicate_marker_assignments = {
        marker_name: duplicates
        for marker_name, notebook_paths in TUTORIAL_NOTEBOOKS_BY_MARKER.items()
        if (duplicates := _duplicate_paths(notebook_paths))
    }

    if not (
        missing_notebooks
        or stale_notebooks
        or overlapping_base_notebooks
        or duplicate_marker_assignments
    ):
        return

    error_parts = ["Tutorial notebook marker assignments are out of sync."]
    if missing_notebooks:
        error_parts.append(
            "Add marker assignments for these notebooks:\n" f"{_format_paths(missing_notebooks)}"
        )
    if stale_notebooks:
        error_parts.append(
            "Remove marker assignments for missing notebooks:\n" f"{_format_paths(stale_notebooks)}"
        )
    if overlapping_base_notebooks:
        error_parts.append(
            "Remove tutorial_tests from notebooks with specialized tutorial markers:\n"
            f"{_format_paths(overlapping_base_notebooks)}"
        )
    if duplicate_marker_assignments:
        duplicate_entries = [
            f"{marker_name}: {notebook_path}"
            for marker_name, notebook_paths in duplicate_marker_assignments.items()
            for notebook_path in notebook_paths
        ]
        error_parts.append(
            "Remove duplicate marker assignments:\n" f"{_format_paths(duplicate_entries)}"
        )

    raise pytest.UsageError("\n\n".join(error_parts))


def marker_names_for_notebook(notebook_path):
    """Return marker names assigned to a tutorial notebook."""
    notebook_path = Path(notebook_path).as_posix()
    return [
        marker_name
        for marker_name, notebook_paths in TUTORIAL_NOTEBOOKS_BY_MARKER.items()
        if notebook_path in notebook_paths
    ]


def markers_for_notebook(notebook_path):
    """Return pytest markers assigned to a tutorial notebook."""
    return [
        getattr(pytest.mark, marker_name)
        for marker_name in marker_names_for_notebook(notebook_path)
    ]


def notebook_param(notebook_path):
    """Return a pytest parameter for a tutorial notebook."""
    notebook_path = Path(notebook_path).as_posix()
    return pytest.param(
        notebook_path,
        marks=markers_for_notebook(notebook_path),
        id=notebook_path,
    )
