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
"""Test path utils."""

from pathlib import Path, PurePath

import pytest

import queens
from queens.utils.path import (
    PATH_TO_QUEENS_SOURCE,
    PATH_TO_ROOT,
    _find_path_to_root,
    check_if_path_exists,
    create_folder_if_not_existent,
    is_empty,
    relative_path_from_queens_source,
    relative_path_from_root,
)

THIS_PATH = Path(__file__).parent


def _write_pyproject(path, project_name):
    """Write a minimal pyproject.toml with the given project name."""
    path.mkdir(parents=True, exist_ok=True)
    (path / "pyproject.toml").write_text(
        f'[project]\nname = "{project_name}"\n',
        encoding="utf-8",
    )


@pytest.fixture(name="path_to_root")
def fixture_path_to_root():
    """Path to root."""
    return THIS_PATH.parents[2]


@pytest.fixture(name="path_to_queens_source")
def fixture_path_to_queens_source():
    """Path to QUEENS source."""
    return Path(queens.__file__).parent


def test_path_to_queens_source(path_to_queens_source):
    """Test path to QUEENS source."""
    assert PATH_TO_QUEENS_SOURCE == path_to_queens_source


def test_path_to_root(path_to_root):
    """Test path to root."""
    assert PATH_TO_ROOT == path_to_root


def test_find_path_to_root_falls_back_to_current_working_directory(tmp_path, monkeypatch):
    """Test path to root fallback from an installed package location."""
    installed_candidate = tmp_path / ".venv" / "lib" / "python3.12" / "site-packages"
    installed_candidate.mkdir(parents=True)
    checkout = tmp_path / "checkout"
    _write_pyproject(checkout, "queens")

    monkeypatch.chdir(checkout)

    assert _find_path_to_root(installed_candidate) == checkout


def test_find_path_to_root_falls_back_to_current_working_directory_subdirectories(
    tmp_path, monkeypatch
):
    """Test path to root fallback via pyproject.toml below the cwd."""
    installed_candidate = tmp_path / ".venv" / "lib" / "python3.12" / "site-packages"
    _write_pyproject(installed_candidate, "not-queens")
    workspace = tmp_path / "workspace"
    checkout = workspace / "queens"
    _write_pyproject(checkout, "queens")

    monkeypatch.chdir(workspace)

    assert _find_path_to_root(installed_candidate) == checkout


def test_find_path_to_root_raises_if_queens_pyproject_is_missing(tmp_path, monkeypatch):
    """Test path to root fallback fails without a QUEENS pyproject.toml."""
    installed_candidate = tmp_path / ".venv" / "lib" / "python3.12" / "site-packages"
    _write_pyproject(installed_candidate, "not-queens")
    workspace = tmp_path / "workspace"
    _write_pyproject(workspace, "not-queens")

    monkeypatch.chdir(workspace)

    with pytest.raises(RuntimeError, match="Could not determine the QUEENS project root"):
        _find_path_to_root(installed_candidate)


def test_check_if_path_exists():
    """Test if path exists."""
    current_folder = Path(__file__).parent
    assert check_if_path_exists(current_folder)


def test_check_if_path_exists_not_existing():
    """Test if path does not exist."""
    tmp_path = Path(__file__).parent / "not_existing"
    with pytest.raises(FileNotFoundError):
        check_if_path_exists(tmp_path)


def test_create_folder_if_not_existent(tmp_path):
    """Test if folder is created."""
    new_path = PurePath(tmp_path, "new/path")
    new_path = create_folder_if_not_existent(new_path)
    assert check_if_path_exists(new_path)


def test_relative_path_from_queens_source(path_to_queens_source):
    """Test relative path from QUEENS source."""
    assert (
        relative_path_from_queens_source("../../../tests/unit_tests/utils")
        == path_to_queens_source / "../../../tests/unit_tests/utils"
    )


def test_relative_path_from_root(path_to_root):
    """Test relative path from root."""
    assert (
        relative_path_from_root("tests/unit_tests/utils") == path_to_root / "tests/unit_tests/utils"
    )


def test_is_empty_with_empty_string():
    """Expected True for an empty string."""
    assert is_empty("")


def test_is_empty_with_non_empty_string():
    """Expected False for a non-empty string."""
    assert not is_empty("not_empty")


def test_is_empty_with_path_object():
    """Expected False for a Path object."""
    assert not is_empty(Path("/some/path"))


def test_is_empty_with_empty_list():
    """Expected True for an empty list."""
    assert is_empty([])


def test_is_empty_with_non_empty_list():
    """Expected False for a non-empty list."""
    assert not is_empty(["item"])


def test_is_empty_with_invalid_type():
    """Expected TypeError for invalid type."""
    with pytest.raises(TypeError, match="paths must be a string, a Path object, or a sequence."):
        is_empty(123)  # Integer input to trigger TypeError
