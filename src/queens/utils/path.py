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
"""Path utilities for QUEENS."""

from pathlib import Path
from typing import Sequence

PATH_TO_QUEENS_SOURCE = Path(__file__).parents[1]
PATH_TO_ROOT = Path(__file__).parents[3]


def relative_path_from_queens_source(relative_path: str) -> Path:
    """Create relative path from *src/queens/*.

    For example, to create *src/queens/folder/file.A*, call
    *relative_path_from_queens_source("folder/file.A")* .

    Args:
        relative_path: Path starting from *src/queens/*
    Returns:
        Absolute path to the file
    """
    full_path = PATH_TO_QUEENS_SOURCE / relative_path
    return full_path


def relative_path_from_root(relative_path: str) -> Path:
    """Create relative path from root directory.

    As an example to create: *src/queens/folder/file.A* .

    Call *relative_path_from_root("src/queens/folder/file.A")* .

    Args:
        relative_path: Path starting from the root directory

    Returns:
        Absolute path to the file
    """
    full_path = PATH_TO_ROOT / relative_path
    return full_path


def create_folder_if_not_existent(path: Path | str) -> Path:
    """Create folder if not existent.

    Args:
        path: Path to be created

    Returns:
        Path object
    """
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def check_if_path_exists(path: Path, error_message: str = "") -> bool:
    """Check if a path exists.

    Args:
        path: Path to be checked
        error_message: If an additional message is desired

    Returns:
        `True` if the path exists, `False` otherwise.

    Raises:
        FileNotFoundError: If the path does not exist.
    """
    path_exists = Path(path).exists()

    if not path_exists:
        raise FileNotFoundError(error_message + f"\nPath {path} does not exist.")

    return path_exists


def is_empty(paths: Sequence | str | Path) -> bool:
    """Check whether paths is empty.

    Args:
        paths: (List of) path-like objects
    """
    if not isinstance(paths, (str, Path, list)):
        raise TypeError("paths must be a string, a Path object, or a list.")
    return not bool(paths)
