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
"""Path helpers for the QUEENS test suite.

These helpers resolve paths in the repository checkout for tests and
tutorials, including tests that build and install QUEENS from a wheel.
Runtime QUEENS code must not import this module; runtime path helpers
belong in ``queens.utils.path``.
"""

from pathlib import Path

PATH_TO_ROOT = Path(__file__).parents[1]


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
