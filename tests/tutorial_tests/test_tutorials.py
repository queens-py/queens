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
"""Unit tests for jupyter notebook tutorials."""

from pathlib import Path

import pytest
from testbook import testbook

from test_utils.tutorial_tests import (
    ALL_TUTORIAL_NOTEBOOKS,
    _validate_tutorial_notebook_markers,
    inject_mock_base_dir,
    inject_notebook_execution_context,
    notebook_param,
)

_validate_tutorial_notebook_markers()

TUTORIAL_NOTEBOOKS_WITH_DEDICATED_TESTS = {
    path.stem.removeprefix("test_")
    for path in Path("tests/tutorial_tests").glob("test_*.py")
    if path.name != Path(__file__).name
}


# Notebooks with dedicated output assertions are collected in their own test modules.
@pytest.mark.parametrize(
    "paths_to_tutorial_notebooks",
    [
        notebook_param(path)
        for path in ALL_TUTORIAL_NOTEBOOKS
        if Path(path).stem not in TUTORIAL_NOTEBOOKS_WITH_DEDICATED_TESTS
    ],
)
def test_notebooks(tmp_path, paths_to_tutorial_notebooks):
    """Parameterized test case for multiple Jupyter notebooks.

    The notebook is run and it is checked that it runs through without
    any errors/assertions.
    """
    with testbook(paths_to_tutorial_notebooks, timeout=-1) as tb:
        notebook_dir = Path(paths_to_tutorial_notebooks).resolve().parent
        inject_notebook_execution_context(tb, notebook_dir)

        # Patch base_directory to avoid writing test data to user's home dir.
        # Note that tb.patch converts the mocked Path to a string, so we have to use tb.inject.
        inject_mock_base_dir(tb, tmp_path)

        # execute the notebook
        tb.execute()
