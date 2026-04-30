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

from test_utils.tutorial_tests import inject_mock_base_dir


# tested jupyter notebooks should be added to the list below
@pytest.mark.parametrize(
    "paths_to_tutorial_notebooks",
    [
        str(patch)
        for patch in sorted(Path("tutorials").glob("*.ipynb"))
        if patch.stem
        not in {
            t.stem.removeprefix("test_") for t in Path("tests/tutorial_tests").glob("test_*.py")
        }
    ],
)
def test_notebooks(tmp_path, paths_to_tutorial_notebooks):
    """Parameterized test case for multiple Jupyter notebooks.

    The notebook is run and it is checked that it runs through without
    any errors/assertions.
    """
    with testbook(paths_to_tutorial_notebooks, timeout=-1) as tb:
        # Patch base_directory to avoid writing test data to user's home dir.
        # Note that tb.patch converts the mocked Path to a string, so we have to use tb.inject.
        inject_mock_base_dir(tb, tmp_path)

        # execute the notebook
        tb.execute()
