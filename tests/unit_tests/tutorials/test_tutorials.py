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

import pytest
from testbook import testbook

from test_utils.unit_tests.notebooks import inject_mock_path


# tested jupyter notebooks should be added to the list below
@pytest.mark.parametrize(
    "notebook_path",
    [
        "tutorials/1-grid-iterator-rosenbrock.ipynb",
        "tutorials/2-uncertainty-propagation-and-quantification.ipynb",
    ],
)
def test_notebooks(tmp_path, notebook_path):
    """Parameterized test case for multiple Jupyter notebooks.

    The notebook is run and it is checked that it runs through without
    any errors/assertions.
    """
    with testbook(notebook_path) as tb:
        # Patch base_directory to avoid writing test data to user's home dir.
        # Note that tb.patch converts the mocked Path to a string, so we have to use tb.inject.
        inject_mock_path(tb, tmp_path)

        # execute the notebook
        tb.inject("from queens.utils import config_directories")
        tb.inject(f"config_directories = '{tmp_path}'")
        tb.inject(f"output_dir = '{tmp_path}'")
        tb.execute()
