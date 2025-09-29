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


# tested jupyter notebooks should be added to the list below
@pytest.mark.parametrize(
    "notebook_path",
    [
        "tutorials/2-grid-iterator-rosenbrock.ipynb",
    ],
)
def test_notebooks(notebook_path):
    """Parameterized test case for multiple Jupyter notebooks.

    The notebook is run and it is checked that it runs through without
    any errors/assertions.
    """
    with testbook(notebook_path) as tb:
        # execute the notebook
        tb.execute()
