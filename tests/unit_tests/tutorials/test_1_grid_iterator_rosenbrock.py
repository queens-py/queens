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
"""Unit tests for 1-grid-iterator-rosenbrock tutorial."""

from testbook import testbook
import numpy as np
import json


# tested jupyter notebooks should be mentioned below
@testbook(
    "doc/source/tutorials/1-grid-iterator-rosenbrock/1-grid-iterator-rosenbrock.ipynb",
    execute=[1, 3, 5, 7, 9, 11],
)
def test_result_output(tb, tmp_path):
    """Parameterized test case for Jupyter notebook output.

    The notebook is run and it is checked that the output of a specific cell matches given input.
    """
    expected_results = np.array(
        [
            [3.609e03],
            [9.040e02],
            [4.010e02],
            [9.000e02],
            [3.601e03],
            [2.509e03],
            [4.040e02],
            [1.010e02],
            [4.000e02],
            [2.501e03],
            [1.609e03],
            [1.040e02],
            [1.000e00],
            [1.000e02],
            [1.601e03],
            [9.090e02],
            [4.000e00],
            [1.010e02],
            [0.000e00],
            [9.010e02],
            [4.090e02],
            [1.040e02],
            [4.010e02],
            [1.000e02],
            [4.010e02],
        ]
    )

    tb.inject(f"output_dir = '{tmp_path}'")
    tb.inject("from queens.utils import config_directories")
    tb.inject("import json")

    tb.inject(f"config_directories = '{tmp_path}'")
    tb.inject('test_results = json.dumps(results["raw_output_data"]["result"].tolist())')

    assert (tb.ref("output_dir")) == str(tmp_path)
    assert tb.ref("config_directories") == str(tmp_path)
    results = np.array(json.loads(tb.ref("test_results")))

    np.testing.assert_allclose(results, expected_results)
