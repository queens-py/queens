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

import numpy as np
from testbook import testbook


# tested jupyter notebooks should be mentioned below
@testbook(
    "tutorials/1-grid-iterator-rosenbrock.ipynb",
)
def test_result_output(tb):
    """Parameterized test case for Jupyter notebook output.

    The notebook is run with injected lines of code, as intended by the
    tutorial
    """
    optimal_fun = 2.986025e-11
    optimal_x = np.array([0.99999463, 0.99998915]).tolist()

    tb.execute_cell([2, 4, 6, 8, 13, 15, 17, 19, 21, 23, 25, 27])
    tb.inject(
        """np.testing.assert_allclose(X1, X1_QUEENS)
np.testing.assert_allclose(X2, X2_QUEENS)
np.testing.assert_allclose(Z, Z_QUEENS)"""
    )

    tb.execute_cell([31])
    tb.inject(f"np.testing.assert_allclose(optimal_fun, {optimal_fun},rtol=1e-4)")
    tb.inject(f"np.testing.assert_allclose(optimal_x, np.array({optimal_x}))")
