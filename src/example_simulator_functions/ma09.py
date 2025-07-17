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
"""Ma test function."""

import numpy as np


def ma09(x1, x2):
    r"""Ma09 function: Two dimensional benchmark for UQ defined in [1].

    :math:`f({\bf x}) = \frac{1}{|0.3-x_1^2 - x_2^2|+0.1}`

    Args:
        x1 (float): Input parameter 1 in [0, 1]
        x2 (float): Input parameter 2 in [0, 1]

    Returns:
        float: Value of the `ma09` function


    References:
        [1] Ma, X., & Zabaras, N. (2009). An adaptive hierarchical sparse grid
            collocation algorithm for the solution of stochastic differential
            equations. Journal of Computational Physics, 228(8), 3084?3113.
    """
    y = 1 / (np.abs(0.3 - x1**2 - x2**2) + 0.1)

    return y
