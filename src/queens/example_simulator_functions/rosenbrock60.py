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
"""Rosenbrock function.

[1]: Rosenbrock, H. H. (1960). An Automatic Method for Finding the
Greatest or Least Value of a      Function. The Computer Journal, 3(3),
175–184. doi:10.1093/comjnl/3.3.175
"""

import numpy as np


def rosenbrock60(x1, x2):
    """Rosenbrocks banana function.

    Args:
        x1 (float):  Input parameter 1
        x2 (float):  Input parameter 2

    Returns:
        float: Value of the Rosenbrock function
    """
    a = 1.0 - x1
    b = x2 - x1 * x1
    return a * a + b * b * 100.0


def rosenbrock60_residual(x1, x2):
    """Residuals of the Rosenbrock banana function.

    Args:
        x1 (float):  Input parameter 1
        x2 (float):  Input parameter 2

    Returns:
        ndarray: Vector of residuals of the Rosenbrock function
    """
    res1 = 10.0 * (x2 - x1 * x1)
    res2 = 1.0 - x1

    return np.array([res1, res2])


def rosenbrock60_residual_1d(x1):
    """Residuals of the Rosenbrock banana function.

    Args:
        x1 (float):  Input parameter 1

    Returns:
        ndarray: Vector of residuals of the Rosenbrock function
    """
    return rosenbrock60_residual(x1, x2=1)


def rosenbrock60_residual_3d(x1, x2, x3):
    """Residuals of the Rosenbrock banana function.

    Args:
        x1 (float):  Input parameter 1
        x2 (float):  Input parameter 2
        x3 (float):  Input parameter 3

    Returns:
        ndarray: Vector of residuals of the Rosenbrock function
    """
    # Reuse the rosenbrock60_residual function for res1 and res2
    res1, res2 = rosenbrock60_residual(x1, x2)
    res3 = x3**2

    return np.array([res1, res2, res3])
