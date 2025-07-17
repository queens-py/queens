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
"""Gradner2014a function.

This is a two-dimensional benchmark function for constraint Bayesian
optimization.
"""

import numpy as np


def gardner14a(x1, x2):
    r"""Gradner2014a function.

    Two-dimensional benchmark function for constraint Bayesian optimization [1]:

    :math:`f({\bf x}) = \cos(2x_1)\cos(x_2)+ \sin(x_1)`

    with the corresponding constraint function:

    :math:`c({\bf x}) = \cos(x_1)\cos(x_2) - \sin(x_1)\sin(x_2)`

    with

    :math:`c({\bf x}) \leq 0.5`


    Args:
        x1 (float): Input parameter 1 in [0, 6]
        x2 (float): Input parameter 2 in [0, 6]

    Returns:
        np.ndarray: Value of the *gardner2014a* function, value of corresponding
        constraint function


    References:
        [1] Gardner, Jacob R., Matt J. Kusner, Zhixiang Eddie Xu, Kilian Q.
            Weinberger, and John P. Cunningham. "Bayesian Optimization with
            Inequality Constraints." In ICML, pp. 937-945. 2014
    """
    y = np.cos(2 * x1) * np.cos(x2) + np.sin(x1)
    c = np.cos(x1) * np.cos(x2) - np.sin(x1) * np.sin(x2)

    return np.array([y, c])
