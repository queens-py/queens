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
"""Agawal09 function.

[1]: Agarwal, N., & Aluru, N. R. (2009). A domain adaptive stochastic
collocation approach for analysis of MEMS under uncertainties. Journal
of Computational Physics, 228(20), 7662?7688.
http://doi.org/10.1016/j.jcp.2009.07.014
"""

import numpy as np


def agawal09a(x1, x2, a1=0.5, a2=0.5):
    r"""Compute the Agawal09a function.

    Two dimensional benchmark function for UQ approaches proposed in [1].

    The function is defined as follows:

    .. math::
       f({\bf x}) = \begin{cases}
            0  & \textrm{if } x_1 > \alpha_1 \textrm{ or } x_2 > \alpha_2 \\
            \sin(\pi x_1 )\sin(\pi x_2 ) &  \textrm{otherwise}
            \end{cases}



    Distribution of the input random variables is probably uniform on [0,1].

    Args:
        x1 (float): First input parameter
        x2 (float): Second input parameter
        a1 (float): Coefficient (optional), with default value 0.5
        a2 (float): Coefficient (optional), with default value 0.5

    Returns:
        float: Value of `agawal09a` function
    """
    if x1 > a1 or x2 > a2:
        y = 0
    else:
        y = np.sin(np.pi * x1) * np.sin(np.pi * x2)

    return y
