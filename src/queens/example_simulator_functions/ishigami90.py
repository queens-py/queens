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
"""Ishigami function.

Ishigami function is a three-dimensional test function for sensitivity
analysis and UQ.

It is nonlinear and non-monotonous.
"""

import numpy as np

# default parameter values
P1 = 7
P2 = 0.1


def ishigami90(x1, x2, x3, p1=P1, p2=P2):
    r"""Three-dimensional benchmark function.

    Three-dimensional benchmark function from [2], used for UQ because it
    exhibits strong nonlinearity and non-monotonicity.
    It also has a peculiar dependence on :math:`x_3`, as described [5]).
    The values of :math:`a` and :math:`b` used in [1] and [3] are:
    :math:`p_1 = a = 7`
    and
    :math:`p_2 = b = 0.1`. In [5] the values :math:`a = 7` and :math:`b = 0.05` are used.

    The function is defined as:

     :math:`f({\bf x}) = \sin(x_1) + p_1 \sin^2(x_2 +p_2x_3^4\sin(x_1))`

    Typically, distributions of the input random variables are:
    :math:`x_i \sim` Uniform[:math:`-\pi, \pi`], for all i = 1, 2, 3.

    Args:
        x1 (float): Input parameter 1
        x2 (float): Input parameter 2
        x3 (float): Input parameter 3
        p1 (float): Coefficient (optional), with default value 7
        p2 (float): Coefficient (optional), with default value 0.1

    Returns:
        float: Value of the *ishigami* function at [`x1`, `x2`, `x3`]

    References:
        [1] Crestaux, T., Martinez, J.-M., Le Maitre, O., & Lafitte, O. (2007).
            Polynomial chaos expansion for uncertainties quantification and
            sensitivity analysis [PowerPoint slides]. Retrieved from
            SAMO 2007 website: http://samo2007.chem.elte.hu/lectures/Crestaux.pdf.

        [2] Ishigami, T., & Homma, T. (1990, December). An importance
            quantification technique in uncertainty analysis for computer models.
            In Uncertainty Modeling and Analysis, 1990. Proceedings.,

        [3] Marrel, A., Iooss, B., Laurent, B., & Roustant, O. (2009).
            Calculations of sobol indices for the gaussian process metamodel.
            Reliability Engineering & System Safety, 94(3), 742-751.

        [4] Saltelli, A., Chan, K., & Scott, E. M. (Eds.). (2000).
            Sensitivity analysis (Vol. 134). New York: Wiley.

        [5] Sobol, I. M., & Levitan, Y. L. (1999). On the use of variance
            reducing multipliers in Monte Carlo computations of a global
            sensitivity index. Computer Physics Communications, 117(1), 52-61.
    """
    term1 = np.sin(x1)
    term2 = p1 * (np.sin(x2)) ** 2
    term3 = p2 * x3**4 * np.sin(x1)

    return term1 + term2 + term3


def variance(p1=P1, p2=P2):
    """Variance of Ishigami test function.

    According to (50) in [1].

    [1] Homma, T., & Saltelli, A. (1996). Importance measures in global
        sensitivity analysis of nonlinear models. Reliability Engineering &
        System Safety, 52(1), 1–17.
        https://doi.org/10.1016/0951-8320(96)00002-6

    Args:
        p1 (float): Coefficient (optional), with default value 7
        p2 (float): Coefficient (optional), with default value 0.1

    Returns:
        float: Value of variance of the *ishigami* function
    """
    return 0.125 * p1**2 + 0.2 * p2 * np.pi**4 + p2**2 * np.pi**8 / 18 + 0.5


def first_effect_variance(p1=P1, p2=P2):
    """Total variance of the Ishigami test function.

    According to (50)-(53) in [1].

    [1] Homma, T., & Saltelli, A. (1996). Importance measures in global
        sensitivity analysis of nonlinear models. Reliability Engineering &
        System Safety, 52(1), 1–17.
        https://doi.org/10.1016/0951-8320(96)00002-6

    Args:
        p1 (float): Coefficient (optional), with default value 7
        p2 (float): Coefficient (optional), with default value 0.1

    Returns:
        float: Value of first effect (conditional) variance of *ishigami* function
    """
    variance_1 = 0.2 * p2 * np.pi**4 + 0.02 * p2**2 * np.pi**8 + 0.5
    variance_2 = 0.125 * p1**2
    variance_3 = 0
    return np.array([variance_1, variance_2, variance_3])


def first_order_indices(p1=P1, p2=P2):
    """First order Sobol' indices of Ishigami test function.

    According to (50)-(53) in [1].

    [1] Homma, T., & Saltelli, A. (1996). Importance measures in global
        sensitivity analysis of nonlinear models. Reliability Engineering &
        System Safety, 52(1), 1–17.
        https://doi.org/10.1016/0951-8320(96)00002-6

    Args:
        p1 (float): Coefficient (optional), with default value 7
        p2 (float): Coefficient (optional), with default value 0.1

    Returns:
        float: Analytical values of first order Sobol indices of the *ishigami* function
    """
    total_variance = variance(p1=p1, p2=p2)
    variance_i = first_effect_variance(p1=p1, p2=p2)
    return variance_i / total_variance


def total_order_indices(p1=P1, p2=P2):
    """Total order Sobol' indices of Ishigami test function.

    According to (50)-(57) in [1].

    [1] Homma, T., & Saltelli, A. (1996). Importance measures in global
        sensitivity analysis of nonlinear models. Reliability Engineering &
        System Safety, 52(1), 1–17.
        https://doi.org/10.1016/0951-8320(96)00002-6

    Args:
        p1 (float): Coefficient (optional), with default value 7
        p2 (float): Coefficient (optional), with default value 0.1

    Returns:
        float: Analytical values of total order Sobol indices of the *ishigami* function
    """
    total_variance = variance(p1=p1, p2=p2)
    variance_i = first_effect_variance(p1=p1, p2=p2)
    variance_1 = variance_i[0]
    variance_2 = variance_i[1]
    variance_3 = variance_i[2]
    variance_12 = 0
    variance_13 = p2**2 * np.pi**8 * (1.0 / 18.0 - 0.02)
    variance_23 = 0
    variance_123 = 0
    total_variance_1 = variance_1 + variance_12 + variance_13 + variance_123
    total_variance_2 = variance_2 + variance_12 + variance_23 + variance_123
    total_variance_3 = variance_3 + variance_13 + variance_23 + variance_123
    return np.array([total_variance_1, total_variance_2, total_variance_3]) / total_variance
