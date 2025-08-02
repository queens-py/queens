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
"""Sobol's G function.

A test function for sensitivity analysis.

References:
    [1]: Saltelli, A., Annoni, P., Azzini, I., Campolongo, F., Ratto, M., & Tarantola, S. (2010).
    Variance based sensitivity analysis of model output.
    Design and estimator for the total sensitivity index.
    Computer Physics Communications, 181(2), 259–270.
    https://doi.org/10.1016/j.cpc.2009.09.018
"""

import numpy as np

# values  of G*_6 in Table 5 of [1]
DIM = 10
A = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.8, 1, 2, 3, 4])
ALPHA = np.array([2] * DIM)
DELTA = np.array([0] * DIM)


def sobol_g_function(a=A, alpha=ALPHA, delta=DELTA, **kwargs):
    """Compute generalized Sobol's G function.

    With variable dimension. See (33) in [1].
    Default is a 10-dimensional version as defined in [1].

    Args:
        a (ndarray): Vector of a parameter values
        alpha (ndarray): Vector of alpha parameter values
        delta (ndarray): Vector of delta parameter values
        kwargs (dict): Contains the input x_i ~ U(0,1) (uniformly from [0,1])

    Returns:
        double: Value of the Sobol function
    """
    # some safety checks:
    if not np.all(a >= 0):
        raise ValueError("a >= 0 not satisfied")
    if not np.all(alpha > 0):
        raise ValueError("alpha > 0 not satisfied")
    if not (np.all(delta >= 0) and np.all(delta <= 1)):
        raise ValueError("0 <= delta <= 1 not satisfied")

    # From kwargs get the x_i
    x = []
    for key, value in kwargs.items():
        if key.find("x") > -1:
            x.append(value)
    x = np.array(x)

    if not (x.shape == a.shape and x.shape == alpha.shape and x.shape == delta.shape):
        raise ValueError("shape mismatch")

    g = ((1 + alpha) * np.abs(2 * (x + delta - np.floor(x + delta)) - 1) ** alpha + a) / (1 + a)

    return np.prod(g)


def first_effect_variance(a=A, alpha=ALPHA):
    r"""Calculate first effect variance :math:`V_{x_{i}}[E_{x \sim i}[Y|x_i]]`.

    Args:
        a (ndarray): Vector of *a* parameter values
        alpha (ndarray): Vector of *alpha* parameter values

    Returns:
        ndarray: Vector of first order variances
    """
    variance_i = alpha**2 / (1 + 2 * alpha) / (1 + a) ** 2
    return variance_i


def variance(variance_i=first_effect_variance(a=A, alpha=ALPHA)):
    """Calculate variance of Sobol function :math:`V_x[Y]`.

    Args:
        variance_i (ndarray): Vector of first effect variances

    Returns:
        double: Variance of Sobol's G function
    """
    total_variance = np.prod(1 + variance_i) - 1
    return total_variance


def first_order_indices(a=A, alpha=ALPHA):
    """Compute first order indices of the Sobol test function.

    See (32) in [1].

    Args:
        a (ndarray): Vector of *a* parameter values
        alpha (ndarray): Vector of *alpha* parameter values

    Returns:
        ndarray: Vector of first order indices
    """
    variance_i = first_effect_variance(a, alpha)
    total_variance = variance(variance_i=variance_i)
    first_order_index_i = variance_i / total_variance
    return first_order_index_i


def total_order_indices(a=A, alpha=ALPHA):
    """Compute total indices of Sobol test function.

    See (31)-(32) in [1].

    Args:
        a (ndarray): Vector of *a* parameter values
        alpha (ndarray): Vector of *alpha* parameter values

    Returns:
        ndarray: Vector of total order indices
    """
    variance_i = first_effect_variance(a=a, alpha=alpha)
    total_variance = variance(variance_i=variance_i)

    total_index = np.empty(variance_i.shape)
    for i in range(variance_i.shape[0]):
        mask = np.ones(variance_i.shape, dtype=bool)
        mask[i] = False
        total_index[i] = variance_i[i] * np.prod(1 + variance_i[mask]) / total_variance

    return total_index
