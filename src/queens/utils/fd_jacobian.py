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
"""Calculate finite-difference-based approximation of Jacobian.

Note:
    Implementation is heavily based on the *scipy.optimize._numdiff* module.
    We do NOT support complex scheme 'cs' and sparsity.

The motivation behind this reimplementation is to enable the parallel computation of all function
values required for the finite difference scheme.

In theory, when computing the Jacobian of function at a specific position via a specific finite
difference scheme, all positions where the function needs to be evaluated (the perturbed positions)
are known immediately/at once, because they do not depend on each other. The evaluation of the
function at these perturbed positions may consequently be done "perfectly" (embarrassingly)
parallel.

Most implementations of finite-difference-based approximations do not exploit this inherent
potential for parallel evaluations because for cheap functions, the communication overhead is too
high. For expensive functions, the exploitation ensures significant speed up.
"""

from typing import Literal

import numpy as np
from scipy.optimize._numdiff import (
    _adjust_scheme_to_bounds,
    _compute_absolute_step,
    _prepare_bounds,
)


def compute_step_with_bounds(
    x0: np.ndarray,
    method: Literal["2-point", "3-point"],
    rel_step: float | np.ndarray | None,
    bounds: tuple | np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute step sizes of finite difference scheme adjusted to bounds.

    Args:
        x0: Point at which the derivative shall be evaluated
        method: Finite difference method to use:

            * `2-point`:
                use the first order accuracy forward or backward difference
            * `3-point`:
                use central difference in interior points and the second order accuracy forward or
                backward difference near the boundary

        rel_step: Relative step size to use. The absolute step size is computed as
            `h = rel_step * sign(x0) * max(1, abs(x0))`, possibly adjusted to fit into the bounds.
            For *method='3-point'* the sign of *h* is ignored.
            If None (default) then step is selected automatically, see Notes.
        bounds: Lower and upper bounds on independent variables. Defaults to no bounds.
            Each bound must match the size of *x0* or be a scalar, in the latter case the bound will
            be the same for all variables. Use it to limit the range of function evaluation.

    Returns:
        Adjusted step sizes
        Whether to switch to one-sided scheme due to closeness to bounds. Informative only for
            3-point method
    """
    lb, ub = _prepare_bounds(bounds, x0)

    if lb.shape != x0.shape or ub.shape != x0.shape:
        raise ValueError("Inconsistent shapes between bounds and `x0`.")
    # f0: empty array that the method _compute_absolute_step requires now
    # not needed for computation only datatype is checked once in method
    # seems to be a bug in scipy
    f0 = np.array([])
    h = _compute_absolute_step(rel_step, x0, f0, method)

    if method == "2-point":
        h, use_one_sided = _adjust_scheme_to_bounds(x0, h, 1, "1-sided", lb, ub)
    elif method == "3-point":
        h, use_one_sided = _adjust_scheme_to_bounds(x0, h, 1, "2-sided", lb, ub)
    else:
        raise NotImplementedError(f"Method '{method}' is not implemented.")

    return h, use_one_sided


def get_positions(
    x0: np.ndarray,
    method: Literal["2-point", "3-point"],
    rel_step: np.ndarray | None,
    bounds: tuple | np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute all positions needed for the finite difference approximation.

    The Jacobian is defined for a vector-valued function at a given position.

    **Note:** The implementation is supposed to remain very closed to
    *scipy._numdiff.approx_derivative*.

    Args:
        x0: Position or sample at which the Jacobian shall be computed.
        method: Finite difference method that is used to compute the Jacobian.
        rel_step: Finite difference step size.
        bounds: Lower and upper bounds on independent variables. Defaults to no bounds. Each bound
            must match the size of *x0* or be a scalar, in the latter case the bound will be the
            scalar, in the latter case the bound will be the same for all variables. Use it to limit
            the range of function evaluation.

    Returns:
        List with additional stencil positions that are necessary to calculate the finite difference
            approximation to the gradient
        Delta between positions used to approximate Jacobian
    """
    h, use_one_sided = compute_step_with_bounds(x0, method, rel_step, bounds)

    h_vecs = np.diag(h)

    x1_stack = []
    x2_stack = []
    dx_stack = []

    for i in range(h.size):
        if method == "2-point":
            x1 = x0 + h_vecs[i]
            dx = x1[i] - x0[i]  # Recompute dx as exactly representable number.
            # df = fun(x1) - f0
        elif method == "3-point" and use_one_sided[i]:
            x1 = x0 + h_vecs[i]
            x2 = x0 + 2 * h_vecs[i]
            dx = x2[i] - x0[i]
            # f1 = fun(x1)
            # f2 = fun(x2)
            # df = -3.0 * f0 + 4 * f1 - f2
        elif method == "3-point" and not use_one_sided[i]:
            x1 = x0 - h_vecs[i]
            x2 = x0 + h_vecs[i]
            dx = x2[i] - x1[i]
            # f1 = fun(x1)
            # f2 = fun(x2)
            # df = f2 - f1
        else:
            raise NotImplementedError(f"Method '{method}' is not implemented.")

        x1_stack.append(x1)
        if method == "3-point":
            x2_stack.append(x2)
        dx_stack.append(np.array([dx]))

    additional_positions = np.atleast_2d(x1_stack)
    if x2_stack:
        additional_positions = np.vstack((additional_positions, np.atleast_2d(x2_stack)))

    delta_positions = np.array(dx_stack)

    return additional_positions, delta_positions, use_one_sided


def fd_jacobian(
    f0: np.ndarray,
    f_perturbed: np.ndarray,
    dx: np.ndarray,
    use_one_sided: np.ndarray,
    method: Literal["2-point", "3-point"],
) -> np.ndarray:
    """Calculate finite difference approximation of Jacobian of *f* at *x0*.

    The necessary function evaluation have been pre-calculated and are supplied via *f0* and the
    *f_perturbed* vector. Each row in *f_perturbed* corresponds to a function evaluation. The shape
    of *f_perturbed* depends heavily on the chosen finite difference scheme (method) and therefore
    the pre-calculation of *f_perturbed* and *dx* has to be consistent with the requested method.

    Supported methods:
    * `2-point`: a one sided scheme by definition
    * `3-point`: more exact but needs twice as many function evaluations

    **Note:** The implementation is supposed to remain very closed to
    *scipy._numdiff.approx_derivative*.

    Args:
        f0: Function value at *x0*, *f0=f(x0)*
        f_perturbed: Perturbed function values
        dx: Deltas of the input variables
        use_one_sided: Whether to switch to one-sided scheme due to closeness to bounds;
            informative only for 3-point method
        method: Which scheme was used to calculate the perturbed function values and deltas
    Returns:
        Jacobian of the underlying model at x0.
    """
    num_feval_perturbed = f_perturbed.shape[0]

    if method == "2-point":
        f1 = np.stack(f_perturbed, axis=0)  # type: ignore[call-overload]

        df = f1 - f0
    elif method == "3-point":
        len_f1 = int(num_feval_perturbed / 2)
        f1 = np.stack(f_perturbed[0:len_f1], axis=0)  # type: ignore[call-overload]
        f2 = np.stack(f_perturbed[len_f1:], axis=0)  # type: ignore[call-overload]
        df = np.empty(f1.shape)
        for i in range(use_one_sided.size):
            if use_one_sided[i]:
                df[i] = -3.0 * f0 + 4 * f1[i] - f2[i]
            else:
                df[i] = f2[i] - f1[i]

    jacobian_transposed = df / dx

    # shape[1] of transposed Jacobian corresponds to the dimension of the objective function.
    # If it is equal to one (i.e., we deal with a scalar objective function),
    # the respective optimization algorithms expect a 1d array (vector).
    # You may think of the gradient as a degenerated Jacobian in the 1d case.
    if jacobian_transposed.shape[1] == 1:
        jacobian_transposed = np.squeeze(jacobian_transposed)
    return jacobian_transposed.T
