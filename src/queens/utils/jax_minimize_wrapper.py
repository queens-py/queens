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
"""A collection of helper functions for optimization with JAX.

Taken from
https://gist.github.com/slinderman/24552af1bdbb6cb033bfea9b2dc4ecfd
"""

from typing import TYPE_CHECKING, Any, Callable, Sequence, TypeAlias

import numpy as np
import scipy.optimize
from jax import grad, jit
from jax.flatten_util import ravel_pytree

if TYPE_CHECKING:
    from scipy.optimize import OptimizeResult

Constraint: TypeAlias = (
    dict
    | scipy.optimize.LinearConstraint
    | scipy.optimize.NonlinearConstraint
    | list[dict]
    | list[scipy.optimize.LinearConstraint]
    | list[scipy.optimize.NonlinearConstraint]
)


def minimize(
    fun: Callable,
    x0: Any,
    method: str | None = None,
    args: tuple = (),
    bounds: Sequence | scipy.optimize.Bounds | None = None,
    constraints: Constraint = (),
    tol: float | None = None,
    callback: Callable | None = None,
    options: dict | None = None,
) -> "OptimizeResult":
    """A simple wrapper for scipy.optimize.minimize using JAX.

    Args:
        fun: The objective function to be minimized, written in JAX code so that it is automatically
            differentiable. It is of type, ```fun: x, *args -> float``` where `x` is a PyTree and
            args is a tuple of the fixed parameters needed to completely specify the function.
        x0: Initial guess represented as a JAX PyTree.
        args: Extra arguments passed to the objective function and its derivative. Must consist of
            valid JAX types; e.g. the leaves of the PyTree must be floats.
            The remainder of the keyword arguments are inherited from
            `scipy.optimize.minimize`, and their descriptions are copied here for
            convenience.
        method: Type of solver. Should be one of

            - 'Nelder-Mead'
            - 'Powell'
            - 'CG'
            - 'BFGS'
            - 'Newton-CG'
            - 'L-BFGS-B'
            - 'TNC'
            - 'COBYLA'
            - 'SLSQP'
            - 'trust-constr'
            - 'dogleg'
            - 'trust-ncg'
            - 'trust-exact'
            - 'trust-krylov'
            - custom - a callable object (added in version 0.14.0),
              see below for description.

            If not given, chosen to be one of ``BFGS``, ``L-BFGS-B``, ``SLSQP``, depending on if
            the problem has constraints or bounds.
        bounds: Bounds on variables for L-BFGS-B, TNC, SLSQP, Powell, and trust-constr methods.
            There are two ways to specify the bounds:

            1. Instance of `Bounds` class.
            2. Sequence of ``(min, max)`` pairs for each element in `x`.

            None is used to specify no bounds. Note that in order to use `bounds` you will need to
            manually flatten them in the same order as your inputs `x0`.
        constraints: Constraints definition (only for COBYLA, SLSQP and trust-constr).
            Constraints for 'trust-constr' are defined as a single object or a list of objects
            specifying constraints to the optimization problem. Constraints for COBYLA, SLSQP are
            defined as a list of dictionaries. Each dictionary with fields:

                type : str
                    Constraint type: 'eq' for equality, 'ineq' for inequality.
                fun : callable
                    The function defining the constraint.
                jac : callable, optional
                    The Jacobian of `fun` (only for SLSQP).
                args : sequence, optional
                    Extra arguments to be passed to the function and Jacobian.

            Equality constraint means that the constraint function result is to be zero whereas
            inequality means that it is to be non-negative.
            Note that COBYLA only supports inequality constraints.
            Note that in order to use `constraints` you will need to manually flatten them in the
            same order as your inputs `x0`.
        tol: Tolerance for termination. For detailed control, use solver-specific options.
        options: A dictionary of solver options. All methods accept the following generic options:

                maxiter : int
                    Maximum number of iterations to perform. Depending on the
                    method each iteration may use several function evaluations.
                disp : bool
                    Set to True to print convergence messages.

            For method-specific options, see :func:`show_options()`.
        callback: Called after each iteration. For 'trust-constr' it is a callable with the
            signature: ``callback(xk, OptimizeResult state) -> bool`` where ``xk`` is the current
            parameter vector represented as a PyTree, and ``state`` is an `OptimizeResult` object,
            with the same fields as the ones from the return. If callback returns True the algorithm
            execution is terminated.
            For all the other methods, the signature is: ``callback(xk)`` where ``xk`` is the
            current parameter vector, represented as a PyTree.

    Returns:
        The optimization result represented as an OptimizeResult object.
            Important attributes are:

            - ``x``: the solution array, represented as a JAX PyTree
            - ``success``: a Boolean flag indicating if the optimizer exited successfully
            - ``message``: describes the cause of the termination.

            See ``scipy.optimize.OptimizeResult`` for a description of other attributes.
    """
    # Use tree flatten and unflatten to convert params x0 from PyTrees to flat arrays
    x0_flat, unravel = ravel_pytree(x0)

    # Wrap the objective function to consume flat _original_
    # numpy arrays and produce scalar outputs.
    def fun_wrapper(x_flat: Any, *args: Any) -> float:
        x = unravel(x_flat)
        return float(fun(x, *args))

    # Wrap the gradient in a similar manner
    jac = jit(grad(fun))

    def jac_wrapper(x_flat: Any, *args: Any) -> np.ndarray:
        x = unravel(x_flat)
        g_flat, _ = ravel_pytree(jac(x, *args))  # pylint: disable=not-callable
        return np.array(g_flat)

    # Wrap the callback to consume a pytree
    def callback_wrapper(x_flat: Any, *args: Any) -> bool | None:
        if callback is not None:
            x = unravel(x_flat)
            return callback(x, *args)
        return None

    # Minimize with scipy
    results = scipy.optimize.minimize(
        fun_wrapper,
        x0_flat,
        args=args,
        method=method,
        jac=jac_wrapper,
        callback=callback_wrapper,
        bounds=bounds,
        constraints=constraints,
        tol=tol,
        options=options,
    )

    # pack the output back into a PyTree
    results["x"] = unravel(results["x"])
    return results
