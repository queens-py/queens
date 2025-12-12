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
"""Least squares iterator."""

import logging

import numpy as np
from scipy.optimize import Bounds, least_squares

from queens.iterators.optimization import Optimization
from queens.utils.logger_settings import log_init_args

_logger = logging.getLogger(__name__)


class LeastSquares(Optimization):
    """Iterator for least-squares optimization.

    Based on the *scipy.optimize.least_squares* optimization toolbox [1].

    References:
        [1]: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html

    Attributes:
        algorithm (str): Algorithm to perform minimization:

            - trf:
                Trust Region Reflective algorithm, particularly suitable for large sparse problems
                with bounds. Generally robust method.
            - dogbox:
                dogleg algorithm with rectangular trust regions, typical use case is small problems
                with bounds. Not recommended for problems with rank-deficient Jacobian.
            - lm:
                Levenberg-Marquardt algorithm as implemented in MINPACK. Doesn’t handle bounds and
                sparse Jacobians. Usually the most efficient method for small unconstrained
                problems.
    """

    @log_init_args
    def __init__(
        self,
        model,
        parameters,
        global_settings,
        initial_guess,
        result_description,
        verbose_output=False,
        bounds=Bounds(lb=-np.inf, ub=np.inf),
        max_feval=None,
        algorithm="lm",
        jac_method="2-point",
        jac_rel_step=None,
        objective_and_jacobian=True,
    ):
        """Initialize LeastSquares.

        Args:
            model (Model): Model to be evaluated by iterator
            parameters (Parameters): Parameters object
            global_settings (GlobalSettings): settings of the QUEENS experiment including its name
                                              and the output directory
            initial_guess (array like): initial position at which the optimization starts
            result_description (dict): Description of desired post-processing.
            verbose_output (int): Integer encoding which kind of verbose information should be
                                  printed by the optimizers.
            bounds (sequence, Bounds): Bounds on variables for Nelder-Mead, L-BFGS-B, TNC, SLSQP,
                Powell, and trust-constr methods. There are two ways to specify the bounds:

                1. Instance of `Bounds` class.
                2. A sequence with 2 elements. The first element corresponds to a sequence of lower
                   bounds and the second element to a sequence of upper bounds. The length of each
                   of the two subsequences must be equal to the number of variables.

            max_feval (int): Maximum number of function evaluations.
            algorithm (str): Algorithm to perform minimization:

                - 'trf': Trust Region Reflective algorithm, particularly suitable for large sparse
                  problems with bounds. Generally robust method.
                - 'dogbox': dogleg algorithm with rectangular trust regions, typical use case is
                  small problems with bounds. Not recommended for problems with rank-deficient
                  Jacobian.
                - 'lm': Levenberg-Marquardt algorithm as implemented in MINPACK. Doesn’t handle
                  bounds and sparse Jacobians. Usually the most efficient method for small
                  unconstrained problems.

            jac_method (str): Method to calculate a finite difference based approximation of the
                              Jacobian matrix:

                              - '2-point': a one-sided scheme by definition
                              - '3-point': more exact but needs twice as many function evaluations

            jac_rel_step (array_like): Relative step size to use for finite difference approximation
                                       of Jacobian matrix. If None (default) then it is selected
                                       automatically. (see SciPy documentation for details)
            objective_and_jacobian (bool, opt): If true, every time the objective is evaluated also
                                                the jacobian is evaluated. This leads to improved
                                                batching, but can lead to unnecessary evaluations of
                                                the jacobian during line-search.
                                                Default is true.
        """
        super().__init__(
            model=model,
            parameters=parameters,
            global_settings=global_settings,
            initial_guess=initial_guess,
            result_description=result_description,
            verbose_output=verbose_output,
            bounds=bounds,
            max_feval=max_feval,
            algorithm=algorithm,
            jac_method=jac_method,
            jac_rel_step=jac_rel_step,
            objective_and_jacobian=objective_and_jacobian,
        )
        self.algorithm = algorithm  # We don't want algorithm.upper() here

    def core_run(self):
        """Core run of LeastSquares iterator."""
        self.solution = least_squares(
            self.objective,  # pylint: disable=duplicate-code
            self.initial_guess,
            method=self.algorithm,
            jac=self.jacobian,
            bounds=self.bounds,
            max_nfev=self.max_feval,
            verbose=int(self.verbose_output),
        )

    def post_run(self):
        """Analyze the resulting optimum."""
        _logger.info("Optimality:\n\t%s", self.solution.optimality)
        _logger.info("Cost:\n\t%s", self.solution.cost)

        super().post_run()
