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
"""Deterministic optimization toolbox."""

import logging
import time

import numpy as np
from scipy.optimize import Bounds, minimize
from scipy.optimize._numdiff import _prepare_bounds

from queens.iterators._iterator import Iterator
from queens.utils.fd_jacobian import fd_jacobian, get_positions
from queens.utils.logger_settings import log_init_args
from queens.utils.process_outputs import write_results

_logger = logging.getLogger(__name__)


class Optimization(Iterator):
    """Iterator for deterministic optimization problems.

    Based on the *scipy.optimize.minimize* optimization toolbox [1].

    References:
        [1]: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html

    Attributes:
        algorithm (str): String that defines the optimization algorithm to be used:

            - CG:
                Conjugate gradient optimization (unconstrained), using Jacobian
            - BFGS:
                Broyden–Fletcher–Goldfarb–Shanno algorithm (quasi-Newton) for optimization
                (iterative method for unconstrained nonlinear optimization), using Jacobian
            - L-BFGS-B:
                Limited memory Broyden–Fletcher–Goldfarb–Shanno algorithm with box constraints (for
                large number of variables)
            - TNC:
                Truncated Newton method (Hessian free) for nonlinear optimization with bounds
                involving a large number of variables. Jacobian necessary
            - SLSQP:
                Sequential Least Squares Programming minimization with bounds and constraints using
                Jacobian
            - COBYLA:
                Constrained Optimization BY Linear Approximation (no Jacobian)
            - NELDER-MEAD:
                Downhill-simplex search method (unconstrained, unbounded) without the need for a
                Jacobian
            - POWELL:
                Powell's conjugate direction method (unconstrained) without the need for a
                Jacobian. Minimizes the function by a bidirectional search along each search vector

        bounds (sequence, Bounds): Bounds on variables for Nelder-Mead, L-BFGS-B, TNC, SLSQP,
            Powell, and trust-constr methods. There are two ways to specify the bounds:

            1. Instance of `Bounds` class.
            2. A sequence with 2 elements. The first element corresponds to a sequence of lower
               bounds and the second element to a sequence of upper bounds. The length of each of
               the two subsequences must be equal to the number of variables.

        cons (np.array): Nonlinear constraints for the optimization.
                         Only for COBYLA, SLSQP and trust-constr
                         (see SciPy documentation for details)
        initial_guess (np.array): Initial guess, i.e. start point of optimization.
        jac_method (str): Method to calculate a finite difference based approximation of the
                          Jacobian matrix:

                          - '2-point': a one-sided scheme by definition
                          - '3-point': more exact but needs twice as many function evaluations

        jac_rel_step (array_like): Relative step size to use for finite difference approximation
                                   of Jacobian matrix. If None (default) then it is selected
                                   automatically. (see SciPy documentation for details)
        max_feval (int): Maximum number of function evaluations.
        result_description (dict): Description of desired post-processing.
        verbose_output (int): Integer encoding which kind of verbose information should be
                              printed by the optimizers.
        precalculated_positions (dict): Dictionary containing precalculated positions and
                                        corresponding model responses.
        solution (np.array): Solution obtained from the optimization process.
        objective_and_jacobian (bool): If true, every time the objective is evaluated also the
                                       jacobian is evaluated. This leads to improved batching, but
                                       can lead to unnecessary evaluations of the jacobian during
                                       line-search. This option is only available for gradient
                                       methods. Default is false.
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
        constraints=None,
        max_feval=None,
        algorithm="L-BFGS-B",
        jac_method="2-point",
        jac_rel_step=None,
        objective_and_jacobian=False,
    ):
        """Initialize an Optimization.

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

            constraints (np.array): Nonlinear constraints for the optimization.
                                    Only for COBYLA, SLSQP and trust-constr
                                    (see SciPy documentation for details)
            max_feval (int): Maximum number of function evaluations.
            algorithm (str): String that defines the optimization algorithm to be used:

                - 'CG': Conjugate gradient optimization (unconstrained), using Jacobian
                - 'BFGS': Broyden–Fletcher–Goldfarb–Shanno algorithm (quasi-Newton) for optimization
                  (iterative method for unconstrained nonlinear optimization), using Jacobian
                - 'L-BFGS-B': Limited memory Broyden–Fletcher–Goldfarb–Shanno algorithm with box
                  constraints (for large number of variables)
                - 'TNC': Truncated Newton method (Hessian free) for nonlinear optimization with
                  bounds
                  involving a large number of variables. Jacobian necessary
                - 'SLSQP': Sequential Least Squares Programming minimization with bounds and
                  constraints using Jacobian
                - 'COBYLA': Constrained Optimization BY Linear Approximation (no Jacobian)
                - 'NELDER-MEAD': Downhill-simplex search method (unconstrained, unbounded) without
                  the need for a Jacobian
                - 'POWELL': Powell's conjugate direction method (unconstrained) without the need
                  for a Jacobian. Minimizes the function by a bidirectional search along each
                  search vector

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
                                                This option is only available for gradient methods.
                                                Default is false.
        """
        super().__init__(model, parameters, global_settings)

        initial_guess = np.atleast_1d(np.array(initial_guess))

        # check sanity of bounds and extract array of lower and upper bounds to unify the bounds
        if not isinstance(bounds, Bounds):
            if len(bounds) == 2:
                lb, ub = bounds
                # lb or ub can be scalars which don't have a len attribute
                if hasattr(lb, "__len__") and hasattr(ub, "__len__"):
                    # warn if definition of bounds is not unique
                    if len(lb) == 2 and len(ub) == 2 and len(initial_guess) == 2:
                        _logger.warning(
                            "Definition of 'bounds' is not unique. "
                            "Make sure to use the 'new' definition of bounds: "
                            "bounds must contains two elements. "
                            "The first element corresponds to an array_like for the lower bounds"
                            "and the second element to an array_like for the upper bounds."
                        )
            else:
                # ensure "new" style bounds
                raise ValueError(
                    "`bounds` must contain 2 elements.\n"
                    "The first element corresponds to an array_like for the lower bounds"
                    "and the second element to an array_like for the upper bounds."
                )
        else:
            lb, ub = np.squeeze(bounds.lb), np.squeeze(bounds.ub)

        # unify the bounds:
        # make sure that each array contains number of variable entries
        # i.e. we need one lower bound and one upper bound per variable
        lb, ub = _prepare_bounds((lb, ub), initial_guess)

        # convert to Bounds object to ensure correct handling by scipy.optimize
        bounds = Bounds(lb=lb, ub=ub)

        constraints_list = []
        if constraints:
            for value in constraints.values():
                # evaluate string of lambda function into real lambda function
                value["fun"] = eval(value["fun"])  # pylint: disable=eval-used
                constraints_list.append(value)

        algorithm = algorithm.upper()

        self.algorithm = algorithm
        self.bounds = bounds
        self.cons = constraints_list
        self.initial_guess = initial_guess
        self.jac_method = jac_method
        self.jac_rel_step = jac_rel_step
        self.max_feval = max_feval
        self.result_description = result_description
        self.verbose_output = verbose_output
        self.precalculated_positions = {"position": [], "output": []}
        self.solution = None
        self.objective_and_jacobian = objective_and_jacobian
        if self.algorithm in ["COBYLA", "NELDER-MEAD", "POWELL"]:
            self.objective_and_jacobian = False

    def objective(self, x0):
        """Evaluate objective function at *x0*.

        Args:
            x0 (np.array): position to evaluate objective at

        Returns:
            f0 (float): Objective function evaluated at *x0*
        """
        if self.objective_and_jacobian:
            f0 = self.evaluate_fd_positions(x0)[0]
        else:
            f0 = self.eval_model(x0)

        parameter_list = self.parameters.parameters_keys
        _logger.info("The intermediate, iterated parameters %s are:\n\t%s", parameter_list, x0)

        return f0

    def jacobian(self, x0):
        """Evaluate Jacobian of objective function at *x0*.

        Args:
            x0 (np.array): position to evaluate Jacobian at
        Returns:
            jacobian (np.array): Jacobian matrix evaluated at *x0*
        """
        f0, f_perturbed, delta_positions, use_one_sided = self.evaluate_fd_positions(x0)
        jacobian = fd_jacobian(
            f0, f_perturbed, delta_positions, use_one_sided, method=self.jac_method
        )
        # sanity checks:
        # in the case of LSQ, the number of residuals needs to be
        # greater or equal to the number of parameters to be fitted
        if self.algorithm == "LSQ" and jacobian.ndim == 2:
            num_res, num_par = jacobian.shape
            if num_res < num_par:
                raise ValueError(
                    f"Number of residuals (={num_res}) has to be greater or equal to"
                    f" number of parameters (={num_par})."
                    f" You have {num_res}<{num_par}."
                )
        return jacobian

    def evaluate_fd_positions(self, x0):
        """Evaluate objective function at finite difference positions.

        Args:
            x0 (np.array): Position at which the Jacobian is computed.

        Returns:
            f0 (ndarray): Objective function value at *x0*
            f_perturbed (np.array): Perturbed function values
            delta_positions (np.array): Delta between positions used to approximate Jacobian
            use_one_sided (np.array): Whether to switch to one-sided scheme due to closeness to
                                      bounds. Informative only for 3-point method.
        """
        additional_positions, delta_positions, use_one_sided = get_positions(
            x0,
            method=self.jac_method,
            rel_step=self.jac_rel_step,
            bounds=(self.bounds.lb, self.bounds.ub),
        )

        # model response should now correspond to objective function evaluated at positions
        positions = np.vstack((x0, additional_positions))
        f_batch = self.eval_model(positions)

        f0 = f_batch[0].reshape(-1)  # first entry corresponds to f(x0)
        f_perturbed = f_batch[1:].reshape(-1, f0.size)
        return f0, f_perturbed, delta_positions, use_one_sided

    def pre_run(self):
        """Pre run of Optimization iterator."""
        _logger.info("Initialize Optimization run.")

    def core_run(self):
        """Core run of Optimization iterator."""
        _logger.info("Welcome to Optimization core run.")
        start = time.time()

        # minimization with bounds using Jacobian
        if self.algorithm in {"L-BFGS-B", "TNC"}:
            self.solution = minimize(
                self.objective,
                self.initial_guess,
                method=self.algorithm,
                jac=self.jacobian,
                bounds=self.bounds,
                options={"maxiter": int(1e4), "disp": self.verbose_output},
            )
        # Constrained Optimization BY Linear Approximation:
        # minimization with constraints without Jacobian
        elif self.algorithm in {"COBYLA"}:
            self.solution = minimize(
                self.objective,
                self.initial_guess,
                method=self.algorithm,
                constraints=self.cons,
                options={"disp": self.verbose_output},
            )
        # Sequential Least SQuares Programming:
        # minimization with bounds and constraints using Jacobian
        elif self.algorithm in {"SLSQP"}:
            self.solution = minimize(
                self.objective,
                self.initial_guess,
                method=self.algorithm,
                jac=self.jacobian,
                bounds=self.bounds,
                constraints=self.cons,
                options={"disp": self.verbose_output},
            )
        # minimization (unconstrained, unbounded) without Jacobian
        elif self.algorithm in {"NELDER-MEAD", "POWELL"}:
            self.solution = minimize(
                self.objective,
                self.initial_guess,
                method=self.algorithm,
                options={"disp": self.verbose_output},
            )
        # minimization (unconstrained, unbounded) using Jacobian
        elif self.algorithm in {"CG", "BFGS"}:
            self.solution = minimize(
                self.objective,
                self.initial_guess,
                method=self.algorithm,
                jac=self.jacobian,
                options={"disp": self.verbose_output},
            )
        end = time.time()
        _logger.info("Optimization took %E seconds.", end - start)

    def post_run(self):
        """Analyze the resulting optimum."""
        _logger.info("The optimum:\n\t%s", self.solution.x)

        if self.result_description:
            if self.result_description["write_results"]:
                write_results(
                    self.solution,
                    self.global_settings.result_file(".pickle"),
                )

    def eval_model(self, positions):
        """Evaluate model at defined positions.

        Args:
            positions (np.ndarray): Positions at which the model is evaluated

        Returns:
            f_batch (np.ndarray): Model response
        """
        positions = positions.reshape(-1, self.parameters.num_parameters)
        f_batch = [None] * len(positions)
        new_positions_to_evaluate = []
        new_positions_batch_id = []
        for i, position in enumerate(positions):
            precalculated_output = self.check_precalculated(position)
            if precalculated_output is None:
                new_positions_to_evaluate.append(position)
                new_positions_batch_id.append(i)
            else:
                f_batch[i] = precalculated_output
        if new_positions_to_evaluate:
            new_positions_to_evaluate = np.array(new_positions_to_evaluate)
            f_new = self.model.evaluate(new_positions_to_evaluate)["result"]
            for position_id, output in zip(new_positions_batch_id, f_new):
                f_batch[position_id] = output
            self.precalculated_positions["position"].extend(new_positions_to_evaluate)
            self.precalculated_positions["output"].extend(f_new)
        f_batch = np.array(f_batch).squeeze()
        return f_batch

    def check_precalculated(self, position):
        """Check if the model was already evaluated at defined position.

        Args:
            position (np.ndarray): Position at which the model should be evaluated

        Returns:
            np.ndarray: Precalculated model response or *None*
        """
        for i, precalculated_position in enumerate(self.precalculated_positions["position"]):
            if np.equal(position, precalculated_position).all():
                return self.precalculated_positions["output"][i]
        return None
