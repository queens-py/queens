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
"""Bayesian optimization iterator."""

import logging
import time
from copy import deepcopy

import numpy as np
from scipy.optimize import OptimizeResult, minimize

from queens.distributions.uniform import Uniform
from queens.iterators.optimization import OptimizationBase
from queens.parameters.parameters import Parameters
from queens.utils.logger_settings import log_init_args
from queens.utils.sobol_sequence import sample_sobol_sequence

_logger = logging.getLogger(__name__)


class BayesianOptimization(OptimizationBase):
    """Iterator for Gaussian-process Bayesian optimization.

    This iterator minimizes a bounded scalar objective returned by a QUEENS model. It delegates
    surrogate setup, training, scaling, hyperparameter optimization, and prediction to a QUEENS
    surrogate model. A user-given acquisition function is maximized to select one new expensive
    model evaluation per iteration.

    Attributes:
        surrogate_model (Surrogate): QUEENS probabilistic surrogate model.
        acquisition_function (AcquisitionFunction): Acquisition function used
            to select new evaluation points.
        bounds (np.ndarray): Lower and upper parameter bounds.
        num_initial_samples (int): Number of initial model evaluations.
        max_evaluations (int): Total number of model evaluations.
        num_acquisition_restarts (int): Number of starts used to maximize
            expected improvement.
        result_description (dict): Description of requested result processing.
        solution (OptimizeResult): Result of the optimization.
    """

    @log_init_args
    def __init__(
        self,
        model,
        parameters,
        global_settings,
        surrogate_model,
        acquisition_function,
        bounds,
        num_initial_samples,
        max_evaluations,
        result_description,
        num_acquisition_restarts=10,
    ):
        """Initialize the Bayesian optimization iterator.

        Args:
            model (Model): Model evaluated by the iterator.
            parameters (Parameters): QUEENS parameters object.
            global_settings (GlobalSettings): QUEENS global settings.
            surrogate_model (Surrogate): QUEENS surrogate with ``setup``, ``train``, and
                ``predict`` methods. Its ``predict`` method must return ``result`` and
                ``variance`` entries.
            acquisition_function (AcquisitionFunction): Acquisition function used to select new
                evaluation points. It must provide an ``evaluate`` method.
            bounds (array_like): Bounds with shape ``(num_parameters, 2)``.
            num_initial_samples (int): Number of samples in the initial design.
            max_evaluations (int): Total model-evaluation budget.
            result_description (dict): Description of result processing.
            num_acquisition_restarts (int): Number of multi-start L-BFGS-B optimizations used to
                maximize expected improvement.
        """
        super().__init__(model, parameters, global_settings, result_description)

        self.bounds = np.asarray(bounds, dtype=float)
        expected_shape = (parameters.num_parameters, 2)

        if self.bounds.shape != expected_shape:
            raise ValueError(f"Bounds must have shape {expected_shape}, got {self.bounds.shape}.")
        if not np.all(np.isfinite(self.bounds)):
            raise ValueError("All bounds must be finite.")
        if np.any(self.bounds[:, 0] >= self.bounds[:, 1]):
            raise ValueError("Each lower bound must be smaller than its upper bound.")
        if num_initial_samples < 2:
            raise ValueError("At least two initial samples are required.")
        if max_evaluations < num_initial_samples:
            raise ValueError(
                "max_evaluations must be greater than or equal to num_initial_samples."
            )
        if num_acquisition_restarts < 1:
            raise ValueError("At least one acquisition restart is required.")

        self.surrogate_model = surrogate_model
        self.acquisition_function = acquisition_function
        self.num_initial_samples = int(num_initial_samples)
        self.max_evaluations = int(max_evaluations)
        self.num_acquisition_restarts = int(num_acquisition_restarts)

        distributions = {
            parameter_name: Uniform(lower_bound, upper_bound)
            for parameter_name, (lower_bound, upper_bound) in zip(
                self.parameters.parameters_keys,
                self.bounds,
                strict=True,
            )
        }
        self.sampling_parameters = Parameters(**distributions)

        self.random_generator = np.random.default_rng(42)
        self.positions = None
        self.objective_values = None

        gp_optimizer = getattr(self.surrogate_model, "stochastic_optimizer", None)
        self._gp_optimizer_template = deepcopy(gp_optimizer) if gp_optimizer is not None else None

    def objective(self, x0):
        """Evaluate objective (acquisition) function at *x0*.

        Args:
            x0 (np.array): position to evaluate objective at

        Returns:
            f0 (float): Objective (acquisition) function evaluated at *x0*
        """
        x0 = np.asarray(x0, dtype=float).reshape(1, -1)
        prediction = self.surrogate_model.predict(x0, support="f")

        mean = float(np.asarray(prediction["result"]).reshape(-1)[0])
        variance = float(np.asarray(prediction["variance"]).reshape(-1)[0])
        standard_deviation = float(np.sqrt(max(variance, 0.0)))
        best_objective = float(np.min(self.objective_values))

        f0 = self.acquisition_function.evaluate(
            mean=mean, standard_deviation=standard_deviation, best_objective=best_objective
        )

        return f0

    def core_run(self):
        """Run sequential Bayesian optimization."""
        _logger.info("Welcome to Bayesian optimization core run.")
        start_time = time.time()

        # set up initial surrogate model
        self.positions = sample_sobol_sequence(
            dimension=self.parameters.num_parameters,
            number_of_samples=self.num_initial_samples,
            parameters=self.sampling_parameters,
            randomize=True,
            seed=42,
        )
        self.objective_values = np.atleast_1d(self.eval_model(self.positions)).astype(float)

        num_bayesian_iterations = self.max_evaluations - self.num_initial_samples

        for iteration in range(num_bayesian_iterations):
            # update surrogate model
            objective_values = self.objective_values.reshape(-1, 1)

            if not np.all(np.isfinite(objective_values)):
                raise ValueError(f"Non-finite objective values: {objective_values.flatten()}")

            if self._gp_optimizer_template is not None:
                self.surrogate_model.stochastic_optimizer = deepcopy(self._gp_optimizer_template)

            self.surrogate_model.setup(self.positions, objective_values)
            self.surrogate_model.train()

            # optimize acquisition function
            new_position = self.maximize_acquisition_function()
            new_objective = float(np.asarray(self.eval_model(new_position)).reshape(-1)[0])

            self.positions = np.vstack((self.positions, new_position))
            self.objective_values = np.append(self.objective_values, new_objective)

            best_index = int(np.argmin(self.objective_values))

            _logger.info(
                "Bayesian optimization iteration %d/%d: current objective=%g at %s",
                iteration + 1,
                num_bayesian_iterations,
                new_objective,
                new_position,
            )
            _logger.info(
                "Bayesian optimization iteration %d/%d: best objective=%g at %s",
                iteration + 1,
                num_bayesian_iterations,
                self.objective_values[best_index],
                self.positions[best_index],
            )

        best_index = int(np.argmin(self.objective_values))
        self.solution = OptimizeResult(
            x=self.positions[best_index].copy(),
            fun=float(self.objective_values[best_index]),
            nit=num_bayesian_iterations,
            nfev=len(self.objective_values),
            success=True,
            status=0,
            message="Maximum number of Bayesian optimization evaluations reached.",
        )
        self.solution["positions"] = self.positions.copy()
        self.solution["objective_values"] = self.objective_values.copy()

        _logger.info("Bayesian optimization took %E seconds.", time.time() - start_time)

    def maximize_acquisition_function(self):
        """Maximize acquisition function using multi-start L-BFGS-B.

        Returns:
            np.ndarray: Position that approximately maximizes expected improvement.

        Raises:
            RuntimeError: If no candidate different from existing positions can be found.
        """
        initial_positions = sample_sobol_sequence(
            dimension=self.parameters.num_parameters,
            number_of_samples=self.num_acquisition_restarts,
            parameters=self.sampling_parameters,
            randomize=True,
            seed=int(self.random_generator.integers(0, np.iinfo(np.int32).max)),
        )
        candidate_positions = list(initial_positions)

        for initial_position in initial_positions:
            optimization_result = minimize(
                lambda x0: -self.objective(x0),
                initial_position,
                method="L-BFGS-B",
                bounds=self.bounds,
            )

            if optimization_result.success:
                candidate_positions.append(optimization_result.x)

        acquisition_values = np.asarray([self.objective(x0) for x0 in candidate_positions])
        descending_indices = np.argsort(acquisition_values)[::-1]

        for index in descending_indices:
            candidate = np.asarray(candidate_positions[index], dtype=float)
            if self.check_precalculated(candidate) is None:
                return candidate

        raise RuntimeError("Acquisition maximization produced no new position.")
