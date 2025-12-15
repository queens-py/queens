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
"""Finite difference model."""

import logging

import numpy as np

from queens.models.simulation import Simulation
from queens.utils.fd_jacobian import fd_jacobian, get_positions
from queens.utils.logger_settings import log_init_args
from queens.utils.valid_options import check_if_valid_options

_logger = logging.getLogger(__name__)

VALID_FINITE_DIFFERENCE_METHODS = ["2-point", "3-point"]


class FiniteDifference(Simulation):
    """Finite difference model.

    Attributes:
        finite_difference_method (str): Method to calculate a finite-difference-based approximation
            of the Jacobian matrix:

            - '2-point': a one-sided scheme by definition
            - '3-point': more exact but needs twice as many function evaluations

        step_size (float): Step size for the finite difference approximation
        bounds (np.array): Lower and upper bounds on independent variables.
            Defaults to no bounds meaning: [-inf, inf]. Each bound must match the size of *x0* or
            be a scalar, in the latter case the bound will be the same for all variables. Use it to
            limit the range of function evaluation.
    """

    @log_init_args
    def __init__(self, scheduler, driver, finite_difference_method, step_size=1e-5, bounds=None):
        """Initialize model.

        Args:
            scheduler (Scheduler): Scheduler for the simulations
            driver (Driver): Driver for the simulations
            finite_difference_method (str): Method to calculate a finite-difference-based
                approximation of the Jacobian matrix:

                - '2-point': a one-sided scheme by definition
                - '3-point': more exact but needs twice as many function evaluations

            step_size (float, opt): Step size for the finite difference approximation
            bounds (tuple of array_like, opt): Lower and upper bounds on independent variables.
                Defaults to no bounds meaning: [-inf, inf]. Each bound must match the size of *x0*
                or be a scalar, in the latter case the bound will be the same for all variables.
                Use it to limit the range of function evaluation.
        """
        super().__init__(scheduler=scheduler, driver=driver)

        check_if_valid_options(VALID_FINITE_DIFFERENCE_METHODS, finite_difference_method)
        self.finite_difference_method = finite_difference_method
        self.step_size = step_size
        _logger.debug(
            "The gradient calculation via finite differences uses a step size of %s.",
            step_size,
        )
        if bounds is None:
            bounds = [-np.inf, np.inf]
        self.bounds = np.array(bounds)

    def _evaluate(self, samples):
        """Evaluate model with current set of input samples.

        Args:
            samples (np.ndarray): Input samples

        Returns:
            response (dict): Response of the underlying model at input samples
        """
        if not self.evaluate_and_gradient_bool:
            self.response = super()._evaluate(samples)
        else:
            self.response = self.evaluate_finite_differences(samples)
        return self.response

    def grad(self, samples, upstream_gradient):
        r"""Evaluate gradient of model w.r.t. current set of input samples.

        Consider current model f(x) with input samples x, and upstream function g(f). The provided
        upstream gradient is :math:`\frac{\partial g}{\partial f}` and the method returns
        :math:`\frac{\partial g}{\partial f} \frac{df}{dx}`.

        Args:
            samples (np.array): Input samples
            upstream_gradient (np.array): Upstream gradient function evaluated at input samples
                                          :math:`\frac{\partial g}{\partial f}`

        Returns:
            gradient (np.array): Gradient w.r.t. current set of input samples
                                 :math:`\frac{\partial g}{\partial f} \frac{df}{dx}`
        """
        gradient = np.sum(upstream_gradient[:, :, np.newaxis] * self.response["gradient"], axis=1)
        return gradient

    def evaluate_finite_differences(self, samples):
        """Evaluate model gradient based on FDs.

        Args:
            samples (np.array): Current samples at which model should be evaluated.

        Returns:
            response (np.array): Array with model response for given input samples
            gradient_response (np.array): Array with row-wise model/objective fun gradients for
                                          given samples.
        """
        num_samples = samples.shape[0]

        # calculate the additional sample points for the stencil per sample
        stencil_samples_lst = []
        delta_positions_lst = []
        for sample in samples:
            stencil_sample, delta_positions, _ = get_positions(
                sample,
                method=self.finite_difference_method,
                rel_step=self.step_size,
                bounds=self.bounds,
            )
            stencil_samples_lst.append(stencil_sample)
            delta_positions_lst.append(delta_positions)

        num_stencil_points_per_sample = stencil_sample.shape[1]
        stencil_samples = np.array(stencil_samples_lst).reshape(-1, num_stencil_points_per_sample)

        # stack samples and stencil points and evaluate entire batch
        combined_samples = np.vstack((samples, stencil_samples))
        all_responses = super()._evaluate(combined_samples)["result"]

        # Correct the number of model calls by the number of additional evaluations
        # from the finite difference stencil
        self.num_evaluations += len(combined_samples) - len(samples)
        all_responses = all_responses.reshape(combined_samples.shape[0], -1)

        response = all_responses[:num_samples, :]
        additional_response_lst = np.array_split(all_responses[num_samples:, :], num_samples)

        # calculate the model gradients re-using the already computed model responses
        model_gradients_lst = []
        for output, delta_positions, additional_model_output_stencil in zip(
            response, delta_positions_lst, additional_response_lst
        ):
            model_gradients_lst.append(
                fd_jacobian(
                    output.reshape(1, -1),
                    additional_model_output_stencil,
                    delta_positions,
                    False,
                    method=self.finite_difference_method,
                ).reshape(output.size, -1)
            )

        gradient_response = np.array(model_gradients_lst)

        return {"result": response, "gradient": gradient_response}
