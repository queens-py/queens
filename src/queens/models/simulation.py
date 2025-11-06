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
"""Simulation model class."""

import numpy as np

from queens.models._model import Model
from queens.utils.logger_settings import log_init_args


class Simulation(Model):
    """Simulation model class.

    Attributes:
        scheduler (Scheduler): Scheduler for the simulations
        driver (Driver): Driver for the simulations
    """

    @log_init_args
    def __init__(self, scheduler, driver):
        """Initialize simulation model.

        Args:
            scheduler (Scheduler): Scheduler for the simulations
            driver (Driver): Driver for the simulations
        """
        super().__init__()
        self.scheduler = scheduler
        self.driver = driver
        self.scheduler.copy_files_to_experiment_dir(self.driver.files_to_copy)

    def _evaluate(self, samples):
        """Evaluate model with current set of input samples.

        Args:
            samples (np.ndarray): Input samples

        Returns:
            response (dict): Response of the underlying model at input samples
        """
        self.response = self.scheduler.evaluate(samples, self.driver)
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
        if self.response.get("gradient") is None:
            raise ValueError("Gradient information not available.")
        # The shape of the returned gradient is weird
        response_gradient = np.swapaxes(self.response["gradient"], 1, 2)
        gradient = np.sum(upstream_gradient[:, :, np.newaxis] * response_gradient, axis=1)
        return gradient
