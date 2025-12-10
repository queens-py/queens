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

from queens.drivers._driver import Driver
from queens.models._model import Model
from queens.schedulers._scheduler import Scheduler, SchedulerCallableSignature
from queens.utils.logger_settings import log_init_args


class Simulation(Model):
    """Simulation model class."""

    @log_init_args
    def __init__(self, scheduler: Scheduler, driver: Driver | SchedulerCallableSignature):
        """Initialize simulation model.

        Args:
            scheduler: Scheduler for the simulations
            driver: Driver for the simulations
        """
        super().__init__()
        self.scheduler = scheduler
        self.function: SchedulerCallableSignature
        if isinstance(driver, Driver):
            self.function = driver.run_from_parameters
            self.scheduler.copy_files_to_experiment_dir(driver.files_to_copy)
        else:
            self.function = driver

    def _evaluate(self, samples: np.ndarray) -> dict:
        """Evaluate model with current set of input samples.

        Args:
            samples (np.ndarray): Input samples

        Returns:
            response (dict): Response of the underlying model at input samples
        """
        self.response = self.create_result_dict_from_scheduler_output(
            self.scheduler.evaluate(samples, self.function)
        )
        return self.response

    @staticmethod
    def create_result_dict_from_scheduler_output(scheduler_response: list) -> dict[str, np.ndarray]:
        """Create a dictionary from scheduler response.

        Args:
            scheduler_response: Results from scheduler.

        Returns:
            Results
        """
        known_keys = set(scheduler_response[0].keys())
        results = {key: [] for key in known_keys}

        for response in scheduler_response:

            if set(response.keys()).symmetric_difference(known_keys):
                raise KeyError("All scheduler responses need to provide the same result names!")

            for result_name, result_value in response.items():

                if result_name == "result":
                    # We should remove this squeeze!
                    # It is only introduced for consistency with old test.
                    results[result_name].append(np.atleast_1d(np.array(result_value).squeeze()))
                else:
                    results[result_name].append(result_value)

        return {
            result_name: np.array(result_value) for result_name, result_value in results.items()
        }

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
