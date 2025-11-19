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

from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import numpy as np

from queens.drivers._driver import Driver
from queens.models._model import Model
from queens.parameters import Parameters
from queens.schedulers._scheduler import Scheduler, SchedulerCallableSignature
from queens.utils.logger_settings import log_init_args
from queens.utils.metadata import SimulationMetadata


class Simulation(Model):
    """Simulation model class for parallel evaluations."""

    @log_init_args
    def __init__(self, scheduler: Scheduler, function: SchedulerCallableSignature):
        """Initialize simulation model.

        Args:
            scheduler (Scheduler): Scheduler for the simulations
            driver (Driver): Driver for the simulations
        """
        super().__init__()
        self.scheduler = scheduler
        self.function = function

    @classmethod
    def from_simulation_code(
        cls,
        scheduler: Scheduler,
        parameters: Parameters,
        driver: Driver,
        data_processor: Callable[[Path], dict] | None,
    ):

        function = cls.scheduler_function_from_simulation_code(parameters, driver, data_processor)

        # Copy common files
        scheduler.copy_files_to_experiment_dir(driver.files_to_copy)

        return cls(scheduler, function)

    @staticmethod
    def scheduler_function_from_simulation_code(
        parameters: Parameters,
        driver: SchedulerCallableSignature,
    ):
        def function(
            sample: np.ndarray,
            job_id: int,
            job_dir: Path,
            num_procs: int,
            experiment_dir: Path,
            experiment_name: str,
        ):
            metadata = SimulationMetadata(job_id, sample, job_dir, file_name="job_metadata")

            # Transform the array into a dictionary (including random fields)
            with metadata.time_code("create_sample_dict"):
                sample_dict = parameters.sample_as_dict(sample)

            # Make job dir
            job_dir.mkdir(exist_ok=True, parents=True)

            # Run the simulations
            with metadata.time_code("run_driver"):
                data = driver(
                    sample_dict, job_id, job_dir, num_procs, experiment_dir, experiment_name
                )

            return data

        return function

    def _evaluate(self, samples: Iterable) -> dict:
        """Evaluate model with current set of input samples.

        Args:
            samples (np.ndarray): Input samples

        Returns:
            response (dict): Response of the underlying model at input samples
        """
        self.response = self.create_result_dict_from_scheduler_output(
            self.scheduler.evaluate(samples, self.driver)
        )
        return self.response

    @staticmethod
    def create_result_dict_from_scheduler_output(scheduler_response: list) -> dict[str, np.ndarray]:
        """Create a dictionary from scheduler response.

        Args:
            scheduler_response (list): Results from scheduler.

        Returns:
            dict: results and eventually graident values.
        """
        results = {}

        for response in scheduler_response:

            for result_name, result_value in response.items():
                if result_name not in results:
                    results[result_name] = []

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
