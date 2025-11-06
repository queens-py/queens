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
"""Adjoint model."""

import logging

from queens.models.simulation import Simulation
from queens.utils.config_directories import current_job_directory
from queens.utils.io import write_to_csv
from queens.utils.logger_settings import log_init_args

_logger = logging.getLogger(__name__)


class Adjoint(Simulation):
    """Adjoint model.

    Attributes:
        adjoint_file (str): Name of the adjoint file that contains the evaluated derivative of the
                            functional w.r.t. to the simulation output.
        gradient_driver (Driver): Driver object for the adjoint simulation run.
    """

    @log_init_args
    def __init__(
        self,
        scheduler,
        driver,
        gradient_driver,
        adjoint_file="adjoint_grad_objective.csv",
    ):
        """Initialize model.

        Args:
            scheduler (Scheduler): Scheduler for the simulations
            driver (Driver): Driver for the simulations
            gradient_driver (Driver): Driver object for the adjoint simulation run.
            adjoint_file (str): Name of the adjoint file that contains the evaluated derivative of
                                the functional w.r.t. to the simulation output.
        """
        super().__init__(scheduler=scheduler, driver=driver)
        self.gradient_driver = gradient_driver
        self.adjoint_file = adjoint_file

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
        num_samples = samples.shape[0]
        # get last job_ids
        last_job_ids = [self.scheduler.next_job_id - num_samples + i for i in range(num_samples)]
        experiment_dir = self.scheduler.experiment_dir

        # write adjoint data for each sample to adjoint files in old job directories
        for job_id, grad_objective in zip(last_job_ids, upstream_gradient):
            job_dir = current_job_directory(experiment_dir, job_id)
            adjoint_file_path = job_dir.joinpath(self.adjoint_file)
            write_to_csv(adjoint_file_path, grad_objective.reshape(1, -1))

        # evaluate the adjoint model
        gradient = self.scheduler.evaluate(
            samples, function=self.gradient_driver, job_ids=last_job_ids
        )["result"]
        return gradient
