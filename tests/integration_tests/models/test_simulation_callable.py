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
"""Integration test for the simulation model with a plain callable."""

import numpy as np

from queens.distributions.uniform import Uniform
from queens.drivers.function import Function
from queens.models.simulation import Simulation
from queens.parameters.parameters import Parameters
from queens.schedulers.pool import Pool


def rosenbrock(x1, x2):
    """Rosenbrock function."""
    return (1 - x1) ** 2 + 100 * (x2 - x1**2) ** 2


def rosenbrock_job(sample, job_id, num_procs, experiment_dir, experiment_name):
    """Rosenbrock function with the signature required by the schedulers."""
    # pylint: disable=unused-argument
    return {"result": np.atleast_1d(rosenbrock(*sample))}


def test_simulation_with_callable(global_settings, tmp_path):
    """Test that a plain callable matches the function driver."""
    parameters = Parameters(x1=Uniform(-2, 2), x2=Uniform(-2, 2))
    samples = np.array([[0.3, 1.7], [-1.2, 0.4], [1.9, -0.8]])

    scheduler = Pool(experiment_name=global_settings.experiment_name, experiment_base_dir=tmp_path)
    model_with_driver = Simulation(
        scheduler=scheduler, driver=Function(parameters=parameters, function=rosenbrock)
    )
    model_with_callable = Simulation(scheduler=scheduler, driver=rosenbrock_job)

    result_driver = model_with_driver.evaluate(samples)["result"]
    result_callable = model_with_callable.evaluate(samples)["result"]

    np.testing.assert_allclose(result_callable, result_driver)
    np.testing.assert_allclose(result_callable[:, 0], [rosenbrock(*s) for s in samples])
