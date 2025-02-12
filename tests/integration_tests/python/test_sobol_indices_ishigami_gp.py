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
"""Integration test for Sobol indices estimation with Ishigami function.

This test uses a Gaussian process surrogate.
"""

import numpy as np

from queens.distributions.uniform import Uniform
from queens.drivers.function import Function
from queens.iterators.lhs_iterator import LHSIterator
from queens.iterators.sobol_index_iterator import SobolIndexIterator
from queens.main import run_iterator
from queens.models.simulation_model import SimulationModel
from queens.models.surrogate_models.gp_approximation_gpflow import GPFlowRegressionModel
from queens.parameters.parameters import Parameters
from queens.schedulers.pool_scheduler import PoolScheduler
from queens.utils.io_utils import load_result


def test_sobol_indices_ishigami_gp(global_settings):
    """Test Sobol indices estimation with Gaussian process surrogate."""
    # Parameters
    x1 = Uniform(lower_bound=-3.14159265359, upper_bound=3.14159265359)
    x2 = Uniform(lower_bound=-3.14159265359, upper_bound=3.14159265359)
    x3 = Uniform(lower_bound=-3.14159265359, upper_bound=3.14159265359)
    parameters = Parameters(x1=x1, x2=x2, x3=x3)

    # Setup iterator
    driver = Function(parameters=parameters, function="ishigami90")
    scheduler = PoolScheduler(experiment_name=global_settings.experiment_name)
    simulation_model = SimulationModel(scheduler=scheduler, driver=driver)
    training_iterator = LHSIterator(
        seed=42,
        num_samples=50,
        num_iterations=10,
        result_description={"write_results": True, "plot_results": False},
        model=simulation_model,
        parameters=parameters,
        global_settings=global_settings,
    )
    gpflow_regression_model = GPFlowRegressionModel(
        number_restarts=10,
        number_training_iterations=1000,
        dimension_lengthscales=3,
        training_iterator=training_iterator,
    )
    iterator = SobolIndexIterator(
        seed=42,
        calc_second_order=False,
        num_samples=128,
        confidence_level=0.95,
        num_bootstrap_samples=1000,
        result_description={"write_results": True, "plot_results": True},
        model=gpflow_regression_model,
        parameters=parameters,
        global_settings=global_settings,
        skip_values=1024,
    )

    # Actual analysis
    run_iterator(iterator, global_settings=global_settings)

    # Load results
    results = load_result(global_settings.result_file(".pickle"))

    expected_result_s1 = np.array([0.37365542, 0.49936914, -0.00039217])
    expected_result_s1_conf = np.array([0.14969221, 0.18936135, 0.0280309])

    np.testing.assert_allclose(results["sensitivity_indices"]["S1"], expected_result_s1, atol=1e-05)
    np.testing.assert_allclose(
        results["sensitivity_indices"]["S1_conf"], expected_result_s1_conf, atol=1e-05
    )
