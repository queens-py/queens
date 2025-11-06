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
"""Integration test for the GPflow based SVGP model."""

import numpy as np
import pytest

from queens.distributions.uniform import Uniform
from queens.drivers.function import Function
from queens.iterators.monte_carlo import MonteCarlo
from queens.main import run_iterator
from queens.models.simulation import Simulation
from queens.models.surrogates.variational_gaussian_process import VariationalGaussianProcess
from queens.parameters.parameters import Parameters
from queens.schedulers.pool import Pool
from queens.utils.io import load_result
from test_utils.integration_tests import assert_monte_carlo_iterator_results


@pytest.mark.max_time_for_test(60)
def test_variational_gaussian_process_branin78(expected_mean, expected_var, global_settings):
    """Test case for GPflow based SVGP model."""
    # Parameters
    x1 = Uniform(lower_bound=-5, upper_bound=10)
    x2 = Uniform(lower_bound=0, upper_bound=15)
    parameters = Parameters(x1=x1, x2=x2)

    # Setup iterator
    driver = Function(parameters=parameters, function="branin78_hifi")
    scheduler = Pool(experiment_name=global_settings.experiment_name)
    model = Simulation(scheduler=scheduler, driver=driver)
    training_iterator = MonteCarlo(
        seed=42,
        num_samples=100,
        model=model,
        parameters=parameters,
        global_settings=global_settings,
    )
    model = VariationalGaussianProcess(
        plotting_options={
            "plot_booleans": [False, False],
            "plotting_dir": "dummy",
            "plot_names": ["1D", "2D"],
            "save_bool": [False, False],
        },
        train_likelihood_variance=False,
        seed=41,
        mini_batch_size=50,
        number_inducing_points=50,
        train_inducing_points_location=True,
        number_training_iterations=10000,
        dimension_lengthscales=2,
        training_iterator=training_iterator,
    )
    iterator = MonteCarlo(
        seed=44,
        num_samples=10,
        result_description={
            "write_results": True,
            "plot_results": False,
            "bayesian": False,
            "num_support_points": 10,
            "estimate_all": False,
        },
        model=model,
        parameters=parameters,
        global_settings=global_settings,
    )

    # Actual analysis
    run_iterator(iterator, global_settings=global_settings)

    # Load results
    results = load_result(global_settings.result_file(".pickle"))

    assert_monte_carlo_iterator_results(results, expected_mean, expected_var)


@pytest.fixture(name="expected_mean")
def fixture_expected_mean():
    """Expected mean values."""
    mean = np.array(
        [
            [181.62057979],
            [37.95455295],
            [47.86422341],
            [32.47391656],
            [23.99246991],
            [167.32578661],
            [106.07427664],
            [92.93591941],
            [50.72976800],
            [22.10505115],
        ]
    )
    return mean


@pytest.fixture(name="expected_var")
def fixture_expected_var():
    """Expected variance values."""
    var = np.array(
        [
            [4.62061],
            [1.38456],
            [0.96146],
            [0.20286],
            [0.34231],
            [1.03465],
            [0.24111],
            [0.40275],
            [0.22169],
            [0.58071],
        ]
    )
    return var
