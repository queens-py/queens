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
"""Integration test for the GPflow based heteroskedastic model."""

import numpy as np
import pytest

from queens.distributions.uniform import Uniform
from queens.drivers.function import Function
from queens.iterators.monte_carlo import MonteCarlo
from queens.main import run_iterator
from queens.models import HeteroskedasticGaussianProcess
from queens.models.simulation import Simulation
from queens.parameters.parameters import Parameters
from queens.schedulers.pool import Pool
from queens.utils.io import load_result


@pytest.mark.max_time_for_test(30)
def test_heteroskedastic_gaussian_process_branin78(expected_mean, expected_var, global_settings):
    """Test case for GPflow based heteroskedastic model."""
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
    gp_model = HeteroskedasticGaussianProcess(
        eval_fit=None,
        error_measures=[
            "sum_squared",
            "mean_squared",
            "root_mean_squared",
            "sum_abs",
            "mean_abs",
            "abs_max",
        ],
        num_posterior_samples=None,
        num_inducing_points=30,
        num_epochs=100,
        adams_training_rate=0.1,
        random_seed=1,
        num_samples_stats=1000,
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
        model=gp_model,
        parameters=parameters,
        global_settings=global_settings,
    )

    # Actual analysis
    run_iterator(iterator, global_settings=global_settings)

    # Load results
    results = load_result(global_settings.result_file(".pickle"))

    np.testing.assert_array_almost_equal(
        results["raw_output_data"]["result"], expected_mean, decimal=2
    )
    np.testing.assert_array_almost_equal(
        results["raw_output_data"]["variance"], expected_var, decimal=2
    )


@pytest.fixture(name="expected_mean")
def fixture_expected_mean():
    """Expected mean values."""
    mean = np.array(
        [
            [
                3.111393972224277,
                4.871511970865743,
                9.490107236480078,
                4.102804715168645,
                6.834075486918755,
                3.520856969539679,
                4.1233575884069715,
                5.168085181585406,
                9.71670249292498,
                8.305574341146198,
            ]
        ]
    ).T
    return mean


@pytest.fixture(name="expected_var")
def fixture_expected_var():
    """Expected variance values."""
    var = np.array(
        [
            [
                6069.028670706884,
                9751.90259532228,
                1016.0473220348358,
                1574.7038740682183,
                894.769154573094,
                7084.631563266536,
                4062.370531720105,
                6767.033077142452,
                2350.3605807973404,
                232.3100481516621,
            ]
        ]
    ).T
    return var
