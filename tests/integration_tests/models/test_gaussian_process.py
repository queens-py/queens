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
"""Integration test for the GPflow based GP model."""

import numpy as np
import pytest

from queens.distributions.uniform import Uniform
from queens.drivers.function import Function
from queens.iterators.monte_carlo import MonteCarlo
from queens.main import run_iterator
from queens.models.simulation import Simulation
from queens.models.surrogates.gaussian_process import GaussianProcess
from queens.parameters.parameters import Parameters
from queens.schedulers.pool import Pool
from queens.utils.io import load_result


def test_gaussian_process_branin78(
    expected_mean,
    expected_variance,
    expected_posterior_samples,
    global_settings,
):
    """Test case for GPflow based GP model."""
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
        num_samples=20,
        model=model,
        parameters=parameters,
        global_settings=global_settings,
    )
    model = GaussianProcess(
        train_likelihood_variance=False,
        number_restarts=5,
        number_training_iterations=1000,
        number_posterior_samples=3,
        seed_posterior_samples=42,
        dimension_lengthscales=2,
        plotting_options={
            "plot_booleans": [False, False],
            "plotting_dir": "dummy",
            "plot_names": ["1D", "2D"],
            "save_bool": [False, False],
        },
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

    np.testing.assert_array_almost_equal(
        results["raw_output_data"]["result"], expected_mean, decimal=3
    )
    np.testing.assert_array_almost_equal(
        results["raw_output_data"]["variance"], expected_variance, decimal=2
    )
    np.testing.assert_array_almost_equal(
        results["raw_output_data"]["post_samples"], expected_posterior_samples, decimal=2
    )


@pytest.fixture(name="expected_mean")
def fixture_expected_mean():
    """Expected mean values."""
    expected_mean = np.array(
        [
            [127.97233506],
            [39.73551321],
            [47.00641347],
            [28.88934819],
            [22.40199886],
            [150.69211917],
            [104.25630329],
            [92.22700928],
            [50.69060622],
            [22.18886383],
        ]
    )
    return expected_mean


@pytest.fixture(name="expected_variance")
def fixture_expected_variance():
    """Expected variance values."""
    expected_variance = np.array(
        [
            [788.8004288],
            [1.8365012],
            [2.25043994],
            [4.24878946],
            [1.97026586],
            [174.50881662],
            [14.06623098],
            [8.34025715],
            [0.95922611],
            [0.33420735],
        ]
    )
    return expected_variance


@pytest.fixture(name="expected_posterior_samples")
def fixture_expected_posterior_samples():
    """Expected posterior samples."""
    expected_posterior_samples = np.array(
        [
            [1.890, 0.136, 2.294, -0.231, 0.461, 4.178, 0.695, 0.284, -0.265, 1.990],
            [0.375, -0.206, 1.942, 3.694, -0.037, 2.271, -0.222, 0.584, 3.734, 0.643],
            [0.444, 3.226, 0.662, 0.309, -0.227, 1.903, 2.412, -0.030, 2.387, -0.266],
        ]
    )
    return expected_posterior_samples.transpose()
