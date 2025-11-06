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
"""Integration tests for various Gaussian Process approximation methods."""

import numpy as np
import pytest

from example_simulator_functions.park91a import park91a_hifi_on_grid
from queens.distributions.uniform import Uniform
from queens.drivers.function import Function
from queens.iterators.adaptive_sampling import AdaptiveSampling
from queens.iterators.sequential_monte_carlo_chopin import SequentialMonteCarloChopin
from queens.main import run_iterator
from queens.models.likelihoods.gaussian import Gaussian
from queens.models.logpdf_gaussian_process import LogpdfGaussianProcess
from queens.models.simulation import Simulation
from queens.parameters.parameters import Parameters
from queens.schedulers.pool import Pool
from queens.utils.io import load_result


@pytest.fixture(
    name="approx_type",
    params=[
        "GPMAP-I",
        "CGPMAP-II",
        "CFBGP",
    ],
)
def fixture_approx_type(request):
    """Different approximation types."""
    return request.param


@pytest.fixture(name="parameters")
def fixture_parameters():
    """Two uniformly distributed parameters."""
    x1 = Uniform(lower_bound=0, upper_bound=1)
    x2 = Uniform(lower_bound=0, upper_bound=1)
    parameters = Parameters(x1=x1, x2=x2)
    return parameters


@pytest.fixture(name="likelihood_model")
def fixture_likelihood_model(parameters, global_settings):
    """A Gaussian likelihood model."""
    np.random.seed(42)
    driver = Function(parameters=parameters, function=park91a_hifi_on_grid)
    scheduler = Pool(experiment_name=global_settings.experiment_name)
    forward_model = Simulation(scheduler=scheduler, driver=driver)

    y_obs = park91a_hifi_on_grid(x1=0.3, x2=0.7)
    noise_var = 1e-4
    y_obs += np.random.randn(y_obs.size) * noise_var ** (1 / 2)

    likelihood_model = Gaussian(
        forward_model=forward_model,
        noise_type="fixed_variance",
        noise_value=noise_var,
        y_obs=y_obs,
    )
    return likelihood_model


@pytest.fixture(name="expected_mean")
def fixture_expected_mean():
    """Expected mean values."""
    expected_mean = {
        "GPMAP-I": [0.301425, 0.653193],
        "CGPMAP-II": [0.301557, 0.64682],
        "CFBGP": [0.301444, 0.653865],
    }
    return expected_mean


@pytest.fixture(name="expected_std")
def fixture_expected_std():
    """Expected standard deviation values."""
    expected_std = {
        "GPMAP-I": [0.00086233, 0.02220657],
        "CGPMAP-II": [0.00087329, 0.02323444],
        "CFBGP": [0.001561, 0.028399],
    }
    return expected_std


def test_logpdf_gaussian_process_park91a(
    approx_type,
    likelihood_model,
    parameters,
    expected_mean,
    expected_std,
    global_settings,
):
    """Test for constrained GP with IP park."""
    num_steps = 4
    num_new_samples = 4
    quantile = 0.90
    seed = 41
    np.random.seed(seed)
    initial_train_samples = parameters.draw_samples(int(num_new_samples * 2))

    logpdf_gp_model = LogpdfGaussianProcess(
        approx_type=approx_type,
        num_hyper=10,
        num_optimizations=3,
        hmc_burn_in=100,
        hmc_steps=100,
        prior_rate=[1.0e-1, 10.0, 1.0e8],
        prior_gp_mean=-1.0,
        quantile=quantile,
        jitter=1.0e-16,
    )

    solving_iterator = SequentialMonteCarloChopin(
        model=logpdf_gp_model,
        parameters=parameters,
        global_settings=global_settings,
        seed=42,
        waste_free=True,
        feynman_kac_model="adaptive_tempering",
        max_feval=1_000_000_000,
        num_particles=3000,
        num_rejuvenation_steps=30,
        resampling_method="residual",
        resampling_threshold=0.5,
        result_description={},
    )

    adaptive_sampling_iterator = AdaptiveSampling(
        model=logpdf_gp_model,
        parameters=parameters,
        global_settings=global_settings,
        likelihood_model=likelihood_model,
        initial_train_samples=initial_train_samples,
        solving_iterator=solving_iterator,
        num_new_samples=num_new_samples,
        num_steps=num_steps,
    )

    run_iterator(adaptive_sampling_iterator, global_settings)

    # Load results
    results = load_result(global_settings.result_file(".pickle"))

    particles = results["particles"][-1]
    weights = results["weights"][-1]

    mean = np.average(particles, weights=weights, axis=0)
    std = np.average((particles - mean) ** 2, weights=weights, axis=0) ** (1 / 2)

    np.testing.assert_allclose(mean, expected_mean[approx_type], rtol=5e-2)
    np.testing.assert_allclose(std, expected_std[approx_type], rtol=5e-1)
