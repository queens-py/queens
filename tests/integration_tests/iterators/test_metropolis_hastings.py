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
"""Integration tests for the Metropolis Hastings iterator."""

import numpy as np
import pandas as pd
import pytest
from mock import patch

from example_simulator_functions.gaussian_logpdf import GAUSSIAN_2D, gaussian_2d_logpdf
from queens.distributions.normal import Normal
from queens.drivers.function import Function
from queens.iterators.metropolis_hastings import MetropolisHastings
from queens.iterators.sequential_monte_carlo import SequentialMonteCarlo
from queens.main import run_iterator
from queens.models.likelihoods.gaussian import Gaussian
from queens.models.simulation import Simulation
from queens.parameters.parameters import Parameters
from queens.schedulers.pool import Pool
from queens.utils.experimental_data_reader import ExperimentalDataReader
from queens.utils.io import load_result


def test_metropolis_hastings_univariate_gaussian(
    tmp_path,
    target_density_gaussian_1d,
    _create_experimental_data_gaussian_1d,
    global_settings,
):
    """Test case for Metropolis Hastings iterator.

    This test samples from a univariate Gaussian distribution.
    """
    # Parameters
    x = Normal(mean=2, covariance=1)
    parameters = Parameters(x=x)

    # Setup iterator
    experimental_data_reader = ExperimentalDataReader(
        file_name_identifier="*.csv",
        csv_data_base_dir=tmp_path,
        output_label="y_obs",
    )
    proposal_distribution = Normal(mean=0.0, covariance=1.0)
    driver = Function(parameters=parameters, function="patch_for_likelihood")
    scheduler = Pool(experiment_name=global_settings.experiment_name)
    forward_model = Simulation(scheduler=scheduler, driver=driver)
    model = Gaussian(
        noise_type="fixed_variance",
        noise_value=1.0,
        nugget_noise_variance=1e-05,
        experimental_data_reader=experimental_data_reader,
        forward_model=forward_model,
    )
    iterator = MetropolisHastings(
        seed=42,
        num_samples=10,
        num_burn_in=5,
        scale_covariance=1.0,
        result_description={"write_results": True, "plot_results": False, "cov": True},
        proposal_distribution=proposal_distribution,
        model=model,
        parameters=parameters,
        global_settings=global_settings,
    )

    # Actual analysis
    with patch.object(MetropolisHastings, "eval_log_likelihood", target_density_gaussian_1d):
        run_iterator(iterator, global_settings=global_settings)

    # Load results
    results = load_result(global_settings.result_file(".pickle"))

    # note that the analytical solution would be:
    # posterior mean: [1.]
    # posterior var: [0.5]
    # posterior std: [0.70710678]
    # however, we only have a very inaccurate approximation here:
    assert results["mean"] == pytest.approx(1.046641592648936)
    assert results["var"] == pytest.approx(0.3190199514534667)


def test_metropolis_hastings_multivariate_gaussian(
    tmp_path,
    target_density_gaussian_2d,
    _create_experimental_data_gaussian_2d,
    global_settings,
):
    """Test case for Metropolis Hastings iterator.

    This test samples from a multivariate Gaussian distribution.
    """
    # Parameters
    x1 = Normal(mean=2.0, covariance=1.0)
    x2 = Normal(mean=-2.0, covariance=0.01)
    parameters = Parameters(x1=x1, x2=x2)

    # Setup iterator
    experimental_data_reader = ExperimentalDataReader(
        file_name_identifier="*.csv",
        csv_data_base_dir=tmp_path,
        output_label="y_obs",
    )
    proposal_distribution = Normal(mean=[0.0, 0.0], covariance=[[1.0, 0.0], [0.0, 0.1]])
    driver = Function(parameters=parameters, function="patch_for_likelihood")
    scheduler = Pool(experiment_name=global_settings.experiment_name)
    forward_model = Simulation(scheduler=scheduler, driver=driver)
    model = Gaussian(
        noise_type="fixed_variance",
        noise_value=1.0,
        experimental_data_reader=experimental_data_reader,
        forward_model=forward_model,
    )
    iterator = MetropolisHastings(
        seed=42,
        num_samples=10,
        num_burn_in=5,
        scale_covariance=1.0,
        result_description={"write_results": True, "plot_results": False, "cov": True},
        proposal_distribution=proposal_distribution,
        model=model,
        parameters=parameters,
        global_settings=global_settings,
    )

    # Actual analysis
    with patch.object(MetropolisHastings, "eval_log_likelihood", target_density_gaussian_2d):
        run_iterator(iterator, global_settings=global_settings)

    # Load results
    results = load_result(global_settings.result_file(".pickle"))

    # note that the analytical solution would be:
    # posterior mean: [0.29378531 -1.97175141]
    # posterior cov: [[0.42937853 0.00282486] [0.00282486 0.00988701]]
    # however, we only have a very inaccurate approximation here:

    np.testing.assert_allclose(
        results["mean"], np.array([[0.7240107551260684, -2.045891088599629]])
    )
    np.testing.assert_allclose(
        results["cov"],
        np.array(
            [
                [
                    [0.30698538649168755, -0.027059991075557278],
                    [-0.027059991075557278, 0.004016365725389411],
                ]
            ]
        ),
    )


def test_metropolis_hastings_multivariate_gaussian_multiple_chains(
    tmp_path,
    target_density_gaussian_2d,
    _create_experimental_data_gaussian_2d,
    global_settings,
):
    """Test case for Metropolis Hastings iterator.

    This test samples from a multivariate Gaussian distribution using
    three chains.
    """
    # Parameters
    x1 = Normal(mean=2.0, covariance=1.0)
    x2 = Normal(mean=-2.0, covariance=0.01)
    parameters = Parameters(x1=x1, x2=x2)

    # Setup iterator
    experimental_data_reader = ExperimentalDataReader(
        file_name_identifier="*.csv",
        csv_data_base_dir=tmp_path,
        output_label="y_obs",
    )
    proposal_distribution = Normal(mean=[0.0, 0.0], covariance=[[1.0, 0.0], [0.0, 0.1]])
    driver = Function(parameters=parameters, function="patch_for_likelihood")
    scheduler = Pool(experiment_name=global_settings.experiment_name)
    forward_model = Simulation(scheduler=scheduler, driver=driver)
    model = Gaussian(
        noise_type="fixed_variance",
        noise_value=1.0,
        nugget_noise_variance=1e-05,
        experimental_data_reader=experimental_data_reader,
        forward_model=forward_model,
    )
    iterator = MetropolisHastings(
        seed=42,
        num_chains=3,
        num_samples=10,
        num_burn_in=5,
        scale_covariance=1.0,
        result_description={"write_results": True, "plot_results": False, "cov": True},
        proposal_distribution=proposal_distribution,
        model=model,
        parameters=parameters,
        global_settings=global_settings,
    )

    # Actual analysis
    # mock methods related to likelihood
    with patch.object(SequentialMonteCarlo, "eval_log_likelihood", target_density_gaussian_2d):
        with patch.object(MetropolisHastings, "eval_log_likelihood", target_density_gaussian_2d):
            run_iterator(iterator, global_settings=global_settings)

    # Load results
    results = load_result(global_settings.result_file(".pickle"))

    # note that the analytical solution would be:
    # posterior mean: [0.29378531 -1.97175141]
    # posterior cov: [[0.42937853 0.00282486] [0.00282486 0.00988701]]
    # however, we only have a very inaccurate approximation here:

    np.testing.assert_allclose(
        results["mean"],
        np.array(
            [
                [1.9538477050387937, -1.980155948698723],
                [-0.024456540006756778, -1.9558862932202299],
                [0.8620026644863327, -1.8385635263327393],
            ]
        ),
    )
    np.testing.assert_allclose(
        results["cov"],
        np.array(
            [
                [
                    [0.15127359388133552, 0.07282531084034029],
                    [0.07282531084034029, 0.05171405742642703],
                ],
                [
                    [0.17850797646369507, -0.012342979562824052],
                    [-0.012342979562824052, 0.0023510303586270057],
                ],
                [
                    [0.0019646760257596243, 0.002417903725921208],
                    [0.002417903725921208, 0.002975685737073754],
                ],
            ]
        ),
    )


@pytest.fixture(name="_create_experimental_data_gaussian_2d")
def fixture_create_experimental_data_gaussian_2d(tmp_path):
    """Create a csv file with experimental data from a 2D Gaussian."""
    # generate 10 samples from the same gaussian
    samples = GAUSSIAN_2D.draw(10)
    pdf = gaussian_2d_logpdf(samples)

    pdf = np.array(pdf)

    # write the data to a csv file in tmp_path
    data_dict = {"y_obs": pdf}
    experimental_data_path = tmp_path / "experimental_data.csv"
    df = pd.DataFrame.from_dict(data_dict)
    df.to_csv(experimental_data_path, index=False)
