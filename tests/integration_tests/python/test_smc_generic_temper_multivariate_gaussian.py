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
"""Integration test for the SMC iterator.

This test uses a multivariate Gaussian and generic tempering.
"""

import numpy as np
import pandas as pd
import pytest
from mock import patch

from queens.distributions.normal import Normal
from queens.drivers.function import Function
from queens.example_simulator_functions.gaussian_logpdf import GAUSSIAN_4D, gaussian_4d_logpdf
from queens.iterators.metropolis_hastings_iterator import MetropolisHastingsIterator
from queens.iterators.sequential_monte_carlo_iterator import SequentialMonteCarloIterator
from queens.main import run_iterator
from queens.models.likelihood_models.gaussian_likelihood import GaussianLikelihood
from queens.models.simulation_model import SimulationModel
from queens.parameters.parameters import Parameters
from queens.schedulers.pool_scheduler import PoolScheduler
from queens.utils.experimental_data_reader import ExperimentalDataReader
from queens.utils.io_utils import load_result


def test_smc_generic_temper_multivariate_gaussian(
    tmp_path, _create_experimental_data, global_settings
):
    """Test SMC with a multivariate Gaussian and generic tempering."""
    # Parameters
    x1 = Normal(mean=1.0, covariance=5.0)
    x2 = Normal(mean=3.0, covariance=5.0)
    x3 = Normal(mean=-3.0, covariance=5.0)
    x4 = Normal(mean=1.0, covariance=5.0)
    parameters = Parameters(x1=x1, x2=x2, x3=x3, x4=x4)

    # Setup iterator
    experimental_data_reader = ExperimentalDataReader(
        file_name_identifier="*.csv",
        csv_data_base_dir=tmp_path,
        output_label="y_obs",
    )
    mcmc_proposal_distribution = Normal(
        mean=[0.0, 0.0, 0.0, 0.0],
        covariance=[
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
    )
    driver = Function(parameters=parameters, function="patch_for_likelihood")
    scheduler = PoolScheduler(experiment_name=global_settings.experiment_name)
    forward_model = SimulationModel(scheduler=scheduler, driver=driver)
    model = GaussianLikelihood(
        noise_type="fixed_variance",
        noise_value=1.0,
        nugget_noise_variance=1e-05,
        experimental_data_reader=experimental_data_reader,
        forward_model=forward_model,
    )
    iterator = SequentialMonteCarloIterator(
        seed=42,
        num_particles=200,
        temper_type="generic",
        plot_trace_every=0,
        num_rejuvenation_steps=20,
        result_description={"write_results": True, "plot_results": False, "cov": True},
        mcmc_proposal_distribution=mcmc_proposal_distribution,
        model=model,
        parameters=parameters,
        global_settings=global_settings,
    )

    # Actual analysis
    with patch.object(SequentialMonteCarloIterator, "eval_log_likelihood", target_density):
        with patch.object(MetropolisHastingsIterator, "eval_log_likelihood", target_density):
            run_iterator(iterator, global_settings=global_settings)

    # Load results
    results = load_result(global_settings.result_file(".pickle"))

    # note that the analytical solution can be found in multivariate_gaussian_4D_logpdf
    # we only have a very inaccurate approximation here:
    np.testing.assert_array_almost_equal(
        results["mean"], np.array([[0.884713, 2.903405, -3.112647, 1.56134]]), decimal=5
    )

    np.testing.assert_almost_equal(
        results["var"], np.array([[3.255066, 4.143380, 1.838545, 2.834356]]), decimal=5
    )

    np.testing.assert_almost_equal(
        results["cov"],
        np.array(
            [
                [
                    [3.255066, 1.781563, 0.313565, -0.090972],
                    [1.781563, 4.143380, 0.779616, 1.704881],
                    [0.313565, 0.779616, 1.838545, 0.630236],
                    [-0.090972, 1.704881, 0.630236, 2.834356],
                ]
            ]
        ),
        decimal=5,
    )


def target_density(self, samples):  # pylint: disable=unused-argument
    """Returns the log likelihood of samples with 4D Gaussian distribution."""
    samples = np.atleast_2d(samples)
    log_likelihood = gaussian_4d_logpdf(samples)

    return log_likelihood


@pytest.fixture(name="_create_experimental_data")
def fixture_create_experimental_data(tmp_path):
    """Create a csv file with experimental data."""
    # generate 10 samples from the same gaussian
    samples = GAUSSIAN_4D.draw(10)
    pdf = gaussian_4d_logpdf(samples)

    # write the data to a csv file in tmp_path
    data_dict = {"y_obs": pdf}
    experimental_data_path = tmp_path / "experimental_data.csv"
    df = pd.DataFrame.from_dict(data_dict)
    df.to_csv(experimental_data_path, index=False)
