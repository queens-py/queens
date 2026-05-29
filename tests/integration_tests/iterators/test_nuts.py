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
"""Integration test for the NUTS Iterator."""

import numpy as np
import pandas as pd
import pytest
from mock import patch

from queens.distributions.normal import Normal
from queens.drivers.function import Function
from queens.iterators.nuts import NUTS
from queens.main import run_iterator
from queens.models.likelihoods.gaussian import Gaussian
from queens.models.simulation import Simulation
from queens.parameters.parameters import Parameters
from queens.schedulers.pool import Pool
from queens.utils.experimental_data_reader import ExperimentalDataReader
from queens.utils.io import load_result

SAMPLER_STAT_RTOL = 1e-5
SAMPLER_STAT_ATOL = 1e-8


def test_nuts_gaussian(
    tmp_path,
    target_density_gaussian_2d_with_grad,
    _create_experimental_data,
    global_settings,
):
    """Test NUTS sampling for a Gaussian-Gaussian Bayesian inference problem.

    The test samples from a two-dimensional posterior with Gaussian prior and Gaussian
    likelihood. Since both prior and likelihood are Gaussian, the posterior is Gaussian
    again. The prior is

        x ~ N(mu_0, Sigma_0),
        mu_0 = [-2, 2]^T,
        Sigma_0 = [[1, 0], [0, 1]].

    The likelihood is evaluated at the observed value y = [0, 0]^T with

        y | x ~ N(x, Sigma_L),
        Sigma_L = [[1, 1/2], [1/2, 1]].

    Therefore,

        Sigma_p = (Sigma_0^{-1} + Sigma_L^{-1})^{-1}
                = [[7/15, 2/15], [2/15, 7/15]],

        mu_p = Sigma_p (Sigma_0^{-1} mu_0 + Sigma_L^{-1} y)
             = Sigma_p mu_0
             = [-2/3, 2/3]^T.

    The converged Markov chain should therefore approximate

        E[x | y] = [-2/3, 2/3]^T,
        Var[x | y] = [7/15, 7/15],
        Std[x | y] = [sqrt(7/15), sqrt(7/15)].

    Note:
        This behaviour is achieved by patching the Gaussian likelihood model
        evaluation and gradient with ``target_density_gaussian_2d_with_grad``.
        Instead of evaluating against the experimental data, the likelihood is
        replaced by a fixed analytic Gaussian log-density corresponding to the
        target distribution described above.
    """
    # Parameters
    x1 = Normal(mean=[-2.0, 2.0], covariance=[[1.0, 0.0], [0.0, 1.0]])
    parameters = Parameters(x1=x1)

    # Setup iterator
    experimental_data_reader = ExperimentalDataReader(
        file_name_identifier="*.csv",
        csv_data_base_dir=tmp_path,
        output_label="y_obs",
    )
    driver = Function(parameters=parameters, function="patch_for_likelihood")
    scheduler = Pool(experiment_name=global_settings.experiment_name)
    forward_model = Simulation(scheduler=scheduler, driver=driver)
    model = Gaussian(
        noise_type="fixed_variance",
        noise_value=1.0,
        experimental_data_reader=experimental_data_reader,
        forward_model=forward_model,
    )
    iterator = NUTS(
        seed=42,
        num_samples=10,
        num_burn_in=2,
        num_chains=1,
        use_queens_prior=False,
        progressbar=False,
        result_description={"write_results": True, "plot_results": False, "cov": True},
        model=model,
        parameters=parameters,
        global_settings=global_settings,
    )

    # Actual analysis
    with patch.object(Gaussian, "evaluate_and_gradient", target_density_gaussian_2d_with_grad):
        run_iterator(iterator, global_settings=global_settings)

    # Load results
    results = load_result(global_settings.result_file(".pickle"))

    np.testing.assert_allclose(
        results["mean"].mean(axis=0),
        np.array([-1.0964337346677933, 0.9148542463484473]),
        rtol=SAMPLER_STAT_RTOL,
        atol=SAMPLER_STAT_ATOL,
    )
    np.testing.assert_allclose(
        results["var"].mean(axis=0),
        np.array([0.33594238408352364, 1.053294709724648]),
        rtol=SAMPLER_STAT_RTOL,
        atol=SAMPLER_STAT_ATOL,
    )


@pytest.fixture(name="_create_experimental_data")
def fixture_create_experimental_data(tmp_path):
    """Create a csv file with experimental data."""
    samples = np.array([0, 0]).flatten()

    # write the data to a csv file in tmp_path
    data_dict = {"y_obs": samples}
    experimental_data_path = tmp_path / "experimental_data.csv"
    df = pd.DataFrame.from_dict(data_dict)
    df.to_csv(experimental_data_path, index=False)
