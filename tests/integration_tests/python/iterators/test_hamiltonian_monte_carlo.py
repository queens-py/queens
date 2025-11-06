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
"""Integration test for the HMC iterator."""

import numpy as np
import pytest
from mock import patch

from queens.distributions.normal import Normal
from queens.drivers.function import Function
from queens.iterators.hamiltonian_monte_carlo import HamiltonianMonteCarlo
from queens.main import run_iterator
from queens.models.likelihoods.gaussian import Gaussian
from queens.models.simulation import Simulation
from queens.parameters.parameters import Parameters
from queens.schedulers.pool import Pool
from queens.utils.experimental_data_reader import ExperimentalDataReader
from queens.utils.io import load_result


def test_hamiltonian_monte_carlo_gaussian(
    tmp_path,
    target_density_gaussian_2d_with_grad,
    _create_experimental_data_zero,
    global_settings,
):
    """Test case for hmc iterator."""
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
    iterator = HamiltonianMonteCarlo(
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

    assert results["mean"].mean(axis=0) == pytest.approx(
        np.array([0.19363280864587615, -1.1303341362165935])
    )
    assert results["var"].mean(axis=0) == pytest.approx([0, 0])
