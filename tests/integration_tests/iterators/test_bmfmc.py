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
"""Integration tests for the BMFMC routine.

The test is based on the high-fidelity Currin function.
"""

import numpy as np

# pylint: disable=invalid-name
import pytest
from scipy.stats import entropy

from example_simulator_functions.currin88 import currin88_hifi, currin88_lofi
from queens.distributions.uniform import Uniform
from queens.drivers.function import Function
from queens.iterators.bmfmc import BMFMC
from queens.main import run_iterator
from queens.models.bmfmc import BMFMC as BMFMCModel
from queens.models.simulation import Simulation
from queens.models.surrogates.gaussian_process import GaussianProcess
from queens.parameters.parameters import Parameters
from queens.schedulers.pool import Pool
from queens.utils.io import load_result
from queens.utils.pdf_estimation import estimate_bandwidth_for_kde, estimate_pdf
from queens.utils.process_outputs import write_results


# ---- actual integration tests -------------------------------------------------
def test_bmfmc_currin88_random_vars_diverse_design(
    tmp_path,
    _write_lf_mc_data_to_pickle,
    hf_mc_data,
    bandwidth_lf_mc,
    design_method,
    global_settings,
):
    """Integration tests for BMFMC routine using *currin88* function.

    The test uses a high-fidelity (HF) and a low-fidelity (LF) version
    of the *currin88* function.
    """
    plot_dir = tmp_path
    lf_mc_data_name = "LF_MC_data.pickle"
    path_lf_mc_pickle_file = tmp_path / lf_mc_data_name
    # Parameters
    x1 = Uniform(lower_bound=0.0, upper_bound=1.0)
    x2 = Uniform(lower_bound=0.0, upper_bound=1.0)
    parameters = Parameters(x1=x1, x2=x2)

    # Setup iterator
    probabilistic_mapping = GaussianProcess(
        train_likelihood_variance=False,
        number_restarts=2,
        number_training_iterations=1000,
        dimension_lengthscales=2,
    )
    driver = Function(parameters=parameters, function="currin88_hifi")
    scheduler = Pool(experiment_name=global_settings.experiment_name)
    hf_model = Simulation(scheduler=scheduler, driver=driver)
    model = BMFMCModel(
        predictive_var=False,
        BMFMC_reference=False,
        y_pdf_support_min=-0.5,
        y_pdf_support_max=15.0,
        path_to_lf_mc_data=(path_lf_mc_pickle_file,),
        path_to_hf_mc_reference_data=None,
        features_config="opt_features",
        num_features=1,
        probabilistic_mapping=probabilistic_mapping,
        hf_model=hf_model,
        parameters=parameters,
        global_settings=global_settings,
    )
    iterator = BMFMC(
        global_settings=global_settings,
        result_description={
            "write_results": True,
            "plotting_options": {
                "plot_booleans": [False, False, False],
                "plotting_dir": plot_dir,
                "plot_names": ["pdfs.eps", "manifold.eps", "ranking.eps"],
                "save_bool": [False, False, False],
                "animation_bool": False,
            },
        },
        initial_design={
            "num_HF_eval": 100,
            "num_bins": 50,
            "method": design_method,
            "seed": 1,
            "master_LF": 0,
        },
        model=model,
        parameters=parameters,
    )

    # Actual analysis
    run_iterator(iterator, global_settings=global_settings)

    # Load results
    results = load_result(global_settings.result_file(".pickle"))

    # get the y_support and calculate HF MC reference
    y_pdf_support = results["raw_output_data"]["y_pdf_support"]

    p_yhf_mc, _ = estimate_pdf(
        np.atleast_2d(hf_mc_data).T, bandwidth_lf_mc, support_points=np.atleast_2d(y_pdf_support)
    )

    kl_divergence = entropy(p_yhf_mc, results["raw_output_data"]["p_yhf_mean"])
    assert kl_divergence < 0.3


@pytest.fixture(name="monte_carlo_samples_x")
def fixture_monte_carlo_samples_x():
    """1000 uniform Monte Carlo samples for x1 and x2 between 0 and 1."""
    np.random.seed(1)
    n_samples = 1000
    monte_carlo_samples_x = np.random.uniform(low=0.0, high=1.0, size=(n_samples, 2))
    return monte_carlo_samples_x


@pytest.fixture(name="lf_mc_data")
def fixture_lf_mc_data(monte_carlo_samples_x):
    """Samples of low-fidelity model output using currin88_lofi."""
    y = []
    for x_vec in monte_carlo_samples_x:
        params = {"x1": x_vec[0], "x2": x_vec[1]}
        y.append(currin88_lofi(**params))

    y_lf_mc = np.array(y).reshape((monte_carlo_samples_x.shape[0], -1))

    return y_lf_mc


@pytest.fixture(name="hf_mc_data")
def fixture_hf_mc_data(monte_carlo_samples_x):
    """Samples of high-fidelity model output using currin88_hifi."""
    y = []
    for x_vec in monte_carlo_samples_x:
        params = {"x1": x_vec[0], "x2": x_vec[1]}
        y.append(currin88_hifi(**params))

    y_lf_mc = np.array(y).reshape((monte_carlo_samples_x.shape[0], -1))

    return y_lf_mc


@pytest.fixture(name="bandwidth_lf_mc")
def fixture_bandwidth_lf_mc(lf_mc_data):
    """Estimated bandwidth for KDE for low-fidelity data."""
    bandwidth_lf_mc = estimate_bandwidth_for_kde(
        lf_mc_data[:, 0], np.amin(lf_mc_data[:, 0]), np.amax(lf_mc_data[:, 0])
    )
    return bandwidth_lf_mc


@pytest.fixture(name="_write_lf_mc_data_to_pickle")
def fixture_write_lf_mc_data_to_pickle(tmp_path, monte_carlo_samples_x, lf_mc_data):
    """Write low-fidelity model data to a pickle file."""
    file_name = "LF_MC_data.pickle"
    input_description = {
        "x1": {
            "type": "uniform",
            "lower_bound": 0.0,
            "upper_bound": 1.0,
        },
        "x2": {
            "type": "uniform",
            "lower_bound": 0.0,
            "upper_bound": 1.0,
        },
    }
    data = {
        "input_data": monte_carlo_samples_x,
        "input_description": input_description,
        "output": lf_mc_data,
        "eigenfunc": None,
        "eigenvalue": None,
    }
    write_results(data, tmp_path / file_name)


@pytest.fixture(name="design_method", params=["random", "diverse_subset"])
def fixture_design_method(request):
    """Different design methods for parameterized tests."""
    design = request.param
    return design
