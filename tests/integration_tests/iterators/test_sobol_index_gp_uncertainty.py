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
"""Integration tests for Sobol indices estimation with Ishigami function.

This test uses metamodel uncertainty.
"""

import numpy as np

from queens.distributions.uniform import Uniform
from queens.drivers.function import Function
from queens.iterators.latin_hypercube_sampling import LatinHypercubeSampling
from queens.iterators.sobol_index_gp_uncertainty import SobolIndexGPUncertainty
from queens.main import run_iterator
from queens.models.simulation import Simulation
from queens.models.surrogates.gaussian_process import GaussianProcess
from queens.parameters.parameters import Parameters
from queens.schedulers.pool import Pool
from queens.utils.io import load_result


def test_sobol_index_gp_uncertainty_ishigami(global_settings):
    """Test case for Sobol indices based on GP realizations."""
    # Parameters
    x1 = Uniform(lower_bound=-3.14159265359, upper_bound=3.14159265359)
    x2 = Uniform(lower_bound=-3.14159265359, upper_bound=3.14159265359)
    x3 = Uniform(lower_bound=-3.14159265359, upper_bound=3.14159265359)
    parameters = Parameters(x1=x1, x2=x2, x3=x3)

    # Setup iterator
    driver = Function(parameters=parameters, function="ishigami90")
    scheduler = Pool(experiment_name=global_settings.experiment_name)
    simulation_model = Simulation(scheduler=scheduler, driver=driver)
    training_iterator = LatinHypercubeSampling(
        seed=42,
        num_samples=100,
        num_iterations=10,
        model=simulation_model,
        parameters=parameters,
        global_settings=global_settings,
    )
    testing_iterator = LatinHypercubeSampling(
        seed=30,
        num_samples=100,
        num_iterations=10,
        model=simulation_model,
        parameters=parameters,
        global_settings=global_settings,
    )
    model = GaussianProcess(
        error_measures=["nash_sutcliffe_efficiency"],
        train_likelihood_variance=False,
        number_restarts=5,
        number_training_iterations=1000,
        dimension_lengthscales=3,
        seed_posterior_samples=42,
        training_iterator=training_iterator,
        testing_iterator=testing_iterator,
    )
    iterator = SobolIndexGPUncertainty(
        seed_monte_carlo=42,
        number_monte_carlo_samples=1000,
        number_gp_realizations=3,
        number_bootstrap_samples=2,
        second_order=True,
        sampling_approach="quasi_random",
        num_procs=6,
        seed_posterior_samples=42,
        first_order_estimator="Gratiet2014",
        result_description={"write_results": True},
        model=model,
        parameters=parameters,
        global_settings=global_settings,
    )

    # Actual analysis
    run_iterator(iterator, global_settings=global_settings)

    # Load results
    results = load_result(global_settings.result_file(".pickle"))

    expected_s1 = np.array(
        [
            [0.30469190, 0.00014149, 0.00005653, 0.00016402, 0.02331390, 0.01473639, 0.02510155],
            [0.38996188, 0.00039567, 0.00049108, 0.00003742, 0.03898644, 0.04343343, 0.01198891],
            [0.00383826, 0.00030052, 0.00008825, 0.00044747, 0.03397690, 0.01841250, 0.04146019],
        ]
    )
    expected_st = np.array(
        [
            [0.55816767, 0.00050181, 0.00001702, 0.00082728, 0.04390555, 0.00808476, 0.05637328],
            [0.50645929, 0.00022282, 0.00022212, 0.00010188, 0.02925636, 0.02921057, 0.01978344],
            [0.30344671, 0.00010415, 0.00011769, 0.00004659, 0.02000237, 0.02126261, 0.01337864],
        ]
    )
    expected_s2 = np.array(
        [
            [0.00461299, 0.00215561, 0.00006615, 0.00352044, 0.09099820, 0.01594134, 0.11629120],
            [0.19526686, 0.00147909, 0.00059668, 0.00169727, 0.07537822, 0.04787620, 0.08074639],
            [0.06760761, 0.00004854, 0.00002833, 0.00007491, 0.01365552, 0.01043203, 0.01696364],
        ]
    )

    np.testing.assert_allclose(results["first_order"].values, expected_s1, atol=1e-05)
    np.testing.assert_allclose(results["second_order"].values, expected_s2, atol=1e-05)
    np.testing.assert_allclose(results["total_order"].values, expected_st, atol=1e-05)


def test_sobol_index_gp_uncertainty_ishigami_third_order(global_settings):
    """Test case for third-order Sobol indices."""
    # Parameters
    x1 = Uniform(lower_bound=-3.14159265359, upper_bound=3.14159265359)
    x2 = Uniform(lower_bound=-3.14159265359, upper_bound=3.14159265359)
    x3 = Uniform(lower_bound=-3.14159265359, upper_bound=3.14159265359)
    parameters = Parameters(x1=x1, x2=x2, x3=x3)

    # Setup iterator
    driver = Function(parameters=parameters, function="ishigami90")
    scheduler = Pool(experiment_name=global_settings.experiment_name)
    simulation_model = Simulation(scheduler=scheduler, driver=driver)
    training_iterator = LatinHypercubeSampling(
        seed=42,
        num_samples=100,
        num_iterations=10,
        model=simulation_model,
        parameters=parameters,
        global_settings=global_settings,
    )
    testing_iterator = LatinHypercubeSampling(
        seed=30,
        num_samples=100,
        num_iterations=10,
        model=simulation_model,
        parameters=parameters,
        global_settings=global_settings,
    )
    gpflow_regression_model = GaussianProcess(
        error_measures=["nash_sutcliffe_efficiency"],
        train_likelihood_variance=False,
        number_restarts=5,
        number_training_iterations=1000,
        dimension_lengthscales=3,
        seed_posterior_samples=42,
        training_iterator=training_iterator,
        testing_iterator=testing_iterator,
    )
    iterator = SobolIndexGPUncertainty(
        seed_monte_carlo=42,
        number_monte_carlo_samples=1000,
        number_gp_realizations=20,
        number_bootstrap_samples=10,
        third_order=True,
        third_order_parameters=["x1", "x2", "x3"],
        sampling_approach="pseudo_random",
        num_procs=6,
        seed_posterior_samples=42,
        first_order_estimator="Saltelli2010",
        result_description={"write_results": True},
        model=gpflow_regression_model,
        parameters=parameters,
        global_settings=global_settings,
    )

    # Actual analysis
    run_iterator(iterator, global_settings=global_settings)

    # Load results
    results = load_result(global_settings.result_file(".pickle"))

    expected_s3 = np.array(
        [[0.23426643, 0.00801287, 0.00230968, 0.00729179, 0.17544544, 0.09419407, 0.16736517]]
    )

    np.testing.assert_allclose(results["third_order"].values, expected_s3, atol=1e-05)


def test_sobol_index_gp_uncertainty_mean_ishigami(global_settings):
    """Test case for Sobol indices based on GP mean."""
    # Parameters
    x1 = Uniform(lower_bound=-3.14159265359, upper_bound=3.14159265359)
    x2 = Uniform(lower_bound=-3.14159265359, upper_bound=3.14159265359)
    x3 = Uniform(lower_bound=-3.14159265359, upper_bound=3.14159265359)
    parameters = Parameters(x1=x1, x2=x2, x3=x3)

    # Setup iterator
    driver = Function(parameters=parameters, function="ishigami90")
    scheduler = Pool(experiment_name=global_settings.experiment_name)
    simulation_model = Simulation(scheduler=scheduler, driver=driver)
    training_iterator = LatinHypercubeSampling(
        seed=42,
        num_samples=100,
        num_iterations=10,
        model=simulation_model,
        parameters=parameters,
        global_settings=global_settings,
    )
    testing_iterator = LatinHypercubeSampling(
        seed=30,
        num_samples=100,
        num_iterations=10,
        model=simulation_model,
        parameters=parameters,
        global_settings=global_settings,
    )
    gpflow_regression_model = GaussianProcess(
        error_measures=["nash_sutcliffe_efficiency"],
        train_likelihood_variance=False,
        number_restarts=5,
        number_training_iterations=1000,
        dimension_lengthscales=3,
        seed_posterior_samples=42,
        training_iterator=training_iterator,
        testing_iterator=testing_iterator,
    )
    iterator = SobolIndexGPUncertainty(
        seed_monte_carlo=42,
        number_monte_carlo_samples=1000,
        number_gp_realizations=1,
        number_bootstrap_samples=2,
        sampling_approach="pseudo_random",
        second_order=False,
        num_procs=6,
        first_order_estimator="Janon2014",
        result_description={"write_results": True},
        model=gpflow_regression_model,
        parameters=parameters,
        global_settings=global_settings,
    )

    # Actual analysis
    run_iterator(iterator, global_settings=global_settings)

    # Load results
    results = load_result(global_settings.result_file(".pickle"))

    expected_s1 = np.array(
        [
            [0.28879163, 0.00022986, np.nan, 0.00022986, 0.02971550, np.nan, 0.02971550],
            [0.45303182, 0.00000033, np.nan, 0.00000033, 0.00112608, np.nan, 0.00112608],
            [0.07601656, 0.00000084, np.nan, 0.00000084, 0.00179415, np.nan, 0.00179415],
        ]
    )
    expected_st = np.array(
        [
            [0.47333086, 0.00093263, np.nan, 0.00093263, 0.05985535, np.nan, 0.05985535],
            [0.48403078, 0.00000185, np.nan, 0.00000185, 0.00266341, np.nan, 0.00266341],
            [0.23926036, 0.00003290, np.nan, 0.00003290, 0.01124253, np.nan, 0.01124253],
        ]
    )

    np.testing.assert_allclose(results["first_order"].values, expected_s1, atol=1e-05)
    np.testing.assert_allclose(results["total_order"].values, expected_st, atol=1e-05)
