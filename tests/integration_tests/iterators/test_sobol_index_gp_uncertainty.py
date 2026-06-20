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
            [
                2.97063963e-01,
                1.32172811e-04,
                2.56038695e-05,
                1.93443666e-04,
                2.25330072e-02,
                9.91746996e-03,
                2.72599684e-02,
            ],
            [
                4.31449233e-01,
                3.71441195e-04,
                2.00571162e-04,
                3.79370323e-04,
                3.77740130e-02,
                2.77576270e-02,
                3.81750635e-02,
            ],
            [
                5.08066819e-03,
                1.64495536e-04,
                9.87514222e-05,
                2.26925997e-04,
                2.51376774e-02,
                1.94768971e-02,
                2.95250211e-02,
            ],
        ]
    )
    expected_s2 = np.array(
        [
            [-0.0559034, 0.00195262, 0.00046626, 0.00267564, 0.08660769, 0.0423217, 0.10138234],
            [0.1564238, 0.00096758, 0.00046437, 0.00107588, 0.06096652, 0.04223571, 0.06428802],
            [0.02015287, 0.00084594, 0.00046812, 0.00079983, 0.05700563, 0.04240573, 0.05543033],
        ]
    )
    expected_st = np.array(
        [
            [
                5.34063660e-01,
                6.81003214e-04,
                3.10790784e-04,
                7.27864573e-04,
                5.11472952e-02,
                3.45527133e-02,
                5.28778005e-02,
            ],
            [
                5.15767502e-01,
                2.51197174e-04,
                1.61311940e-04,
                2.33854812e-04,
                3.10638632e-02,
                2.48932355e-02,
                2.99723811e-02,
            ],
            [
                2.97958247e-01,
                2.80315660e-04,
                1.75633314e-04,
                2.56443896e-04,
                3.28149519e-02,
                2.59747597e-02,
                3.13866001e-02,
            ],
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
        [
            [0.13701495, 0.00729926, 0.00322222, 0.00587944, 0.1674509, 0.11125658, 0.15028514],
        ]
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
            [
                3.21085015e-01,
                6.84428797e-05,
                np.nan,
                6.84428797e-05,
                1.62148236e-02,
                np.nan,
                1.62148236e-02,
            ],
            [
                5.08557775e-01,
                1.58781059e-03,
                np.nan,
                1.58781059e-03,
                7.80993532e-02,
                np.nan,
                7.80993532e-02,
            ],
            [
                5.48899186e-02,
                9.71093479e-05,
                np.nan,
                9.71093479e-05,
                1.93142839e-02,
                np.nan,
                1.93142839e-02,
            ],
        ]
    )
    expected_st = np.array(
        [
            [
                4.99688577e-01,
                5.86983409e-04,
                np.nan,
                5.86983409e-04,
                4.74854988e-02,
                np.nan,
                4.74854988e-02,
            ],
            [
                5.40964137e-01,
                2.17464690e-04,
                np.nan,
                2.17464690e-04,
                2.89029696e-02,
                np.nan,
                2.89029696e-02,
            ],
            [
                2.63508722e-01,
                4.39817083e-05,
                np.nan,
                4.39817083e-05,
                1.29982276e-02,
                np.nan,
                1.29982276e-02,
            ],
        ]
    )

    np.testing.assert_allclose(results["first_order"].values, expected_s1, atol=1e-05)
    np.testing.assert_allclose(results["total_order"].values, expected_st, atol=1e-05)
