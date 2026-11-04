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
"""Integration tests for the Bayesian Optimization iterator."""

import numpy as np

from example_simulator_functions import goldstein_price
from queens.distributions.uniform import Uniform
from queens.drivers.function import Function
from queens.iterators.bayesian_optimization import BayesianOptimization
from queens.main import run_iterator
from queens.models.simulation import Simulation
from queens.models.surrogates.gaussian_process import GaussianProcess
from queens.models.surrogates.jitted_gaussian_process import JittedGaussianProcess
from queens.parameters.parameters import Parameters
from queens.schedulers.pool import Pool
from queens.stochastic_optimizers import Adam
from queens.utils.acquisition_functions import ExpectedImprovement
from queens.utils.io import load_result


def test_bayesian_optimization_branin78_hifi(global_settings):
    """Test bayesian optimization iterator on branin78_hifi function."""
    x1 = Uniform(lower_bound=-5.0, upper_bound=10.0)
    x2 = Uniform(lower_bound=-0.0, upper_bound=15.0)
    parameters = Parameters(x1=x1, x2=x2)

    driver = Function(parameters=parameters, function="branin78_hifi")
    scheduler = Pool(experiment_name=global_settings.experiment_name)
    model = Simulation(scheduler=scheduler, driver=driver)

    gp_optimizer = Adam(
        learning_rate=0.05,
        optimization_type="max",
        rel_l1_change_threshold=0.005,
        rel_l2_change_threshold=0.005,
    )

    surrogate_model = JittedGaussianProcess(
        stochastic_optimizer=gp_optimizer,
        kernel_type="matern_3_2",
        initial_hyper_params_lst=[1.0, 1.0, 1.0e-5],
        noise_var_lb=1.0e-6,
        data_scaling="standard_scaler",
    )

    acquisition_function = ExpectedImprovement()

    iterator = BayesianOptimization(
        model=model,
        parameters=parameters,
        global_settings=global_settings,
        surrogate_model=surrogate_model,
        acquisition_function=acquisition_function,
        result_description={"write_results": True},
        num_initial_samples=8,
        max_evaluations=32,
        num_acquisition_restarts=16,
        seed=42,
    )

    run_iterator(iterator, global_settings=global_settings)
    results = load_result(global_settings.result_file(".pickle"))

    np.testing.assert_allclose(
        results.fun, np.array(5.0 / (4.0 * np.pi)), atol=1.0e-02, rtol=1.0e-02
    )


def test_bayesian_optimization_forrester(global_settings):
    """Test bayesian optimization iterator on forrester function with GP."""
    x1 = Uniform(lower_bound=-0.0, upper_bound=1.0)
    parameters = Parameters(x1=x1)

    driver = Function(parameters=parameters, function="forrester")
    scheduler = Pool(experiment_name=global_settings.experiment_name)
    model = Simulation(scheduler=scheduler, driver=driver)

    surrogate_model = GaussianProcess(
        dimension_lengthscales=parameters.num_parameters,
        number_restarts=5,
    )

    acquisition_function = ExpectedImprovement()

    iterator = BayesianOptimization(
        model=model,
        parameters=parameters,
        global_settings=global_settings,
        surrogate_model=surrogate_model,
        acquisition_function=acquisition_function,
        result_description={"write_results": True},
        num_initial_samples=4,
        max_evaluations=12,
        num_acquisition_restarts=8,
        seed=42,
    )

    run_iterator(iterator, global_settings=global_settings)
    results = load_result(global_settings.result_file(".pickle"))

    np.testing.assert_allclose(results.x, np.array(0.757248757841856), rtol=1.0e-03)
    np.testing.assert_allclose(results.fun, np.array(-6.020740055767083), atol=1.0e-03)


def test_bayesian_optimization_goldstein_price(global_settings):
    """Test bayesian optimization iterator on goldstein price function."""
    x1 = Uniform(lower_bound=-2.0, upper_bound=2.0)
    x2 = Uniform(lower_bound=-2.0, upper_bound=2.0)
    parameters = Parameters(x1=x1, x2=x2)

    def log_goldstein_price(job_id=None, **parameters):
        for key in ("job_id", "num_procs", "experiment_dir", "experiment_name"):
            parameters.pop(key, None)
        del job_id
        return np.log(goldstein_price(**parameters))

    driver = Function(parameters=parameters, function=log_goldstein_price)
    scheduler = Pool(experiment_name=global_settings.experiment_name)
    model = Simulation(scheduler=scheduler, driver=driver)

    gp_optimizer = Adam(
        learning_rate=0.02,
        optimization_type="max",
        rel_l1_change_threshold=0.005,
        rel_l2_change_threshold=0.005,
    )

    surrogate_model = JittedGaussianProcess(
        stochastic_optimizer=gp_optimizer,
        kernel_type="matern_3_2",
        initial_hyper_params_lst=[1.0, 1.0, 1.0e-5],
        noise_var_lb=1.0e-6,
        data_scaling="standard_scaler",
    )

    acquisition_function = ExpectedImprovement(exploration_parameter=0.0)

    iterator = BayesianOptimization(
        model=model,
        parameters=parameters,
        global_settings=global_settings,
        surrogate_model=surrogate_model,
        acquisition_function=acquisition_function,
        result_description={"write_results": True},
        num_initial_samples=8,
        max_evaluations=38,
        num_acquisition_restarts=8,
        seed=42,
    )

    run_iterator(iterator, global_settings=global_settings)
    results = load_result(global_settings.result_file(".pickle"))

    np.testing.assert_allclose(np.exp(results.fun), 3.0, atol=1.0e-01, rtol=1.0e-01)
