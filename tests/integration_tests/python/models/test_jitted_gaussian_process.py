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
"""Integration tests for the jitted GP model."""

from copy import deepcopy

import numpy as np
import pytest

from example_simulator_functions.sinus import gradient_sinus_test_fun, sinus_test_fun
from queens.models.surrogates.jitted_gaussian_process import JittedGaussianProcess
from queens.stochastic_optimizers import Adam
from test_utils.integration_tests import (  # pylint: disable=wrong-import-order
    assert_surrogate_model_output,
)


@pytest.fixture(name="gp_model")
def fixture_gp_model():
    """A jitted GP model."""
    optimizer = Adam(
        learning_rate=0.05,
        optimization_type="max",
        rel_l1_change_threshold=0.005,
        rel_l2_change_threshold=0.005,
    )
    model = JittedGaussianProcess(
        stochastic_optimizer=optimizer,
        kernel_type="squared_exponential",
        initial_hyper_params_lst=[1.0, 1.0, 0.01],
        plot_refresh_rate=10,
        noise_var_lb=1.0e-4,
        data_scaling="standard_scaler",
    )
    return model


def test_jitted_gaussian_process_one_dim(gp_model):
    """Test one dimensional jitted GP."""
    n_train = 25
    x_train = np.linspace(-5, 5, n_train).reshape(-1, 1)
    y_train = sinus_test_fun(x_train)

    # evaluate the testing/benchmark function at testing inputs
    x_test = np.linspace(-5, 5, 200).reshape(-1, 1)
    mean_ref, gradient_mean_ref = gradient_sinus_test_fun(x_test)
    var_ref = np.zeros(mean_ref.shape)

    # -- squared exponential kernel --
    # --- get the mean and variance of the model (no gradient call here) ---
    my_model = deepcopy(gp_model)
    my_model.setup(x_train, y_train)
    my_model.train()

    output = my_model.predict(x_test)
    assert_surrogate_model_output(output, mean_ref, var_ref)

    # -- now call the gradient function of the model---
    output = my_model.predict(x_test, gradient_bool=True)

    gradient_variance_ref = np.zeros(gradient_mean_ref.shape)
    decimals = (2, 2, 2, 2)
    assert_surrogate_model_output(
        output, mean_ref, var_ref, gradient_mean_ref, gradient_variance_ref, decimals
    )

    # -- matern-3-2 kernel --
    # --- get the mean and variance of the model (no gradient call here) ---
    gp_model.kernel_type = "matern_3_2"
    gp_model.setup(x_train, y_train)
    gp_model.train()

    output = gp_model.predict(x_test)
    assert_surrogate_model_output(output, mean_ref, var_ref, decimals=decimals)

    # -- now call the gradient function of the model---
    with pytest.raises(NotImplementedError):
        gp_model.predict(x_test, gradient_bool=True)


@pytest.mark.max_time_for_test(30)
def test_jitted_gaussian_process_two_dim(gp_model, training_data_park91a, testing_data_park91a):
    """Test two dimensional jitted GP."""
    x_train, y_train = training_data_park91a
    gp_model.setup(x_train, y_train)
    gp_model.train()

    x_test, mean_ref, var_ref, gradient_mean_ref, gradient_variance_ref = testing_data_park91a

    # --- get the mean and variance of the model (no gradient call here) ---
    output = gp_model.predict(x_test)
    assert_surrogate_model_output(output, mean_ref, var_ref)

    # -- now call the gradient function of the model---
    output = gp_model.predict(x_test, gradient_bool=True)

    decimals = (2, 2, 1, 2)
    assert_surrogate_model_output(
        output, mean_ref, var_ref, gradient_mean_ref, gradient_variance_ref, decimals
    )
