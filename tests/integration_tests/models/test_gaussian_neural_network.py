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
"""Integration tests for the Gaussian Neural Network regression model."""

import numpy as np
import pytest

from example_simulator_functions.sinus import gradient_sinus_test_fun, sinus_test_fun
from queens.models.surrogates.gaussian_neural_network import GaussianNeuralNetwork
from test_utils.integration_tests import (  # pylint: disable=wrong-import-order
    assert_surrogate_model_output,
)


@pytest.fixture(name="my_model")
def fixture_my_model():
    """A Gaussian neural network model."""
    model = GaussianNeuralNetwork(
        activation_per_hidden_layer_lst=["elu", "elu", "elu", "elu"],
        nodes_per_hidden_layer_lst=[20, 20, 20, 20],
        adams_training_rate=0.001,
        batch_size=50,
        num_epochs=3000,
        optimizer_seed=42,
        data_scaling="standard_scaler",
        nugget_std=1.0e-02,
        verbosity_on=False,
    )
    return model


def test_gaussian_neural_network_one_dim(my_model):
    """Test one dimensional gaussian nn."""
    n_train = 25
    x_train = np.linspace(-5, 5, n_train).reshape(-1, 1)
    y_train = sinus_test_fun(x_train)

    my_model.setup(x_train, y_train)
    my_model.train()

    # evaluate the testing/benchmark function at testing inputs
    x_test = np.linspace(-5, 5, 200).reshape(-1, 1)
    mean_ref, gradient_mean_ref = gradient_sinus_test_fun(x_test)
    var_ref = np.zeros(mean_ref.shape)

    # --- get the mean and variance of the model (no gradient call here) ---
    output = my_model.predict(x_test)
    assert_surrogate_model_output(output, mean_ref, var_ref)

    # -- now call the gradient function of the model---
    output = my_model.predict(x_test, gradient_bool=True)

    gradient_variance_ref = np.zeros(gradient_mean_ref.shape)
    decimals = (1, 2, 1, 2)
    assert_surrogate_model_output(
        output, mean_ref, var_ref, gradient_mean_ref, gradient_variance_ref, decimals
    )


def test_gaussian_neural_network_two_dim(my_model, training_data_park91a, testing_data_park91a):
    """Test two dimensional gaussian nn."""
    x_train, y_train = training_data_park91a
    my_model.setup(x_train, y_train)
    my_model.train()

    x_test, mean_ref, var_ref, gradient_mean_ref, gradient_variance_ref = testing_data_park91a

    # --- get the mean and variance of the model (no gradient call here) ---
    output = my_model.predict(x_test)
    assert_surrogate_model_output(output, mean_ref, var_ref)

    # -- now call the gradient function of the model---
    output = my_model.predict(x_test, gradient_bool=True)

    decimals = (2, 2, 1, 2)
    assert_surrogate_model_output(
        output, mean_ref, var_ref, gradient_mean_ref, gradient_variance_ref, decimals
    )
