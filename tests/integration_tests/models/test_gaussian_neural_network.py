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
        seed=42,
        data_scaling="standard_scaler",
        nugget_std=1.0e-02,
        verbosity_on=False,
    )
    return model


X_TEST_ONE_DIM = np.linspace(-5, 5, 20).reshape(-1, 1)


def _one_dim_converged_reference_values():
    """Reference values for the converged one-dimensional sine test."""
    mean_ref, gradient_mean_ref = gradient_sinus_test_fun(X_TEST_ONE_DIM)
    var_ref = np.zeros(mean_ref.shape)
    gradient_variance_ref = np.zeros(gradient_mean_ref.shape)

    return mean_ref, var_ref, gradient_mean_ref, gradient_variance_ref


def _one_dim_trained_reference_values():
    """Reference values for the regular one-dimensional integration test."""
    mean_ref = np.array(
        [
            [1.05701616],
            [0.91936273],
            [0.69505582],
            [0.28266456],
            [-0.23943175],
            [-0.69876824],
            [-0.96818476],
            [-0.96579658],
            [-0.71190977],
            [-0.26098673],
            [0.26902289],
            [0.70484324],
            [0.96753779],
            [0.96312091],
            [0.69956277],
            [0.24406864],
            [-0.27454099],
            [-0.72444972],
            [-0.96964804],
            [-0.9596871],
        ]
    )
    var_ref = np.array(
        [
            [2.41518859e-03],
            [1.81828461e-03],
            [1.17050295e-03],
            [5.85351970e-04],
            [2.77973113e-04],
            [1.52831862e-04],
            [9.70890580e-05],
            [7.15952948e-05],
            [6.14416636e-05],
            [5.74559826e-05],
            [5.57975187e-05],
            [5.50417619e-05],
            [5.46976234e-05],
            [5.45183656e-05],
            [5.44144290e-05],
            [5.43489094e-05],
            [5.43056093e-05],
            [5.42796753e-05],
            [5.42634170e-05],
            [5.42580518e-05],
        ]
    )
    gradient_mean_ref = np.array(
        [
            [-0.21565741],
            [-0.32315805],
            [-0.56097067],
            [-1.01188688],
            [-0.95944631],
            [-0.73350695],
            [-0.23375639],
            [0.24355919],
            [0.73050255],
            [0.94720195],
            [0.96924751],
            [0.79680802],
            [0.21733357],
            [-0.24895149],
            [-0.74798671],
            [-0.96806014],
            [-0.95030714],
            [-0.71774647],
            [-0.22157872],
            [0.19674541],
        ]
    )

    gradient_variance_ref = np.array(
        [
            [-1.04206286e-03],
            [-1.20268649e-03],
            [-1.23064309e-03],
            [-8.95813142e-04],
            [-3.57500360e-04],
            [-1.50596134e-04],
            [-7.30025367e-05],
            [-2.97073841e-05],
            [-1.16857120e-05],
            [-4.58430637e-06],
            [-2.10000706e-06],
            [-9.23209819e-07],
            [-4.54364139e-07],
            [-2.54516917e-07],
            [-1.49894128e-07],
            [-1.02965997e-07],
            [-6.36046574e-08],
            [-3.75338863e-08],
            [-2.20058965e-08],
            [-2.47589079e-09],
        ]
    )

    return mean_ref, var_ref, gradient_mean_ref, gradient_variance_ref


@pytest.mark.parametrize(
    ("n_train", "reference_values", "decimals"),
    [
        pytest.param(
            25, _one_dim_trained_reference_values(), (2, 2, 2, 1), id="integration-reference"
        ),
        pytest.param(
            1000,
            _one_dim_converged_reference_values(),
            (2, 4, 0, 8),
            marks=pytest.mark.convergence_tests,
            id="convergence-reference",
        ),
    ],
)
def test_gaussian_neural_network_one_dim(my_model, n_train, reference_values, decimals):
    """Test one dimensional gaussian nn."""
    x_train = np.linspace(-5, 5, n_train).reshape(-1, 1)
    y_train = sinus_test_fun(x_train)

    my_model.setup(x_train, y_train)
    my_model.train()

    x_test = X_TEST_ONE_DIM
    mean_ref, var_ref, gradient_mean_ref, gradient_variance_ref = reference_values

    # --- get the mean and variance of the model (no gradient call here) ---
    output = my_model.predict(x_test)
    assert_surrogate_model_output(output, mean_ref, var_ref, decimals=decimals)

    # -- now call the gradient function of the model---
    output = my_model.predict(x_test, gradient_bool=True)
    assert_surrogate_model_output(
        output, mean_ref, var_ref, gradient_mean_ref, gradient_variance_ref, decimals=decimals
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
