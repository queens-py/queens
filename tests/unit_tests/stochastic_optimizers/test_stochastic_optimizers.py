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
"""Integration tests for stochastic optimizers."""

import numpy as np
import pytest

from queens.stochastic_optimizers import SGD, Adam, Adamax, RMSprop


def test_rmsprop_max(rmsprop_optimizer):
    """Test RMSprop."""
    varparams = 5 * np.ones(5)
    rmsprop_optimizer.current_variational_parameters = varparams
    rmsprop_optimizer.set_gradient_function(gradient)
    result = None
    for r in rmsprop_optimizer:
        result = r

    iterations = rmsprop_optimizer.iteration
    assert iterations == 500
    assert np.mean(result - 0.5) < 0.05


def test_adamax(adamax_optimizer):
    """Test Adamax."""
    varparams = np.ones(5)
    adamax_optimizer.current_variational_parameters = varparams
    adamax_optimizer.set_gradient_function(lambda x: -gradient(x))
    result = adamax_optimizer.run_optimization()
    iterations = adamax_optimizer.iteration
    assert iterations < 1000
    assert np.mean(result - 0.5) < 0.005


def test_adam(adam_optimizer):
    """Test Adam."""
    varparams = np.ones(5)
    adam_optimizer.current_variational_parameters = varparams
    adam_optimizer.set_gradient_function(gradient)
    result = adam_optimizer.run_optimization()
    iterations = adam_optimizer.iteration
    assert iterations < 1000
    assert np.mean(result - 0.5) < 0.005


def test_sgd(sgd_optimizer):
    """Test Adam."""
    varparams = np.ones(5)
    sgd_optimizer.current_variational_parameters = varparams
    sgd_optimizer.set_gradient_function(gradient)
    result = sgd_optimizer.run_optimization()
    iterations = sgd_optimizer.iteration
    assert iterations < 1000
    assert np.mean(result - 0.5) < 0.005


@pytest.fixture(name="sgd_optimizer")
def fixture_sgd_optimizer():
    """An SGD optimizer."""
    optimizer = SGD(
        learning_rate=1e-2,
        optimization_type="max",
        rel_l1_change_threshold=1e-4,
        rel_l2_change_threshold=1e-6,
        max_iteration=1000,
    )
    return optimizer


@pytest.fixture(name="adam_optimizer")
def fixture_adam_optimizer():
    """An Adam optimizer."""
    optimizer = Adam(
        learning_rate=1e-2,
        optimization_type="max",
        rel_l1_change_threshold=1e-4,
        rel_l2_change_threshold=1e-6,
        max_iteration=1000,
    )
    return optimizer


@pytest.fixture(name="adamax_optimizer")
def fixture_adamax_optimizer():
    """An Adamax optimizer."""
    optimizer = Adamax(
        learning_rate=1e-2,
        optimization_type="min",
        rel_l1_change_threshold=1e-4,
        rel_l2_change_threshold=1e-6,
        max_iteration=1000,
    )
    return optimizer


@pytest.fixture(name="rmsprop_optimizer")
def fixture_rmsprop_optimizer():
    """A RMSprop optimzer."""
    optimizer = RMSprop(
        learning_rate=5e-2,
        optimization_type="max",
        rel_l1_change_threshold=-1,
        rel_l2_change_threshold=-1,
        max_iteration=500,
    )
    return optimizer


def gradient(x):
    """Gradient function."""
    return -2 * (x - 0.5)
