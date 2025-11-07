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
"""Fixtures for the integration tests of models."""

import numpy as np
import pytest

from example_simulator_functions.park91a import park91a_hifi
from test_utils.integration_tests import get_input_park91a


@pytest.fixture(name="training_data_park91a")
def fixture_training_data_park91a():
    """Create training data from the park91a benchmark function."""
    # create training inputs
    n_train = 7
    x_train, x_3, x_4 = get_input_park91a(n_train)

    # evaluate the testing/benchmark function at training inputs
    y_train = park91a_hifi(x_train[:, 0], x_train[:, 1], x_3, x_4, gradient_bool=False)
    y_train = y_train.reshape(-1, 1)

    return x_train, y_train


@pytest.fixture(name="testing_data_park91a")
def fixture_testing_data_park91a():
    """Create testing data for the park91a benchmark function."""
    # create testing inputs
    n_test = 25
    x_test, x_3, x_4 = get_input_park91a(n_test)

    # evaluate the testing/benchmark function at testing inputs
    mean_ref, gradient_mean_ref = park91a_hifi(
        x_test[:, 0], x_test[:, 1], x_3, x_4, gradient_bool=True
    )
    mean_ref = mean_ref.reshape(-1, 1)
    gradient_mean_ref = np.array(gradient_mean_ref).T
    var_ref = np.zeros(mean_ref.shape)
    gradient_variance_ref = np.zeros(gradient_mean_ref.shape)

    return x_test, mean_ref, var_ref, gradient_mean_ref, gradient_variance_ref
