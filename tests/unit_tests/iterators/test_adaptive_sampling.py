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
"""Unit tests for AdaptiveSampling iterator.

Currently only tests the internal method _filter_failed_evaluations
since there is an integration test covering the full functionality
already.
"""

from unittest.mock import Mock

import numpy as np
import pytest

from queens.iterators.adaptive_sampling import AdaptiveSampling


@pytest.fixture(name="adaptive_sampling_iterator")
def fixture_adaptive_sampling_iterator(global_settings, default_parameters_uniform_2d):
    """Fixture for AdaptiveSampling iterator with mocked dependencies."""
    # Mock model
    model = Mock()

    # Mock likelihood model with observations
    likelihood_model = Mock()
    likelihood_model.observations = np.array([1.0, 2.0, 3.0])

    # Mock solving iterator
    solving_iterator = Mock()

    # Initial training samples
    initial_train_samples = np.array([[0.1, 0.2], [0.3, 0.4]])

    iterator = AdaptiveSampling(
        model=model,
        parameters=default_parameters_uniform_2d,
        global_settings=global_settings,
        likelihood_model=likelihood_model,
        initial_train_samples=initial_train_samples,
        solving_iterator=solving_iterator,
        num_new_samples=2,
        num_steps=1,
    )

    return iterator


def test_filter_failed_evaluations_some_failures(adaptive_sampling_iterator):
    """Test _filter_failed_evaluations when some evaluations fail."""
    # Set up test data with some NaN values (row 1 has NaN)
    adaptive_sampling_iterator.x_train = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
    adaptive_sampling_iterator.model_outputs = np.array(
        [[1.0, 2.0, 3.0], [np.nan, 5.0, 6.0], [7.0, 8.0, 9.0]]
    )
    adaptive_sampling_iterator.x_train_new = np.array([[0.3, 0.4], [0.5, 0.6]])

    # Run the method
    adaptive_sampling_iterator._filter_failed_evaluations()  # pylint: disable=protected-access

    # Verify successful evaluations remain (rows 0 and 2)
    expected_x_train = np.array([[0.1, 0.2], [0.5, 0.6]])
    expected_model_outputs = np.array([[1.0, 2.0, 3.0], [7.0, 8.0, 9.0]])

    np.testing.assert_array_equal(adaptive_sampling_iterator.x_train, expected_x_train)
    np.testing.assert_array_equal(adaptive_sampling_iterator.model_outputs, expected_model_outputs)

    # Verify failed evaluations were stored (row 1)
    expected_x_train_failed = np.array([[0.3, 0.4]])
    expected_model_outputs_failed = np.array([[np.nan, 5.0, 6.0]])

    np.testing.assert_array_equal(
        adaptive_sampling_iterator.x_train_failed, expected_x_train_failed
    )
    np.testing.assert_array_equal(
        adaptive_sampling_iterator.model_outputs_failed, expected_model_outputs_failed
    )


def test_filter_failed_evaluations_all_failures(adaptive_sampling_iterator):
    """Test _filter_failed_evaluations when all evaluations fail."""
    # Set up test data with all NaN values
    adaptive_sampling_iterator.x_train = np.array([[0.1, 0.2], [0.3, 0.4]])
    adaptive_sampling_iterator.model_outputs = np.array([[np.nan, 2.0, 3.0], [4.0, np.nan, 6.0]])
    adaptive_sampling_iterator.x_train_new = np.array([[0.1, 0.2], [0.3, 0.4]])

    # Run the method
    adaptive_sampling_iterator._filter_failed_evaluations()  # pylint: disable=protected-access

    # Verify all evaluations were filtered out
    assert adaptive_sampling_iterator.x_train.shape[0] == 0
    assert adaptive_sampling_iterator.model_outputs.shape[0] == 0

    # Verify all failed evaluations were stored
    expected_x_train_failed = np.array([[0.1, 0.2], [0.3, 0.4]])
    np.testing.assert_array_equal(
        adaptive_sampling_iterator.x_train_failed, expected_x_train_failed
    )
    assert adaptive_sampling_iterator.model_outputs_failed.shape[0] == 2


def test_filter_failed_evaluations_multiple_calls(adaptive_sampling_iterator):
    """Test _filter_failed_evaluations across multiple calls."""
    # First call with one failure
    adaptive_sampling_iterator.x_train = np.array([[0.1, 0.2], [0.3, 0.4]])
    adaptive_sampling_iterator.model_outputs = np.array([[np.nan, 2.0, 3.0], [4.0, 5.0, 6.0]])
    adaptive_sampling_iterator.x_train_new = np.array([[0.1, 0.2], [0.3, 0.4]])

    adaptive_sampling_iterator._filter_failed_evaluations()  # pylint: disable=protected-access

    # Verify first filtering
    assert adaptive_sampling_iterator.x_train_failed.shape[0] == 1
    np.testing.assert_array_equal(adaptive_sampling_iterator.x_train_failed, [[0.1, 0.2]])

    # Second call with another failure
    adaptive_sampling_iterator.x_train = np.array([[0.3, 0.4], [0.5, 0.6]])
    adaptive_sampling_iterator.model_outputs = np.array([[4.0, 5.0, 6.0], [7.0, np.nan, 9.0]])
    adaptive_sampling_iterator.x_train_new = np.array([[0.5, 0.6]])

    adaptive_sampling_iterator._filter_failed_evaluations()  # pylint: disable=protected-access

    # Verify failed evaluations were accumulated
    assert adaptive_sampling_iterator.x_train_failed.shape[0] == 2
    expected_x_train_failed = np.array([[0.1, 0.2], [0.5, 0.6]])
    np.testing.assert_array_equal(
        adaptive_sampling_iterator.x_train_failed, expected_x_train_failed
    )
