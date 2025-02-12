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
"""Test discrete uniform distributions."""

import numpy as np
import pytest

from queens.distributions.uniform_discrete import UniformDiscrete
from test_utils.unit_tests.distributions import covariance_discrete


@pytest.fixture(name="reference_data", params=[1, 2])
def fixture_reference_data(request):
    """Data for the distribution."""
    reference_weights = [1, 1, 1, 1]
    reference_probabilities = np.array([0.25, 0.25, 0.25, 0.25])
    if request.param == 1:
        return reference_weights, reference_probabilities, [[0], [1], [2], [3]], 1

    return reference_weights, reference_probabilities, [[1, 2], [3, 4], [2, 2], [0, 3]], 2


@pytest.fixture(name="distribution")
def fixture_distribution(reference_data):
    """Distribution fixture."""
    _, _, sample_space, _ = reference_data
    return UniformDiscrete(sample_space)


@pytest.fixture(name="distribution_for_init", params=[0, 1])
def fixture_distribution_for_init(reference_data, distribution, request):
    """Distribution fixture."""
    if request.param == 0:
        return distribution

    (
        _,
        _,
        reference_sample_space,
        _,
    ) = reference_data

    return UniformDiscrete(sample_space=reference_sample_space)


def test_init_success(reference_data, distribution_for_init):
    """Test init method."""
    _, reference_probabilities, reference_sample_space, reference_dimension = reference_data

    reference_mean = np.sum(
        [
            probability * np.array(value)
            for probability, value in zip(reference_probabilities, reference_sample_space)
        ],
        axis=0,
    )

    reference_covariance = covariance_discrete(
        reference_probabilities, reference_sample_space, reference_mean
    )

    np.testing.assert_allclose(reference_probabilities, distribution_for_init.probabilities)
    np.testing.assert_allclose(reference_sample_space, distribution_for_init.sample_space)
    np.testing.assert_allclose(reference_dimension, distribution_for_init.dimension)
    np.testing.assert_allclose(reference_mean, distribution_for_init.mean)
    np.testing.assert_allclose(reference_covariance, distribution_for_init.covariance)


@pytest.mark.parametrize("sample_space", [[1, 1, 2], [[1, 1], [2, 1], [1, 1]]])
def test_init_failure(sample_space):
    """Test if invalid options lead to errors."""
    with pytest.raises(ValueError, match="The sample space contains duplicate events"):
        UniformDiscrete(sample_space)
