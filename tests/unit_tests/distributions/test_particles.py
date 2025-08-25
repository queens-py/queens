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
"""Test discrete particles distributions."""

import numpy as np
import pytest

from queens.distributions.particle import Particle
from test_utils.unit_tests.distributions import covariance_discrete


@pytest.fixture(name="reference_data", params=[1, 2])
def fixture_reference_data(request):
    """Data for the distribution."""
    reference_weights = [1, 2, 3, 4]
    reference_probabilities = np.array([0.1, 0.2, 0.3, 0.4])

    if request.param == 1:
        reference_sample_space = [[0], [1], [2], [3]]
        reference_dimension = 1
    else:
        reference_sample_space = [[1, 2], [3, 4], [2, 2], [0, 3]]
        reference_dimension = 2

    reference_mean = np.sum(
        [
            probability * np.array(value)
            for probability, value in zip(reference_probabilities, reference_sample_space)
        ],
        axis=0,
    )
    return (
        reference_weights,
        reference_probabilities,
        reference_sample_space,
        reference_dimension,
        reference_mean,
    )


@pytest.fixture(name="distribution")
def fixture_distribution(reference_data):
    """Distribution fixture."""
    _, probabilities, sample_space, _, _ = reference_data
    return Particle(probabilities, sample_space)


@pytest.fixture(name="distribution_fcc")
def fixture_distribution_fcc(reference_data):
    """Distribution fixture."""
    reference_probabilities, _, reference_sample_space, _, _ = reference_data

    distribution = Particle(
        probabilities=reference_probabilities,
        sample_space=reference_sample_space,
    )
    return distribution


@pytest.fixture(name="distributions", params=["init", "fcc"])
def fixture_distributions(request, distribution, distribution_fcc):
    """Distributions fixture once from init once from fcc."""
    if request.param == "init":
        return distribution
    return distribution_fcc


def test_init_probabilities(reference_data, distributions):
    """Test probabilities in the init method."""
    _, reference_probabilities, _, _, _ = reference_data

    np.testing.assert_allclose(reference_probabilities, distributions.probabilities)


def test_init_sample_space(reference_data, distributions):
    """Test sample_space in the init method."""
    _, _, reference_sample_space, _, _ = reference_data

    np.testing.assert_allclose(reference_sample_space, distributions.sample_space)


def test_init_dimension(reference_data, distributions):
    """Test dimension in the init method."""
    _, _, _, reference_dimension, _ = reference_data

    np.testing.assert_allclose(reference_dimension, distributions.dimension)


def test_init_mean(reference_data, distributions):
    """Test mean in the init method."""
    _, _, _, _, reference_mean = reference_data

    np.testing.assert_allclose(reference_mean, distributions.mean)


def test_init_covariance(reference_data, distributions):
    """Test covariance in the init method."""
    _, reference_probabilities, reference_sample_space, _, reference_mean = reference_data

    reference_covariance = covariance_discrete(
        reference_probabilities, reference_sample_space, reference_mean
    )

    np.testing.assert_allclose(reference_covariance, distributions.covariance)


def test_init_failure_mismatching_probabilities():
    """Test if mismatching number of probability leads to failure."""
    with pytest.raises(ValueError, match="The number of probabilities"):
        Particle([0.1, 1], [[1], [2], [3]])


def test_init_failure_negative_probabilities():
    """Test if negative values lead to errors."""
    with pytest.raises(ValueError, match="The parameter 'probabilities' has to be positive."):
        Particle([0.1, -1], [[1], [2]])


def test_init_failure_mismatching_dimension():
    """Test if mismatching dimensions of the event sa."""
    with pytest.raises(ValueError, match="Dimensions of the sample events do not match."):
        Particle([0.1, 1], [[1], [1, 2]])


def test_draw(mocker, reference_data, distribution):
    """Test draw."""
    _, _, reference_sample_space, _, _ = reference_data
    first_sample_event = 2
    second_sample_event = 3
    third_sample_event = 1
    mocker.patch(
        "queens.distributions.categorical.np.random.multinomial",
        return_value=np.array([first_sample_event, second_sample_event, third_sample_event]),
    )
    mocker.patch(
        "queens.distributions.categorical.np.random.shuffle",
    )
    reference_samples = [reference_sample_space[0]] * first_sample_event
    reference_samples.extend([reference_sample_space[1]] * second_sample_event)
    reference_samples.extend([reference_sample_space[2]] * third_sample_event)
    reference_samples = np.array(reference_samples, dtype=object).reshape(-1, 1)
    np.testing.assert_equal(reference_samples, distribution.draw(6))


def test_pdf(reference_data, distribution):
    """Test pdf."""
    _, reference_probabilities, reference_sample_space, _, _ = reference_data
    sample_location = np.array(reference_sample_space[::-1])
    np.testing.assert_allclose(reference_probabilities[::-1], distribution.pdf(sample_location))


def test_logpdf(reference_data, distribution):
    """Test logpdf."""
    _, reference_probabilities, reference_sample_space, _, _ = reference_data
    sample_location = np.array(reference_sample_space[::-1])
    np.testing.assert_allclose(
        np.log(reference_probabilities[::-1]), distribution.logpdf(sample_location)
    )


def test_cdf(reference_data, distribution):
    """Test cdf."""
    _, reference_probabilities, reference_sample_space, reference_dimension, _ = reference_data
    sample_location = np.array(reference_sample_space[::-1])

    if reference_dimension == 1:
        np.testing.assert_allclose(
            np.cumsum(reference_probabilities)[::-1], distribution.cdf(sample_location)
        )
    else:
        with pytest.raises(ValueError, match="Method does not support multivariate distributions!"):
            distribution.cdf(sample_location)


def test_ppf(reference_data, distribution):
    """Test ppf."""
    _, reference_probabilities, reference_sample_space, reference_dimension, _ = reference_data
    quantiles = np.array(np.cumsum(reference_probabilities)[::-1])

    if reference_dimension == 1:
        np.testing.assert_allclose(
            np.array(reference_sample_space[::-1]).reshape(-1), distribution.ppf(quantiles)
        )
    else:
        with pytest.raises(ValueError, match="Method does not support multivariate distributions!"):
            distribution.ppf(quantiles)
