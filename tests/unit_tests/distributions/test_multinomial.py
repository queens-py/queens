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
"""Test multinomial distribution."""

import numpy as np
import pytest

from queens.distributions.multinomial import Multinomial


@pytest.fixture(name="reference_data")
def fixture_reference_data():
    """Data for the distribution."""
    reference_n_trials = 10
    reference_probabilities = [0.1, 0.2, 0.3, 0.4]
    reference_dimension = 4
    reference_mean = reference_n_trials * np.array(reference_probabilities)
    return reference_n_trials, reference_probabilities, reference_dimension, reference_mean


@pytest.fixture(name="distribution")
def fixture_distribution(reference_data):
    """Distribution fixture."""
    n_trials, probabilities, _, _ = reference_data
    return Multinomial(n_trials, probabilities)


@pytest.fixture(name="distribution_fcc")
def fixture_distribution_fcc(reference_data):
    """Distribution fixture."""
    reference_n_trials, reference_probabilities, _, _ = reference_data
    return Multinomial(probabilities=reference_probabilities, n_trials=reference_n_trials)


# pylint: disable=duplicate-code
@pytest.fixture(name="distributions", params=["init", "fcc"])
def fixture_distributions(request, distribution, distribution_fcc):
    """Distributions fixture."""
    if request.param == "init":
        return distribution
    return distribution_fcc


# pylint: enable=duplicate-code


def test_init_probabilities(reference_data, distributions):
    """Test probabilities in the init method."""
    _, reference_probabilities, _, _ = reference_data

    np.testing.assert_allclose(reference_probabilities, distributions.probabilities)


def test_init_sample_space(reference_data, distributions):
    """Test sample_space in the init method."""
    reference_n_trials, _, reference_dimension, _ = reference_data
    reference_sample_space = np.ones((reference_dimension, 1)) * reference_n_trials

    np.testing.assert_allclose(reference_sample_space, distributions.sample_space)


def test_init_dimension(reference_data, distributions):
    """Test dimension in the init method."""
    _, _, reference_dimension, _ = reference_data

    np.testing.assert_allclose(reference_dimension, distributions.dimension)


def test_init_mean(reference_data, distributions):
    """Test mean in the init method."""
    _, _, _, reference_mean = reference_data

    np.testing.assert_allclose(reference_mean, distributions.mean)


def test_init_covariance(reference_data, distributions):
    """Test covariance in the init method."""
    reference_n_trials, reference_probabilities, _, _ = reference_data

    reference_covariance = reference_n_trials * (
        np.diag(reference_probabilities)
        - np.outer(reference_probabilities, reference_probabilities)
    )

    np.testing.assert_allclose(reference_covariance, distributions.covariance)


def test_draw(reference_data, distribution):
    """Test draw method."""
    reference_n_trials, _, reference_dimension, _ = reference_data
    n_draws = 5
    samples = distribution.draw(n_draws)

    # assert if the shape is correct
    assert samples.shape == (n_draws, reference_dimension)

    # assert if the samples sum up to the number of trials
    np.testing.assert_equal(reference_n_trials, np.sum(samples, axis=1))


def test_pdf(distribution):
    """Test pdf."""
    locations = np.array([[1, 2, 3, 4], [10, 0, 0, 0]])
    reference_pdf = [0.03483648, 1e-10]
    np.testing.assert_allclose(reference_pdf, distribution.pdf(locations))


def test_logpdf(distribution):
    """Test logpdf."""
    locations = np.array([[1, 2, 3, 4], [10, 0, 0, 0]])
    reference_logpdf = np.log([0.03483648, 1e-10])
    np.testing.assert_allclose(reference_logpdf, distribution.logpdf(locations))


def test_cdf(distribution):
    """Test if cdf raises value error."""
    with pytest.raises(ValueError, match="Method does not support multivariate distributions!"):
        distribution.cdf(1)


def test_ppf(distribution):
    """Test if ppf raises value error."""
    with pytest.raises(ValueError, match="Method does not support multivariate distributions!"):
        distribution.ppf(1)
