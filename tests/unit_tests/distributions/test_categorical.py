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
"""Test categorical distributions."""

import numpy as np
import pytest

from queens.distributions.categorical import Categorical


@pytest.fixture(name="reference_distribution_data")
def fixture_reference_distribution_data():
    """Data for the categorical distribution."""
    weights = [1, 3]
    probabilities = [1 / 4, 3 / 4]
    categories = ["aa", 5]
    return weights, categories, probabilities


@pytest.fixture(name="categorical_distribution")
def fixture_categorical_distribution(reference_distribution_data):
    """Categorical distribution fixture."""
    weights, categories, _ = reference_distribution_data

    categorical_distribution = Categorical(weights, categories)
    return categorical_distribution


def test_probabilities(reference_distribution_data, categorical_distribution):
    """Test if probabilities are set correctly."""
    _, _, probabilities = reference_distribution_data
    np.testing.assert_almost_equal(probabilities, categorical_distribution.probabilities)


def test_init(reference_distribution_data, categorical_distribution):
    """Test from config create."""
    weights, categories, _ = reference_distribution_data
    distribution = Categorical(probabilities=weights, categories=categories)
    np.testing.assert_equal(categorical_distribution.probabilities, distribution.probabilities)
    np.testing.assert_equal(categorical_distribution.categories, distribution.categories)


def test_pmf(reference_distribution_data, categorical_distribution):
    """Test pmf."""
    _, _, probabilities = reference_distribution_data
    sample_location = np.array([[5], ["aa"]], dtype=object)
    np.testing.assert_allclose(probabilities[::-1], categorical_distribution.pdf(sample_location))


def test_logpmf(reference_distribution_data, categorical_distribution):
    """Test logpmf."""
    _, _, probabilities = reference_distribution_data
    sample_location = np.array([[5], ["aa"]], dtype=object)
    np.testing.assert_allclose(
        np.log(probabilities[::-1]), categorical_distribution.logpdf(sample_location)
    )


def test_draw(mocker, reference_distribution_data, categorical_distribution):
    """Test drawing."""
    _, categories, _ = reference_distribution_data
    first_category_samples = 2
    second_category_samples = 3
    mocker.patch(
        "queens.distributions.categorical.np.random.multinomial",
        return_value=np.array([first_category_samples, second_category_samples]),
    )
    mocker.patch(
        "queens.distributions.categorical.np.random.shuffle",
    )
    reference_samples = [categories[0]] * first_category_samples
    reference_samples.extend([categories[1]] * second_category_samples)
    reference_samples = np.array(reference_samples, dtype=object).reshape(-1, 1)
    np.testing.assert_equal(reference_samples, categorical_distribution.draw(5))
