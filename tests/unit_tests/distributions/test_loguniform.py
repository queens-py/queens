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
"""Tests for the log-uniform distribution."""

import numpy as np
import pytest
import scipy.stats

from queens.distributions.loguniform import LogUniform


def test_one_dimensional_distribution() -> None:
    """One-dimensional values agree with SciPy, also outside bounds."""
    lower_bound = 0.5
    upper_bound = 4.0
    distribution = LogUniform(lower_bound, upper_bound)
    scipy_reference = scipy.stats.loguniform
    points = np.array([-1.0, 0.0, 0.5, 1.0, 2.0, 4.0, 5.0])
    quantiles = np.array([-0.1, 0.0, 0.25, 0.5, 1.0, 1.1])

    assert distribution.dimension == 1
    np.testing.assert_array_equal(distribution.lower_bound, [lower_bound])
    np.testing.assert_array_equal(distribution.upper_bound, [upper_bound])
    np.testing.assert_allclose(distribution.mean, [scipy_reference.mean(lower_bound, upper_bound)])
    np.testing.assert_allclose(
        distribution.covariance, [[scipy_reference.var(lower_bound, upper_bound)]]
    )
    np.testing.assert_allclose(
        distribution.pdf(points), scipy_reference.pdf(points, lower_bound, upper_bound)
    )
    np.testing.assert_allclose(
        distribution.logpdf(points),
        scipy_reference.logpdf(points, lower_bound, upper_bound),
    )
    np.testing.assert_allclose(
        distribution.cdf(points), scipy_reference.cdf(points, lower_bound, upper_bound)
    )
    np.testing.assert_allclose(
        distribution.ppf(quantiles),
        scipy_reference.ppf(quantiles, lower_bound, upper_bound),
    )


def test_two_dimensional_distribution() -> None:
    """Independent coordinates give the expected joint density and CDF."""
    lower_bounds = np.array([0.5, 0.25])
    upper_bounds = np.array([4.0, 2.0])
    distribution = LogUniform(lower_bounds, upper_bounds)
    scipy_reference = scipy.stats.loguniform
    points = np.array([[0.1, 0.5], [0.5, 0.25], [1.0, 1.0], [4.0, 2.0], [5.0, 1.0], [1.0, 3.0]])

    assert distribution.dimension == 2
    np.testing.assert_array_equal(distribution.lower_bound, lower_bounds)
    np.testing.assert_array_equal(distribution.upper_bound, upper_bounds)
    np.testing.assert_allclose(distribution.mean, scipy_reference.mean(lower_bounds, upper_bounds))
    np.testing.assert_allclose(
        distribution.covariance,
        np.diag(scipy_reference.var(lower_bounds, upper_bounds)),
    )
    np.testing.assert_allclose(
        distribution.pdf(points),
        np.prod(scipy_reference.pdf(points, lower_bounds, upper_bounds), axis=1),
    )
    np.testing.assert_allclose(
        distribution.logpdf(points),
        np.sum(scipy_reference.logpdf(points, lower_bounds, upper_bounds), axis=1),
    )
    np.testing.assert_allclose(
        distribution.cdf(points),
        np.prod(scipy_reference.cdf(points, lower_bounds, upper_bounds), axis=1),
    )

    with pytest.raises(ValueError, match="does not support multivariate"):
        distribution.ppf(np.array([[0.5, 0.5]]))


@pytest.mark.parametrize("lower, upper", [(1.0, 0.5), (-1.0, 4.0), (0.0, 4.0)])
def test_invalid_bounds(lower: float, upper: float) -> None:
    """Bounds must be positive and ordered."""
    with pytest.raises(ValueError):
        LogUniform(lower, upper)


@pytest.mark.parametrize(
    "lower, upper",
    [([0.5], [4.0]), ([0.5, 0.25], [4.0, 2.0])],
)
def test_draw(lower: list[float], upper: list[float]) -> None:
    """Draws have right shape, expected quantiles and stay within bounds."""
    np.random.seed(42)
    num_draws = 10_000
    draws = LogUniform(lower, upper).draw(num_draws)
    lower_bounds = np.array(lower)
    upper_bounds = np.array(upper)

    assert draws.shape == (num_draws, len(lower))
    assert np.all((draws >= lower_bounds) & (draws <= upper_bounds))

    # Each SciPy quantile should have roughly that fraction of draws below it.
    for quantile in (0.1, 0.25, 0.5, 0.75, 0.9):
        threshold = scipy.stats.loguniform.ppf(quantile, lower_bounds, upper_bounds)
        fraction_below = np.mean(draws < threshold, axis=0)
        np.testing.assert_allclose(fraction_below, quantile, atol=0.03)


def test_gradient() -> None:
    """The gradient agrees with a numerical derivative of SciPy's log-PDF."""
    lower = np.array([0.5, 0.25])
    upper = np.array([4.0, 2.0])
    distribution = LogUniform(lower, upper)
    scipy_reference = scipy.stats.loguniform
    points = np.array([[0.5, 0.25], [1.0, 1.0], [4.0, 2.0]])
    offsets = np.eye(2) * 1e-6
    expected = (
        np.sum(scipy_reference.logpdf(points[1] + offsets, lower, upper), axis=1)
        - np.sum(scipy_reference.logpdf(points[1] - offsets, lower, upper), axis=1)
    ) / (2e-6)

    np.testing.assert_allclose(distribution.grad_logpdf(points)[1], expected, rtol=1e-6)

    point_outside = np.array([[0.0, 1.0]])
    assert np.isnan(distribution.grad_logpdf(point_outside)).all()
