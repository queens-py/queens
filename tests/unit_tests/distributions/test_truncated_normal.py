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
"""Test-module for truncated normal distribution."""

import numpy as np
import pytest
import scipy.stats

from queens.distributions.truncated_normal import TruncatedNormal


@pytest.fixture(name="sample_pos", params=[0.25, [0.1, 0.25, 0.4, 0.5]])
def fixture_sample_pos(request):
    """Sample position to be evaluated."""
    return np.array(request.param)


@pytest.fixture(name="normal_mean", scope="module")
def fixture_normal_mean():
    """Mean of the underlying normal distribution."""
    return 0.3


@pytest.fixture(name="normal_std", scope="module")
def fixture_normal_std():
    """Standard deviation of the underlying normal distribution."""
    return 0.05


@pytest.fixture(name="lower_bound", scope="module")
def fixture_lower_bound():
    """Lower bound of the distribution."""
    return 0.1


@pytest.fixture(name="upper_bound", scope="module")
def fixture_upper_bound():
    """Upper bound of the distribution."""
    return 0.45


@pytest.fixture(name="truncated_normal", scope="module")
def fixture_truncated_normal(normal_mean, normal_std, lower_bound, upper_bound):
    """A truncated normal distribution."""
    return TruncatedNormal(
        mean=normal_mean, std=normal_std, lower_bound=lower_bound, upper_bound=upper_bound
    )


@pytest.fixture(name="scipy_reference", scope="module")
def fixture_scipy_reference(normal_mean, normal_std, lower_bound, upper_bound):
    """Reference scipy frozen truncnorm for comparison."""
    a = (lower_bound - normal_mean) / normal_std
    b = (upper_bound - normal_mean) / normal_std
    return scipy.stats.truncnorm(a, b, loc=normal_mean, scale=normal_std)


# -----------------------------------------------------------------------
# ---------------------------- TESTS ------------------------------------
# -----------------------------------------------------------------------


def test_init_truncated_normal(
    truncated_normal, normal_mean, normal_std, lower_bound, upper_bound, scipy_reference
):
    """Test init method of TruncatedNormal distribution class."""
    assert truncated_normal.dimension == 1
    np.testing.assert_equal(truncated_normal.mu, np.array(normal_mean).reshape(-1))
    np.testing.assert_equal(truncated_normal.sigma, np.array(normal_std).reshape(-1))
    np.testing.assert_equal(truncated_normal.lower_bound, np.array(lower_bound).reshape(-1))
    np.testing.assert_equal(truncated_normal.upper_bound, np.array(upper_bound).reshape(-1))
    np.testing.assert_equal(truncated_normal.mean, scipy_reference.mean())
    np.testing.assert_equal(truncated_normal.covariance, scipy_reference.var())


def test_init_truncated_normal_wrong_interval(normal_mean, normal_std):
    """Test init with lower bound greater than upper bound."""
    with pytest.raises(ValueError, match=r"Lower bound must be smaller than upper bound*"):
        TruncatedNormal(mean=normal_mean, std=normal_std, lower_bound=0.5, upper_bound=0.1)


def test_init_truncated_normal_negative_std(normal_mean, lower_bound, upper_bound):
    """Test init with non-positive std."""
    with pytest.raises(ValueError, match=r"The parameter \'std\' has to be positive.*"):
        TruncatedNormal(
            mean=normal_mean, std=-0.1, lower_bound=lower_bound, upper_bound=upper_bound
        )


def test_init_truncated_normal_multivariate(normal_std, lower_bound, upper_bound):
    """Test init with multivariate mean raises NotImplementedError."""
    with pytest.raises(NotImplementedError, match=r"Only one-dimensional*"):
        TruncatedNormal(
            mean=[0.3, 0.4], std=normal_std, lower_bound=lower_bound, upper_bound=upper_bound
        )


def test_cdf_truncated_normal(truncated_normal, sample_pos, scipy_reference):
    """Test cdf method of truncated normal distribution class."""
    ref_sol = scipy_reference.cdf(sample_pos).reshape(-1)
    np.testing.assert_equal(truncated_normal.cdf(sample_pos), ref_sol)


def test_draw_truncated_normal(truncated_normal, mocker):
    """Test the draw method of truncated normal distribution."""
    sample = np.asarray(0.3).reshape(1, 1)
    mocker.patch("scipy.stats._distn_infrastructure.rv_frozen.rvs", return_value=sample)
    draw = truncated_normal.draw()
    np.testing.assert_equal(draw, sample)


def test_logpdf_truncated_normal(truncated_normal, sample_pos, scipy_reference):
    """Test logpdf method of truncated normal distribution class."""
    ref_sol = scipy_reference.logpdf(sample_pos).reshape(-1)
    np.testing.assert_equal(truncated_normal.logpdf(sample_pos), ref_sol)


def test_pdf_truncated_normal(truncated_normal, sample_pos, scipy_reference):
    """Test pdf method of truncated normal distribution class."""
    ref_sol = scipy_reference.pdf(sample_pos).reshape(-1)
    np.testing.assert_equal(truncated_normal.pdf(sample_pos), ref_sol)


def test_grad_logpdf_truncated_normal(truncated_normal, normal_mean, normal_std, sample_pos):
    """Test grad_logpdf against analytical formula -(x - mean) / std**2."""
    x = np.asarray(sample_pos).reshape(-1)
    ref_sol = (normal_mean - x) / normal_std**2
    np.testing.assert_allclose(truncated_normal.grad_logpdf(sample_pos), ref_sol)


def test_grad_logpdf_matches_numerical(truncated_normal):
    """Test grad_logpdf matches numerical differentiation of logpdf."""
    eps = 1e-6
    for xi in (0.2, 0.3, 0.4):
        numerical = (
            truncated_normal.logpdf(np.array([xi + eps]))[0]
            - truncated_normal.logpdf(np.array([xi - eps]))[0]
        ) / (2 * eps)
        analytical = truncated_normal.grad_logpdf(np.array([xi]))[0]
        np.testing.assert_allclose(analytical, numerical, rtol=1e-5)


def test_ppf_truncated_normal(truncated_normal, scipy_reference):
    """Test ppf method of truncated normal distribution class."""
    quantile = 0.5
    ref_sol = scipy_reference.ppf(quantile).reshape(-1)
    np.testing.assert_equal(truncated_normal.ppf(quantile), ref_sol)
