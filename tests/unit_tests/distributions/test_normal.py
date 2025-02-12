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
"""Test-module for normal distribution."""

import jax.numpy as jnp
import numpy as np
import pytest
import scipy.stats
from jax import grad

from queens.distributions.normal import Normal


# ------------- univariate --------------
@pytest.fixture(name="sample_pos_1d", params=[-2.0, [-1.0, 0.0, 1.0, 2.0]])
def fixture_sample_pos_1d(request):
    """Sample position to be evaluated."""
    return np.array(request.param)


@pytest.fixture(name="mean_1d", scope="module")
def fixture_mean_1d():
    """A possible scalar mean value."""
    return 1.0


@pytest.fixture(name="covariance_1d", scope="module")
def fixture_covariance_1d():
    """A possible scalar variance value."""
    return 2.0


@pytest.fixture(name="normal_1d", scope="module")
def fixture_normal_1d(mean_1d, covariance_1d):
    """A 1d normal distribution."""
    return Normal(mean=mean_1d, covariance=covariance_1d)


@pytest.fixture(name="uncorrelated_vector_1d", scope="module")
def fixture_uncorrelated_vector_1d(num_draws):
    """A vector of uncorrelated samples from standard normal distribution."""
    vec = [[1.0]]
    return np.tile(vec, num_draws)


# ------------- multivariate --------------
@pytest.fixture(
    name="sample_pos_3d", params=[[-2.0, -1.0, 0.0], [[-1.0, -3.0, 1.0], [-1.0, -1.0, -1.0]]]
)
def fixture_sample_pos_3d(request):
    """Sample position to be evaluated."""
    return np.array(request.param)


@pytest.fixture(name="mean_3d", scope="module")
def fixture_mean_3d():
    """A possible mean vector."""
    return np.array([0.0, -1.0, 2.0])


@pytest.fixture(name="low_chol_3d", scope="module")
def fixture_low_chol_3d():
    """Lower triangular matrix of a Cholesky decomposition."""
    return np.array([[1.0, 0.0, 0.0], [0.1, 2.0, 0.0], [1.0, 0.8, 3.0]])


@pytest.fixture(name="covariance_3d", scope="module")
def fixture_covariance_3d(low_chol_3d):
    """Recompose matrix based on given Cholesky decomposition."""
    return np.dot(low_chol_3d, low_chol_3d.T)


@pytest.fixture(name="normal_3d", scope="module")
def fixture_normal_3d(mean_3d, covariance_3d):
    """A multivariate normal distribution."""
    return Normal(mean=mean_3d, covariance=covariance_3d)


@pytest.fixture(name="num_draws", scope="module", params=[1, 4])
def fixture_num_draws(request):
    """Number of samples to draw from distribution."""
    return request.param


@pytest.fixture(name="uncorrelated_vector_3d", scope="module")
def fixture_uncorrelated_vector_3d(num_draws):
    """A vector of uncorrelated samples from standard normal distribution."""
    vec = [[1.0], [-2.0], [3.0]]
    return np.tile(vec, num_draws)


# -----------------------------------------------------------------------
# ---------------------------- TESTS ------------------------------------
# -----------------------------------------------------------------------


# ------------- univariate --------------
def test_init_normal_1d(normal_1d, mean_1d, covariance_1d):
    """Test init method of Normal Distribution class."""
    assert normal_1d.dimension == 1
    np.testing.assert_equal(normal_1d.mean, np.array(mean_1d).reshape(1))
    np.testing.assert_equal(normal_1d.covariance, np.array(covariance_1d).reshape(1, 1))


def test_init_normal_1d_incovariance(mean_1d, covariance_1d):
    """Test init method of Normal Distribution class."""
    with pytest.raises(np.linalg.LinAlgError, match=r"Cholesky decomposition failed *"):
        Normal(mean=mean_1d, covariance=-covariance_1d)


def test_cdf_normal_1d(normal_1d, mean_1d, covariance_1d, sample_pos_1d):
    """Test cdf method of Normal distribution class."""
    std = np.sqrt(covariance_1d)
    ref_sol = scipy.stats.norm.cdf(sample_pos_1d, loc=mean_1d, scale=std).reshape(-1)
    np.testing.assert_allclose(normal_1d.cdf(sample_pos_1d), ref_sol)


def test_draw_normal_1d(normal_1d, mean_1d, covariance_1d, uncorrelated_vector_1d, mocker):
    """Test the draw method of normal distribution."""
    mocker.patch("numpy.random.randn", return_value=uncorrelated_vector_1d)
    draw = normal_1d.draw()
    ref_sol = mean_1d + covariance_1d ** (1 / 2) * uncorrelated_vector_1d.T
    np.testing.assert_equal(draw, ref_sol)


def test_logpdf_normal_1d(normal_1d, mean_1d, covariance_1d, sample_pos_1d):
    """Test pdf method of Normal distribution class."""
    std = np.sqrt(covariance_1d)
    ref_sol = scipy.stats.norm.logpdf(sample_pos_1d, loc=mean_1d, scale=std).reshape(-1)
    np.testing.assert_allclose(normal_1d.logpdf(sample_pos_1d), ref_sol)


def test_grad_logpdf_normal_1d(normal_1d, sample_pos_1d):
    """Test pdf method of normal distribution class."""
    sample_pos_1d = sample_pos_1d.reshape(-1, 1)
    grad_logpdf_jax = grad(logpdf, argnums=0)
    ref_sol_list = []
    for sample in sample_pos_1d:
        ref_sol_list.append(
            grad_logpdf_jax(sample, normal_1d.logpdf_const, normal_1d.mean, normal_1d.precision)
        )
    np.testing.assert_allclose(normal_1d.grad_logpdf(sample_pos_1d), np.array(ref_sol_list))


def test_pdf_normal_1d(normal_1d, mean_1d, covariance_1d, sample_pos_1d):
    """Test pdf method of Normal distribution class."""
    std = np.sqrt(covariance_1d)
    ref_sol = scipy.stats.norm.pdf(sample_pos_1d, loc=mean_1d, scale=std).reshape(-1)
    np.testing.assert_allclose(normal_1d.pdf(sample_pos_1d), ref_sol)


def test_ppf_normal_1d(normal_1d, mean_1d, covariance_1d):
    """Test ppf method of Normal distribution class."""
    std = np.sqrt(covariance_1d)
    quantile = 0.5
    ref_sol = scipy.stats.norm.ppf(quantile, loc=mean_1d, scale=std).reshape(-1)
    np.testing.assert_allclose(normal_1d.ppf(quantile), ref_sol)


# ------------- multivariate --------------
def test_init_normal_3d(normal_3d, mean_3d, covariance_3d):
    """Test init method of Normal Distribution class."""
    assert normal_3d.dimension == 3
    np.testing.assert_equal(normal_3d.mean, mean_3d.reshape(3))
    np.testing.assert_equal(normal_3d.covariance, covariance_3d.reshape(3, 3))


def test_cdf_normal_3d(normal_3d, mean_3d, covariance_3d, sample_pos_3d):
    """Test cdf method of Normal distribution class."""
    sample_pos_3d = sample_pos_3d.reshape(-1, 3)
    ref_sol = scipy.stats.multivariate_normal.cdf(
        sample_pos_3d, mean=mean_3d, cov=covariance_3d
    ).reshape(-1)
    # high tolerance due to numerical errors
    np.testing.assert_allclose(normal_3d.cdf(sample_pos_3d), ref_sol, rtol=1e-3)


def test_draw_normal_3d(normal_3d, mean_3d, low_chol_3d, uncorrelated_vector_3d, mocker):
    """Test the draw method of normal distribution."""
    mocker.patch("numpy.random.randn", return_value=uncorrelated_vector_3d)
    draw = normal_3d.draw()
    ref_sol = mean_3d + np.dot(low_chol_3d, uncorrelated_vector_3d).T
    np.testing.assert_equal(draw, ref_sol)


def test_logpdf_normal_3d(normal_3d, mean_3d, covariance_3d, sample_pos_3d):
    """Test pdf method of Normal distribution class."""
    sample_pos_3d = sample_pos_3d.reshape(-1, 3)
    ref_sol = scipy.stats.multivariate_normal.logpdf(sample_pos_3d, mean=mean_3d, cov=covariance_3d)
    np.testing.assert_allclose(normal_3d.logpdf(sample_pos_3d), ref_sol)


def test_grad_logpdf_normal_3d(normal_3d, sample_pos_3d):
    """Test pdf method of normal distribution class."""
    sample_pos_3d = sample_pos_3d.reshape(-1, 3)
    grad_logpdf_jax = grad(logpdf, argnums=0)
    ref_sol_list = []
    for sample in sample_pos_3d:
        ref_sol_list.append(
            grad_logpdf_jax(sample, normal_3d.logpdf_const, normal_3d.mean, normal_3d.precision)
        )
    np.testing.assert_allclose(
        normal_3d.grad_logpdf(sample_pos_3d), np.array(ref_sol_list), rtol=1e-6
    )


def test_pdf_normal_3d(normal_3d, mean_3d, covariance_3d, sample_pos_3d):
    """Test pdf method of Normal distribution class."""
    sample_pos_3d = sample_pos_3d.reshape(-1, 3)
    ref_sol = scipy.stats.multivariate_normal.pdf(sample_pos_3d, mean=mean_3d, cov=covariance_3d)
    np.testing.assert_allclose(normal_3d.pdf(sample_pos_3d), ref_sol)


def test_ppf_normal_3d(normal_3d):
    """Test ppf method of Normal distribution class."""
    with pytest.raises(ValueError, match="Method does not support multivariate distributions!"):
        normal_3d.ppf(np.zeros(2))


def test_init_normal_wrong_dimension(mean_3d):
    """Test ValueError of init method of Normal Distribution class."""
    covariance = np.array([[[1.0, 0.1], [1.0, 0.1]], [[0.2, 2.0], [0.2, 2.0]]])
    with pytest.raises(ValueError, match=r"Provided covariance is not a matrix.*"):
        Normal(mean=mean_3d, covariance=covariance)


def test_init_normal_not_quadratic(mean_3d):
    """Test ValueError of init method of Normal Distribution class."""
    covariance = np.array([[1.0, 0.1], [0.2, 2.0], [3.0, 0.3]])
    with pytest.raises(ValueError, match=r"Provided covariance matrix is not quadratic.*"):
        Normal(mean=mean_3d, covariance=covariance)


def test_init_normal_not_symmetric(mean_3d):
    """Test ValueError of init method of Normal Distribution class."""
    covariance = np.array([[1.0, 0.1], [0.2, 2.0]])
    with pytest.raises(ValueError, match=r"Provided covariance matrix is not symmetric.*"):
        Normal(mean=mean_3d, covariance=covariance)


def logpdf(x, logpdf_const, mean, precision):
    """Log pdf of normal distribution.

    Args:
        x (np.ndarray): Positions at which the log pdf is evaluated
        logpdf_const (float): Constant for evaluation of log pdf
        mean (np.ndarray): mean of the normal distribution
        precision (np.ndarray): Precision matrix of the normal distribution

    Returns:
        logpdf (np.ndarray): log pdf at evaluated positions
    """
    dist = jnp.array(x - mean).reshape(1, -1)
    return logpdf_const - 0.5 * (jnp.dot(jnp.dot(dist, precision), dist.T)).squeeze()
