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
"""Test-module for lognormal distribution."""

import jax.numpy as jnp
import numpy as np
import pytest
import scipy.stats
from jax import grad

from queens.distributions.lognormal import LogNormal


# ------------- univariate --------------
@pytest.fixture(name="sample_pos_1d", params=[1.0, [5.0, 0.0, -1.0, 2.0]])
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


@pytest.fixture(name="lognormal_1d", scope="module")
def fixture_lognormal_1d(mean_1d, covariance_1d):
    """A 1d lognormal distribution."""
    return LogNormal(normal_mean=mean_1d, normal_covariance=covariance_1d)


@pytest.fixture(name="uncorrelated_vector_1d", scope="module")
def fixture_uncorrelated_vector_1d(num_draws):
    """A vector of uncorrelated samples from standard normal distribution."""
    vec = [[1.0]]
    return np.tile(vec, num_draws)


# ------------- multivariate --------------
@pytest.fixture(
    name="sample_pos_2d",
    params=[[1.0, 2.0], [[2.0, -1.0], [1.0, 0.5], [2.0, -1.0], [0.0, -1.0], [-2.0, -1.0]]],
)
def fixture_sample_pos_2d(request):
    """Sample position to be evaluated."""
    return np.array(request.param)


@pytest.fixture(name="mean_2d", scope="module")
def fixture_mean_2d():
    """A possible mean vector."""
    return np.array([1.0, -2.0])


@pytest.fixture(name="covariance_2d", scope="module")
def fixture_covariance_2d():
    """Recompose matrix based on given Cholesky decomposition."""
    return np.array([[1.0, 0.0], [0.0, 2.0]])


@pytest.fixture(name="lognormal_2d", scope="module")
def fixture_lognormal_2d(mean_2d, covariance_2d):
    """A multivariate lognormal distribution."""
    return LogNormal(normal_mean=mean_2d, normal_covariance=covariance_2d)


@pytest.fixture(name="num_draws", scope="module", params=[1, 4])
def fixture_num_draws(request):
    """Number of samples to draw from distribution."""
    return request.param


@pytest.fixture(name="uncorrelated_vector_2d", scope="module")
def fixture_uncorrelated_vector_2d(num_draws):
    """A vector of uncorrelated samples from standard normal distribution."""
    vec = [[1.0], [-2.0]]
    return np.tile(vec, num_draws)


# -----------------------------------------------------------------------
# ---------------------------- TESTS ------------------------------------
# -----------------------------------------------------------------------


# ------------- univariate --------------
def test_init_lognormal_1d(lognormal_1d, mean_1d, covariance_1d):
    """Test init method of LogNormal Distribution class."""
    std = np.sqrt(covariance_1d)
    mean_ref = scipy.stats.lognorm.mean(scale=np.exp(mean_1d), s=std).reshape(1)
    var_ref = scipy.stats.lognorm.var(scale=np.exp(mean_1d), s=std).reshape(1, 1)

    assert lognormal_1d.dimension == 1
    np.testing.assert_equal(lognormal_1d.normal_distribution.mean, np.array(mean_1d).reshape(1))
    np.testing.assert_equal(
        lognormal_1d.normal_distribution.covariance, np.array(covariance_1d).reshape(1, 1)
    )
    np.testing.assert_allclose(lognormal_1d.mean, mean_ref)
    np.testing.assert_allclose(lognormal_1d.covariance, var_ref)


def test_init_lognormal_1d_incovariance(mean_1d, covariance_1d):
    """Test init method of LogNormal Distribution class."""
    with pytest.raises(np.linalg.LinAlgError, match=r"Cholesky decomposition failed *"):
        LogNormal(normal_mean=mean_1d, normal_covariance=-covariance_1d)


def test_cdf_lognormal_1d(lognormal_1d, mean_1d, covariance_1d, sample_pos_1d):
    """Test cdf method of LogNormal distribution class."""
    std = np.sqrt(covariance_1d)
    ref_sol = scipy.stats.lognorm.cdf(sample_pos_1d, scale=np.exp(mean_1d), s=std).reshape(-1)
    np.testing.assert_allclose(lognormal_1d.cdf(sample_pos_1d), ref_sol)


def test_draw_lognormal_1d(lognormal_1d, mean_1d, covariance_1d, uncorrelated_vector_1d, mocker):
    """Test the draw method of lognormal distribution."""
    mocker.patch("numpy.random.randn", return_value=uncorrelated_vector_1d)
    draw = lognormal_1d.draw()
    ref_sol = np.exp(mean_1d + covariance_1d ** (1 / 2) * uncorrelated_vector_1d.T).reshape(-1, 1)
    np.testing.assert_equal(draw, ref_sol)


def test_logpdf_lognormal_1d(lognormal_1d, mean_1d, covariance_1d, sample_pos_1d):
    """Test logpdf method of LogNormal distribution class."""
    std = np.sqrt(covariance_1d)
    ref_sol = scipy.stats.lognorm.logpdf(sample_pos_1d, scale=np.exp(mean_1d), s=std).reshape(-1)
    ref_sol[ref_sol == -np.inf] = np.nan  # Queens Log Normal is not defined for <=0.
    np.testing.assert_allclose(lognormal_1d.logpdf(sample_pos_1d), ref_sol)


def test_grad_logpdf_lognormal_1d(lognormal_1d, sample_pos_1d):
    """Test *grad_logpdf* method of LogNormal distribution class."""
    sample_pos_1d = sample_pos_1d.reshape(-1, 1)  # pylint: disable=duplicate-code
    grad_logpdf_jax = grad(logpdf, argnums=0)
    ref_sol_list = []
    for sample in sample_pos_1d:
        ref_sol_list.append(
            grad_logpdf_jax(
                sample,
                lognormal_1d.normal_distribution.logpdf_const,
                lognormal_1d.normal_distribution.mean,
                lognormal_1d.normal_distribution.precision,
            )
        )
    ref_sol = np.array(ref_sol_list)
    ref_sol[ref_sol == np.inf] = np.nan
    np.testing.assert_allclose(lognormal_1d.grad_logpdf(sample_pos_1d), ref_sol)


def test_pdf_lognormal_1d(lognormal_1d, mean_1d, covariance_1d, sample_pos_1d):
    """Test pdf method of LogNormal distribution class."""
    std = np.sqrt(covariance_1d)
    ref_sol = scipy.stats.lognorm.pdf(sample_pos_1d, scale=np.exp(mean_1d), s=std).reshape(-1)
    ref_sol[ref_sol == 0] = np.nan  # Queens Log Normal is not defined for <=0.
    np.testing.assert_allclose(lognormal_1d.pdf(sample_pos_1d), ref_sol)


def test_ppf_lognormal_1d(lognormal_1d, mean_1d, covariance_1d):
    """Test ppf method of LogNormal distribution class."""
    std = np.sqrt(covariance_1d)
    quantile = 0.5
    ref_sol = scipy.stats.lognorm.ppf(quantile, scale=np.exp(mean_1d), s=std).reshape(-1)
    np.testing.assert_allclose(lognormal_1d.ppf(quantile), ref_sol)


# ------------- multivariate --------------
def test_init_lognormal_2d(lognormal_2d, mean_2d, covariance_2d):
    """Test init method of LogNormal Distribution class."""
    std = np.diag(np.sqrt(covariance_2d))
    mean_ref = np.array(
        [
            scipy.stats.lognorm.mean(scale=np.exp(mean_2d[0]), s=std[0]),
            scipy.stats.lognorm.mean(scale=np.exp(mean_2d[1]), s=std[1]),
        ]
    )
    var_ref = np.diag(
        [
            scipy.stats.lognorm.var(scale=np.exp(mean_2d[0]), s=std[0]),
            scipy.stats.lognorm.var(scale=np.exp(mean_2d[1]), s=std[1]),
        ]
    )

    assert lognormal_2d.dimension == 2
    np.testing.assert_equal(lognormal_2d.normal_distribution.mean, mean_2d)
    np.testing.assert_equal(lognormal_2d.normal_distribution.covariance, covariance_2d)
    np.testing.assert_allclose(lognormal_2d.mean, mean_ref)
    np.testing.assert_allclose(lognormal_2d.covariance, var_ref)


def test_cdf_lognormal_2d(lognormal_2d, mean_2d, covariance_2d, sample_pos_2d):
    """Test cdf method of LogNormal distribution class."""
    sample_pos_2d = sample_pos_2d.reshape(-1, 2)
    std = np.diag(np.sqrt(covariance_2d))
    ref_sol = scipy.stats.lognorm.cdf(
        sample_pos_2d[:, 0], scale=np.exp(mean_2d[0]), s=std[0]
    ) * scipy.stats.lognorm.cdf(sample_pos_2d[:, 1], scale=np.exp(mean_2d[1]), s=std[1])
    ref_sol[ref_sol == 0] = np.nan  # Queens Log Normal is not defined for <=0.
    np.testing.assert_allclose(lognormal_2d.cdf(sample_pos_2d), ref_sol)


def test_draw_lognormal_2d(lognormal_2d, mean_2d, covariance_2d, uncorrelated_vector_2d, mocker):
    """Test the draw method of lognormal distribution."""
    mocker.patch("numpy.random.randn", return_value=uncorrelated_vector_2d)
    draw = lognormal_2d.draw()
    ref_sol = np.exp(mean_2d + np.dot(np.sqrt(covariance_2d), uncorrelated_vector_2d).T)
    np.testing.assert_equal(draw, ref_sol)


def test_logpdf_lognormal_2d(lognormal_2d, mean_2d, covariance_2d, sample_pos_2d):
    """Test logpdf method of LogNormal distribution class."""
    sample_pos_2d = sample_pos_2d.reshape(-1, 2)
    std = np.diag(np.sqrt(covariance_2d))
    ref_sol = scipy.stats.lognorm.logpdf(
        sample_pos_2d[:, 0], scale=np.exp(mean_2d[0]), s=std[0]
    ) + scipy.stats.lognorm.logpdf(sample_pos_2d[:, 1], scale=np.exp(mean_2d[1]), s=std[1])
    ref_sol[ref_sol == -np.inf] = np.nan  # Queens Log Normal is not defined for <=0.
    np.testing.assert_allclose(lognormal_2d.logpdf(sample_pos_2d), ref_sol)


def test_grad_logpdf_lognormal_2d(lognormal_2d, sample_pos_2d):
    """Test *grad_logpdf* method of LogNormal distribution class."""
    sample_pos_2d = sample_pos_2d.reshape(-1, 2)
    grad_logpdf_jax = grad(logpdf, argnums=0)
    ref_sol_list = []
    for sample in sample_pos_2d:
        ref_sol_list.append(
            grad_logpdf_jax(
                sample,
                lognormal_2d.normal_distribution.logpdf_const,
                lognormal_2d.normal_distribution.mean,
                lognormal_2d.normal_distribution.precision,
            )
        )
    np.testing.assert_allclose(lognormal_2d.grad_logpdf(sample_pos_2d), np.array(ref_sol_list))


def test_pdf_lognormal_2d(lognormal_2d, mean_2d, covariance_2d, sample_pos_2d):
    """Test pdf method of LogNormal distribution class."""
    sample_pos_2d = sample_pos_2d.reshape(-1, 2)
    std = np.diag(np.sqrt(covariance_2d))
    ref_sol = scipy.stats.lognorm.pdf(
        sample_pos_2d[:, 0], scale=np.exp(mean_2d[0]), s=std[0]
    ) * scipy.stats.lognorm.pdf(sample_pos_2d[:, 1], scale=np.exp(mean_2d[1]), s=std[1])
    ref_sol[ref_sol == 0] = np.nan  # Queens Log Normal is not defined for <=0.
    np.testing.assert_allclose(lognormal_2d.pdf(sample_pos_2d), ref_sol)


def test_ppf_lognormal_2d(lognormal_2d):
    """Test ppf method of LogNormal distribution class."""
    with pytest.raises(ValueError, match="Method does not support multivariate distributions!"):
        lognormal_2d.ppf(np.zeros(2))


def test_init_lognormal_wrong_dimension():
    """Test ValueError of init method of LogNormal Distribution class."""
    covariance = np.array([[[1.0, 0.1], [1.0, 0.1]], [[0.2, 2.0], [0.2, 2.0]]])
    with pytest.raises(ValueError, match=r"Provided covariance is not a matrix.*"):
        LogNormal(normal_mean=np.zeros(3), normal_covariance=covariance)


def test_init_lognormal_not_quadratic():
    """Test ValueError of init method of LogNormal Distribution class."""
    covariance = np.array([[1.0, 0.1], [0.2, 2.0], [3.0, 0.3]])
    with pytest.raises(ValueError, match=r"Provided covariance matrix is not quadratic.*"):
        LogNormal(normal_mean=np.zeros(3), normal_covariance=covariance)


def test_init_lognormal_not_symmetric():
    """Test ValueError of init method of LogNormal Distribution class."""
    covariance = np.array([[1.0, 0.1], [0.2, 2.0]])
    with pytest.raises(ValueError, match=r"Provided covariance matrix is not symmetric.*"):
        LogNormal(normal_mean=np.zeros(2), normal_covariance=covariance)


def test_init_lognormal_nonmatching_dimension():
    """Nonmatching dimension of mean and covariance raises ValueError during.

    init of LogNormal.
    """
    covariance = np.array([[1.0, 0.0], [0.0, 2.0]])
    with pytest.raises(ValueError, match=r"Dimension of mean vector and covariance matrix do not*"):
        LogNormal(normal_mean=np.zeros(3), normal_covariance=covariance)


def logpdf(x, logpdf_const, normal_mean, precision):
    """Log pdf of lognormal (using jax.numpy).

    Args:
        x (np.ndarray): Positions at which the log pdf is evaluated
        logpdf_const (float): Constant for evaluation of log pdf
        normal_mean (np.ndarray): mean of the underlying normal distribution
        precision (np.ndarray): Precision matrix of underlying normal distribution

    Returns:
        logpdf (np.ndarray): log pdf at evaluated positions
    """
    log_x = jnp.log(x).reshape(1, -1)
    dist = log_x - normal_mean
    return (
        logpdf_const - jnp.sum(log_x) - 0.5 * (jnp.dot(jnp.dot(dist, precision), dist.T)).squeeze()
    )
