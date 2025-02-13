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
"""Test-module for mean-field normal distribution."""

import jax.scipy.stats as jax_stats
import numpy as np
import pytest
import scipy
from jax import config, grad
from jax import numpy as jnp

from queens.distributions.mean_field_normal import MeanFieldNormal

config.update("jax_enable_x64", True)


@pytest.fixture(name="mean_field_normal")
def fixture_mean_field_normal():
    """Create dummy mean-field normal distribution."""
    mean = np.zeros(5).reshape(-1)
    variance = np.ones(5).reshape(-1)
    dimension = 5
    distribution = MeanFieldNormal(mean, variance, dimension)
    return distribution


@pytest.fixture(name="samples")
def fixture_samples():
    """Create two 5 dimensional samples."""
    np.random.seed(0)
    samples = np.random.normal(2, 3, 10).reshape(2, 5)
    return samples


# ---------------------- actual tests ----------------------
def test_from_config_create_distribution(mocker):
    """Test creation of normal distribution from config."""
    # test instantiation with vector and dimension
    distribution_options = {
        "mean": 3 * np.ones(5).reshape(-1),
        "variance": 7 * np.ones(5).reshape(-1),
        "dimension": 5,
    }

    # mock the get_check_array_dimension_and_reshape function
    mp1 = mocker.patch(
        "queens.distributions.mean_field_normal.MeanFieldNormal"
        ".get_check_array_dimension_and_reshape",
        return_value=distribution_options["mean"],
    )

    distribution = MeanFieldNormal(**distribution_options)
    np.testing.assert_array_equal(
        distribution.mean, np.ones(distribution_options["dimension"]) * distribution_options["mean"]
    )
    # note due to the mock the covariance should be the same as the mean
    np.testing.assert_array_equal(
        distribution.covariance,
        np.ones(distribution_options["dimension"]) * distribution_options["mean"],
    )
    assert distribution.dimension == distribution_options["dimension"]
    assert mp1.call_count == 2


def test_get_check_array_dimension_and_reshape():
    """Test get_check_array_dimension_and_reshape function."""
    # test instantiation with 1d array and  5 dimension
    test_array = np.array([1])
    dimension = 5
    out_array = MeanFieldNormal.get_check_array_dimension_and_reshape(test_array, dimension)
    np.testing.assert_array_equal(out_array, np.ones(dimension))

    # test instantiation with 5d array and  5 dimension
    test_array = np.array([[1, 3, 4, 5, 6]])
    dimension = 5
    out_array = MeanFieldNormal.get_check_array_dimension_and_reshape(test_array, dimension)
    np.testing.assert_array_equal(out_array, test_array)

    # test instantiation with vector and wrong dimension
    test_array = np.array([1, 3])
    dimension = 5
    with pytest.raises(ValueError):
        MeanFieldNormal.get_check_array_dimension_and_reshape(test_array, dimension)

    # test instantiation with float and dimension
    test_array = 1
    dimension = 5
    with pytest.raises(TypeError):
        MeanFieldNormal.get_check_array_dimension_and_reshape(test_array, dimension)


def test_init():
    """Test initialization of normal distribution."""
    mean = np.array([0.0])
    covariance = np.array([1.0])
    standard_deviation = np.sqrt(covariance)
    dimension = 1
    distribution = MeanFieldNormal(mean, covariance, dimension)
    assert distribution.mean == mean
    assert distribution.covariance == covariance
    assert distribution.dimension == dimension
    assert distribution.standard_deviation == standard_deviation


def test_update_variance(mean_field_normal):
    """Test update of variance."""
    # test update with correct dimension
    new_variance = np.ones(5).reshape(-1) * 3
    mean_field_normal.update_variance(new_variance)
    np.testing.assert_array_equal(mean_field_normal.covariance, new_variance)
    np.testing.assert_array_equal(mean_field_normal.standard_deviation, np.sqrt(new_variance))

    # test update with float --> should fail
    with pytest.raises(TypeError):
        new_variance = 1.0
        mean_field_normal.update_variance(new_variance)

    # test update with wrong dimension --> should fail
    with pytest.raises(ValueError):
        new_variance = np.ones(4).reshape(-1) * 3
        mean_field_normal.update_variance(new_variance)


def test_update_mean(mean_field_normal):
    """Test update of mean."""
    # test update with correct dimension
    new_mean = np.ones(5).reshape(-1) * 3
    mean_field_normal.update_mean(new_mean)
    np.testing.assert_array_equal(mean_field_normal.mean, new_mean)

    # test update with float --> should fail
    with pytest.raises(TypeError):
        new_mean = 1.0
        mean_field_normal.update_mean(new_mean)

    # test update with wrong dimension --> should fail
    with pytest.raises(ValueError):
        new_mean = np.ones(4).reshape(-1) * 3
        mean_field_normal.update_mean(new_mean)


def test_cdf(mean_field_normal, samples):
    """Test cdf."""
    cdf_vec = mean_field_normal.cdf(samples)

    # create cdf of scipy multivariate normal to compare
    benchmark_cdf = scipy.stats.multivariate_normal.cdf(
        samples, mean_field_normal.mean, np.diag(mean_field_normal.covariance)
    ).reshape(2, -1)

    # test
    np.testing.assert_array_almost_equal(cdf_vec, benchmark_cdf, decimal=6)


def test_draw(mean_field_normal, mocker):
    """Test draw."""
    # test one sample
    np.random.seed(0)
    uncorrelated_vector = np.array([[1.76405235, 0.40015721, 0.97873798, 2.2408932, 1.86755799]])
    mocker.patch("numpy.random.randn", return_value=uncorrelated_vector)
    one_sample = mean_field_normal.draw()
    expected_sample = (
        mean_field_normal.mean
        + mean_field_normal.covariance ** (1 / 2) * uncorrelated_vector.flatten()
    ).reshape(1, -1)
    np.testing.assert_array_almost_equal(one_sample, expected_sample, decimal=6)

    # test three samples
    num_samples = 3
    uncorrelated_vectors = np.outer(
        np.array([[1.76405235, 0.40015721, 0.97873798, 2.2408932, 1.86755799]]),
        np.array([[0.4], [1.3], [-0.75]]),
    )

    expected_samples = (
        mean_field_normal.mean.reshape(1, -1)
        + mean_field_normal.standard_deviation.reshape(1, -1) * uncorrelated_vectors.T
    )
    mocker.patch("numpy.random.randn", return_value=uncorrelated_vectors.T)
    multiple_samples = mean_field_normal.draw(num_samples)
    np.testing.assert_array_almost_equal(multiple_samples, expected_samples, decimal=6)


def test_logpdf(mean_field_normal, samples):
    """Test log_pdf."""
    logpdf = mean_field_normal.logpdf(samples)

    # create logpdf of jax scipy multivariate normal to compare
    mean = mean_field_normal.mean
    covariance = np.diag(mean_field_normal.covariance)
    benchmark_logpdf = jax_stats.multivariate_normal.logpdf(samples, mean, covariance).flatten()
    np.testing.assert_array_almost_equal(logpdf, benchmark_logpdf, decimal=6)


def test_grad_logpdf(mean_field_normal, samples):
    """Test grad_logpdf."""
    grad_logpdf = mean_field_normal.grad_logpdf(samples)

    # create grad logpdf of jax scipy multivariate normal to compare
    mean = mean_field_normal.mean
    covariance = np.diag(mean_field_normal.covariance)

    grad_benchmark_logpdf_lst = []
    for sample in samples:
        grad_benchmark_logpdf_lst.append(
            grad(jax_stats.multivariate_normal.logpdf)(sample, mean, covariance)
        )
    grad_benchmark_logpdf = np.array(grad_benchmark_logpdf_lst).reshape(2, -1)

    # test
    np.testing.assert_array_almost_equal(grad_logpdf, grad_benchmark_logpdf, decimal=6)


def test_grad_logpdf_var(mean_field_normal, samples):
    """Test grad_logpdf_var."""
    grad_logpdf_var = mean_field_normal.grad_logpdf_var(samples)

    # create grad logpdf var of jax scipy multivariate normal to compare
    mean = mean_field_normal.mean
    cov_vec = mean_field_normal.covariance

    def new_fun(cov, sample):
        return jax_stats.multivariate_normal.logpdf(sample, mean, jnp.diag(cov))

    grad_benchmark_logpdf_var_lst = []
    for sample in samples:
        grad_benchmark_logpdf_var_lst.append(grad(new_fun)(cov_vec, sample))
    grad_benchmark_logpdf_var = np.array(grad_benchmark_logpdf_var_lst).reshape(2, -1)

    # test
    np.testing.assert_array_almost_equal(grad_logpdf_var, grad_benchmark_logpdf_var, decimal=6)


def test_pdf(mean_field_normal, samples):
    """Test pdf."""
    pdf_vec = mean_field_normal.pdf(samples)
    benchmark_pdf_vec = jax_stats.multivariate_normal.pdf(
        samples, mean_field_normal.mean, np.diag(mean_field_normal.covariance)
    ).flatten()
    np.testing.assert_array_almost_equal(pdf_vec, benchmark_pdf_vec, decimal=6)


def test_ppf(mean_field_normal):
    """Test ppf."""
    quantile = 0.8
    # test multivariate case
    with pytest.raises(ValueError):
        mean_field_normal.ppf(quantile)

    # test univariate case
    mean_field_normal.mean = np.array([2.0])
    mean_field_normal.covariance = np.array([3.0])
    mean_field_normal.dimension = 1

    ppf = mean_field_normal.ppf(quantile)
    expected_ppf = scipy.stats.norm.ppf(quantile, loc=2.0, scale=3.0 ** (1 / 2))

    # test
    np.testing.assert_array_almost_equal(ppf, expected_ppf, decimal=6)
