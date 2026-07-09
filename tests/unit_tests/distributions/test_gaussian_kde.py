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
"""Tests for the Gaussian KDE distribution."""

import numpy as np
import pytest

from queens.distributions.gaussian_kde import GaussianKDE


@pytest.fixture(name="gaussian_kde")
def fixture_distribution():
    """Create a weighted two-dimensional Gaussian KDE."""
    samples = np.array([[-1.0, 0.0], [0.0, 1.0], [1.0, -0.5], [2.0, 1.5]])
    weights = np.array([0.1, 0.2, 0.3, 0.4])
    return GaussianKDE(samples, weights=weights)


def test_pdf(gaussian_kde: GaussianKDE):
    """Test that pdf matches the underlying scipy implementation."""
    points = np.array([[-0.5, 0.2], [0.5, 0.7], [1.5, 1.0]])
    np.testing.assert_allclose(gaussian_kde.pdf(points), gaussian_kde.scipy_kde.pdf(points.T))


def test_logpdf(gaussian_kde: GaussianKDE):
    """Test that logpdf matches the underlying scipy implementation."""
    points = np.array([[-0.5, 0.2], [0.5, 0.7], [1.5, 1.0]])
    np.testing.assert_allclose(gaussian_kde.logpdf(points), gaussian_kde.scipy_kde.logpdf(points.T))


def test_cdf(gaussian_kde: GaussianKDE):
    """Test that cdf matches the underlying scipy implementation."""
    points = np.array([[-0.5, 0.2], [0.5, 0.7], [1.5, 1.0]])
    lower = np.full(gaussian_kde.dimension, -np.inf)
    expected_cdf = np.array(
        [
            gaussian_kde.scipy_kde.integrate_box(low_bounds=lower, high_bounds=point)
            for point in points
        ]
    )

    np.testing.assert_allclose(gaussian_kde.cdf(points), expected_cdf)

    point = np.array([-0.5, 0.2])
    expected_single_point_cdf = gaussian_kde.scipy_kde.integrate_box(
        low_bounds=lower, high_bounds=point
    )
    np.testing.assert_allclose(gaussian_kde.cdf(point), [expected_single_point_cdf])


def test_evaluate_single_multivariate_point(gaussian_kde: GaussianKDE):
    """Accept a single point without an explicit sample dimension."""
    point_1d_array = np.array([-0.5, 0.2])
    scipy_point = point_1d_array[:, np.newaxis]
    np.testing.assert_allclose(
        gaussian_kde.pdf(point_1d_array), gaussian_kde.scipy_kde.pdf(scipy_point)
    )
    np.testing.assert_allclose(
        gaussian_kde.logpdf(point_1d_array), gaussian_kde.scipy_kde.logpdf(scipy_point)
    )
    assert gaussian_kde.cdf(point_1d_array).shape == (1,)


@pytest.mark.parametrize("method_name", ["cdf", "pdf", "logpdf"])
@pytest.mark.parametrize("points", [np.array([[1.0, 2.0, 3.0]]), np.array([1.0, 2.0, 3.0, 4.0])])
def test_evaluate_rejects_wrong_point_shape(
    gaussian_kde: GaussianKDE, method_name: str, points: np.ndarray
):
    """Reject points that do not match the distribution dimension."""
    with pytest.raises(ValueError, match="Evaluation points must have shape"):
        getattr(gaussian_kde, method_name)(points)


def test_mean_and_covariance(gaussian_kde: GaussianKDE):
    """Test that mean and covariance are calculated correctly."""
    np.random.seed(42)
    many_samples = gaussian_kde.draw(1_000_000)
    np.testing.assert_allclose(gaussian_kde.mean, np.mean(many_samples, axis=0), rtol=1e-2)
    np.testing.assert_allclose(
        gaussian_kde.covariance, np.cov(many_samples, rowvar=False), rtol=1e-2
    )


def test_draw(gaussian_kde: GaussianKDE):
    """Draw row-wise samples with the correct dimension."""
    np.random.seed(42)
    samples1 = gaussian_kde.draw(20)
    assert samples1.shape == (20, 2)

    # test reproducibility via np.random.seed
    np.random.seed(42)
    samples2 = gaussian_kde.draw(20)
    np.testing.assert_allclose(samples1, samples2)


def test_downsample_univariate():
    """Test downsampling of univariate Gaussian KDE."""
    # create a simple non-uniformly weighted bimodal distribution
    random_generator = np.random.default_rng(42)
    samples = np.concatenate(
        [
            random_generator.normal(-2.0, 0.7, 600),
            random_generator.normal(2.0, 1.0, 400),
        ]
    )
    weights = np.linspace(1.0, 2.0, samples.size)
    weighted_bimodal_kde = GaussianKDE(samples, weights=weights)

    # reuce to 10% of the original number of kernels
    max_kernels = int(0.1 * weighted_bimodal_kde.samples.shape[0])
    reduced_distribution = weighted_bimodal_kde.downsample(max_kernels=max_kernels, random_state=41)

    # assert correct size after reduction
    assert reduced_distribution.samples.shape == (max_kernels, 1)
    assert reduced_distribution.weights.shape == (max_kernels,)

    # assert mean is preserved almost exactly
    np.testing.assert_allclose(reduced_distribution.mean, weighted_bimodal_kde.mean, rtol=1e-15)

    # assert integrated error is small
    grid = np.linspace(-5.0, 5.0, 101)
    integrated_error = np.trapezoid(
        np.abs(reduced_distribution.pdf(grid) - weighted_bimodal_kde.pdf(grid)), grid
    )
    assert integrated_error < 0.002


def test_downsample_multivariate():
    """Test downsampling of multivariate Gaussian KDE."""
    # create a simple non-uniformly weighted bimodal distribution
    random_generator = np.random.default_rng(42)
    samples_x = np.concatenate(
        [
            random_generator.normal(-2.0, 0.7, 600),
            random_generator.normal(2.0, 1.0, 400),
        ]
    )
    samples_y = np.concatenate(
        [
            random_generator.normal(-1.0, 0.5, 600),
            random_generator.normal(1.0, 0.8, 400),
        ]
    )

    samples = np.column_stack((samples_x, samples_y))
    weights = np.linspace(1.0, 2.0, samples.shape[0])
    weighted_bimodal_kde = GaussianKDE(samples, weights=weights)

    # reuce to 10% of the original number of kernels
    max_kernels = int(0.1 * weighted_bimodal_kde.samples.shape[0])
    reduced_distribution = weighted_bimodal_kde.downsample(max_kernels=max_kernels, random_state=41)

    # assert correct size after reduction
    assert reduced_distribution.samples.shape == (max_kernels, 2)
    assert reduced_distribution.weights.shape == (max_kernels,)

    # assert mean is preserved almost exactly
    np.testing.assert_allclose(reduced_distribution.mean, weighted_bimodal_kde.mean, rtol=1e-14)

    # assert integrated error is small
    grid = np.linspace(-5.0, 5.0, 101)
    x_grid, y_grid = np.meshgrid(grid, grid)
    grid_points = np.column_stack((x_grid.ravel(), y_grid.ravel()))

    # integrated_error
    reduced_pdf_at_points = reduced_distribution.pdf(grid_points)
    original_pdf_at_points = weighted_bimodal_kde.pdf(grid_points)
    absolute_pdf_error = np.abs(reduced_pdf_at_points - original_pdf_at_points).reshape(
        x_grid.shape
    )
    integrated_error = np.trapezoid(
        np.trapezoid(absolute_pdf_error, x=grid, axis=0),
        x=grid,
    )

    assert integrated_error < 0.04


def test_downsample_small_kde_does_nothing(gaussian_kde: GaussianKDE):
    """Avoid refitting a KDE that is already sufficiently small."""
    assert gaussian_kde.downsample(max_kernels=4) is gaussian_kde


@pytest.mark.parametrize("max_kernels", [-1, 2])
def test_downsample_rejects_invalid_max_kernels(gaussian_kde: GaussianKDE, max_kernels):
    """Require enough kernels for a nonsingular KDE."""
    with pytest.raises(ValueError, match="larger than the distribution dimension"):
        gaussian_kde.downsample(max_kernels=max_kernels)


@pytest.mark.parametrize("samples", [[[1.0]], np.eye(2)])
def test_init_rejects_too_few_kernels(samples):
    """Require enough kernels to fit a nonsingular KDE."""
    with pytest.raises(ValueError, match="larger than the distribution dimension"):
        GaussianKDE(samples)


def test_ppf(gaussian_kde: GaussianKDE):
    """Reject inverse-CDF evaluation."""
    with pytest.raises(
        NotImplementedError, match="ppf not available for Gaussian KDE distributions."
    ):
        gaussian_kde.ppf(np.array([0.5]))


def test_grad_logpdf(gaussian_kde: GaussianKDE):
    """Reject log-PDF gradient evaluation."""
    with pytest.raises(
        NotImplementedError,
        match="Gradient of logpdf not available for Gaussian KDE distributions.",
    ):
        gaussian_kde.grad_logpdf(np.array([[0.0, 0.0]]))
