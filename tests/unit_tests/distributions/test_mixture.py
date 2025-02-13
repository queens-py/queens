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
"""Test-module for mixture distribution."""

import numpy as np
import pytest

from queens.distributions.mixture import Mixture
from queens.distributions.normal import Normal


@pytest.fixture(name="component_data", params=[1, 2])
def fixture_component_data(request):
    """Data for two compoments."""
    if request.param == 1:
        return (1, 2), (0.5, 3)

    return ([1, -1], [[1, 0], [0, 1]]), ([0, 0.5], [[2, 1], [1, 3]])


@pytest.fixture(name="reference_mixture_model_data")
def fixture_reference_mixture_model_data(component_data):
    """Referencei data model fixture."""
    component_data0, component_data1 = component_data
    normal0 = Normal(*component_data0)
    normal1 = Normal(*component_data1)
    weights = [0.3, 0.7]
    return (weights, normal0, normal1)


@pytest.fixture(name="mixture_model")
def fixture_mixture_model(reference_mixture_model_data):
    """Mixture model fixture."""
    weights, normal1, normal2 = reference_mixture_model_data
    mixture_model = Mixture(weights, (normal1, normal2))
    return mixture_model


def test_mean_and_covariance(mixture_model, reference_mixture_model_data):
    """Check if mean and covariance is correct."""
    weights_ref, normal0_ref, normal1_ref = reference_mixture_model_data
    mean_ref = weights_ref[0] * normal0_ref.mean + weights_ref[1] * normal1_ref.mean
    covariance_ref = (
        weights_ref[0] * (normal0_ref.covariance + np.outer(normal0_ref.mean, normal0_ref.mean))
        + weights_ref[1] * (normal1_ref.covariance + np.outer(normal1_ref.mean, normal1_ref.mean))
        - np.outer(mean_ref, mean_ref)
    )
    np.testing.assert_almost_equal(mean_ref, mixture_model.mean)
    np.testing.assert_almost_equal(covariance_ref, mixture_model.covariance)


def test_draw(mocker, mixture_model):
    """Check if samples are correctly drawn."""
    mocker.patch(
        "queens.distributions.mixture.np.random.multinomial",
        return_value=np.array([2, 3]),
    )
    mocker.patch(
        "queens.distributions.mixture.np.random.shuffle",
    )
    mocker.patch.object(
        mixture_model.component_distributions[0],
        "draw",
        return_value=np.ones((2, mixture_model.dimension)),
    )
    mocker.patch.object(
        mixture_model.component_distributions[1],
        "draw",
        return_value=2 * np.ones((3, mixture_model.dimension)),
    )
    np.testing.assert_equal(
        np.row_stack(
            (np.ones((2, mixture_model.dimension)), 2 * np.ones((3, mixture_model.dimension)))
        ),
        mixture_model.draw(5),
    )


def test_cdf(mixture_model, reference_mixture_model_data):
    """Test cdf method."""
    weights_ref, normal0_ref, normal1_ref = reference_mixture_model_data
    sample_location = np.arange(normal0_ref.dimension * 2).reshape(2, -1)
    cdf_ref = weights_ref[0] * normal0_ref.cdf(sample_location) + weights_ref[1] * normal1_ref.cdf(
        sample_location
    )
    np.testing.assert_almost_equal(cdf_ref, mixture_model.cdf(sample_location))


def test_logpdf(mixture_model, reference_mixture_model_data):
    """Test logpdf method."""
    weights_ref, normal0_ref, normal1_ref = reference_mixture_model_data
    sample_location = np.arange(normal0_ref.dimension * 2).reshape(2, -1)
    logpdf_ref = np.log(
        weights_ref[0] * normal0_ref.pdf(sample_location)
        + weights_ref[1] * normal1_ref.pdf(sample_location)
    )
    np.testing.assert_almost_equal(logpdf_ref, mixture_model.logpdf(sample_location))


def test_pdf(mixture_model, reference_mixture_model_data):
    """Test pdf method."""
    weights_ref, normal0_ref, normal1_ref = reference_mixture_model_data
    sample_location = np.arange(normal0_ref.dimension * 2).reshape(2, -1)
    pdf_ref = weights_ref[0] * normal0_ref.pdf(sample_location) + weights_ref[1] * normal1_ref.pdf(
        sample_location
    )

    np.testing.assert_almost_equal(pdf_ref, mixture_model.pdf(sample_location))


def test_grad_logpdf(mixture_model, reference_mixture_model_data):
    """Test grad_logpdf."""
    weights_ref, normal0_ref, normal1_ref = reference_mixture_model_data
    sample_location = np.arange(normal0_ref.dimension * 3).reshape(3, -1)
    pdf_component_0_ref = weights_ref[0] * normal0_ref.pdf(sample_location)
    pdf_component_1_ref = weights_ref[1] * normal1_ref.pdf(sample_location)
    pdf_ref = weights_ref[0] * normal0_ref.pdf(sample_location) + weights_ref[1] * normal1_ref.pdf(
        sample_location
    )
    responsibilities_ref = np.row_stack(
        (pdf_component_0_ref / pdf_ref, pdf_component_1_ref / pdf_ref)
    ).T

    grad_logpdf = []
    for responsibilities, grad_logpdf0, grad_logpdf1 in zip(
        responsibilities_ref,
        normal0_ref.grad_logpdf(sample_location),
        normal1_ref.grad_logpdf(sample_location),
        strict=True,
    ):
        grad_logpdf.append(responsibilities[0] * grad_logpdf0 + responsibilities[1] * grad_logpdf1)

    np.testing.assert_array_almost_equal(
        np.array(grad_logpdf), mixture_model.grad_logpdf(sample_location)
    )


def test_responsibilities(mixture_model, reference_mixture_model_data):
    """Test responsibilities."""
    weights_ref, normal0_ref, normal1_ref = reference_mixture_model_data
    sample_location = np.arange(normal0_ref.dimension * 3).reshape(3, -1)
    pdf_component_0_ref = weights_ref[0] * normal0_ref.pdf(sample_location)
    pdf_component_1_ref = weights_ref[1] * normal1_ref.pdf(sample_location)
    pdf_ref = weights_ref[0] * normal0_ref.pdf(sample_location) + weights_ref[1] * normal1_ref.pdf(
        sample_location
    )
    responsibilities_ref = np.row_stack(
        (pdf_component_0_ref / pdf_ref, pdf_component_1_ref / pdf_ref)
    ).T

    # Responsibilities should add up to one
    np.testing.assert_allclose(1, np.sum(mixture_model.responsibilities(sample_location), axis=1))
    np.testing.assert_array_almost_equal(
        responsibilities_ref, mixture_model.responsibilities(sample_location)
    )


def test_ppf(mixture_model):
    """Test PPF."""
    with pytest.raises(NotImplementedError, match="PPF not available"):
        mixture_model.ppf(None)


def test_fcc(reference_mixture_model_data):
    """Test from config create."""
    weights, normal0, normal1 = reference_mixture_model_data
    config = {
        "type": "mixture",
        "weights": weights,
        "component_distributions_names": ["normal_0", "normal_1"],
        "normal_0": {
            "type": "normal",
            "mean": normal0.mean.tolist(),
            "covariance": normal0.covariance.tolist(),
        },
        "normal_1": {
            "type": "normal",
            "mean": normal1.mean.tolist(),
            "covariance": normal1.covariance.tolist(),
        },
    }
    mixture = Mixture.from_config_create_distribution(config)

    np.testing.assert_array_equal(weights, mixture.weights)
    np.testing.assert_array_equal(normal0.mean, mixture.component_distributions[0].mean)
    np.testing.assert_array_equal(normal1.mean, mixture.component_distributions[1].mean)
    np.testing.assert_array_equal(normal0.covariance, mixture.component_distributions[0].covariance)
    np.testing.assert_array_equal(normal1.covariance, mixture.component_distributions[1].covariance)
