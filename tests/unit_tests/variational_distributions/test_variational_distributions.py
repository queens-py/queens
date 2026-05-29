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
"""Test variational inference utils."""

from collections import namedtuple

import numpy as np
import pytest
import scipy

from queens.distributions.mixture import Mixture
from queens.distributions.normal import Normal
from queens.distributions.particle import Particle as ParticleDistribution
from queens.variational_distributions import (
    FullRankNormal,
    Joint,
    MeanFieldNormal,
    MixtureModel,
    Particle,
)


def nested_numpy_assertion(obtained, reference):
    """Assert lists/touples containing arrays.

    Args:
        obtained  (tuple, list): obtained data
        reference (tuple, list): reference data
    """
    if isinstance(obtained, (tuple, list)):
        for obtained_sub, reference_sub in zip(obtained, reference, strict=True):
            nested_numpy_assertion(obtained_sub, reference_sub)
    else:
        np.testing.assert_almost_equal(obtained, reference)


# Reference data named tuple. Makes it easier to concentrate data
ReferenceData = namedtuple(
    "ReferenceData",
    [
        "distribution",
        "distribution_parameters",
        "variational_parameters",
        "input_samples",
        "default_variational_parameters",
        "grad_params_logpdf",
        "fisher_information_matrix",
    ],
)

# Distribution to be tested
DISTRIBUTION_NAMES = ["mean_field", "fullrank", "particles", "mixture", "joint"]


@pytest.fixture(name="mean_field_distribution")
def fixture_mean_field_distribution():
    """Mean field Normal distribution."""
    return MeanFieldNormal(3)


@pytest.fixture(name="fullrank_distribution")
def fixture_fullrank_distribution():
    """Fullrank Normal distribution."""
    return FullRankNormal(3)


@pytest.fixture(name="mixture_distribution")
def fixture_mixture_distribution(mean_field_distribution):
    """Mixutre models of mean field distributions."""
    return MixtureModel(mean_field_distribution, mean_field_distribution.dimension, 2)


@pytest.fixture(name="particles_distribution")
def fixture_particles_distribution():
    """Particles distribution."""
    return Particle([[1], [2]])


@pytest.fixture(name="joint_distribution")
def fixture_joint_distribution(mean_field_distribution, fullrank_distribution):
    """Joint distribution."""
    return Joint([mean_field_distribution, fullrank_distribution], 6)


@pytest.fixture(name="mean_field_reference_data")
def fixture_meanfield_reference_data():
    """Reference data for the mean field distribution."""
    mean = np.ones((3, 1))
    cov = np.eye(3) * 2
    variational_parameters = np.concatenate([mean.flatten(), 0.5 * np.log(np.diag(cov))])
    distribution = Normal(mean, cov)
    input_samples = np.array([[0.1, 0.2, 0.3], [0.2, 0.1, 1]])
    score_function_values = np.array(
        [
            [-0.45, -0.4],
            [-0.4, -0.45],
            [-0.35, 0.0],
            [-0.595, -0.68],
            [-0.68, -0.595],
            [-0.755, -1.0],
        ]
    )
    fisher_information_matrix = np.diag([0.5, 0.5, 0.5, 2, 2, 2])
    return ReferenceData(
        distribution,
        (mean, cov),
        variational_parameters,
        input_samples,
        np.zeros(len(variational_parameters)),
        score_function_values,
        fisher_information_matrix,
    )


@pytest.fixture(name="fullrank_reference_data")
def fixture_fullrank_reference_data():
    """Reference data for the fullrank field distribution."""
    mean = np.ones(3).reshape(-1, 1)
    lower_chol = np.array([[2, 0, 0], [1, 2, 0], [1, 1, 2]])
    cov = np.matmul(lower_chol, lower_chol.T)
    cov_part = np.array([2, 1, 2, 1, 1, 2])
    variational_parameters = np.concatenate([mean.flatten(), cov_part])
    distribution = Normal(mean, cov)
    input_samples = np.array([[0.1, 0.2, 0.3], [0.2, 0.1, 1]])

    default_variational_parameters = np.ones(len(variational_parameters))
    default_variational_parameters[: distribution.dimension] = 0

    score_function = np.array(
        [
            [-0.1765625, -0.178125],
            [-0.078125, -0.20625],
            [-0.01875, 0.1625],
            [-0.42054687, -0.42875],
            [0.03515625, 0.0825],
            [-0.48632812, -0.4484375],
            [0.0084375, -0.065],
            [0.00328125, -0.040625],
            [-0.49929688, -0.4471875],
        ]
    )
    fisher_information_matrix = np.array(
        [
            [0.328125, -0.09375, -0.0625, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [-0.09375, 0.3125, -0.125, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [-0.0625, -0.125, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.578125, -0.09375, 0.0, -0.0625, 0.0, 0.0],
            [0.0, 0.0, 0.0, -0.09375, 0.3125, 0.0, -0.125, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.5625, 0.0, -0.125, 0.0],
            [0.0, 0.0, 0.0, -0.0625, -0.125, 0.0, 0.25, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, -0.125, 0.0, 0.25, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5],
        ]
    )
    return ReferenceData(
        distribution,
        (mean, cov),
        variational_parameters,
        input_samples,
        default_variational_parameters,
        score_function,
        fisher_information_matrix,
    )


@pytest.fixture(name="joint_reference_data")
def fixture_joint_reference_data(mean_field_reference_data, fullrank_reference_data):
    """Reference data for the joint field distribution."""
    # join mean values
    mean = np.row_stack(
        (
            mean_field_reference_data.distribution_parameters[0],
            fullrank_reference_data.distribution_parameters[0],
        )
    )

    # construct covariance
    cov = scipy.linalg.block_diag(
        mean_field_reference_data.distribution_parameters[1],
        fullrank_reference_data.distribution_parameters[1],
    )
    distribution = Normal(mean, cov)

    # concatenate variational_parameters
    variational_parameters = np.concatenate(
        (
            mean_field_reference_data.variational_parameters,
            fullrank_reference_data.variational_parameters,
        )
    )

    # concatenate default variational_parameters
    default_variational_parameters = np.concatenate(
        (
            mean_field_reference_data.default_variational_parameters,
            fullrank_reference_data.default_variational_parameters,
        )
    )

    # stack input samples
    input_samples = np.column_stack(
        (
            mean_field_reference_data.input_samples,
            fullrank_reference_data.input_samples,
        )
    )

    # stack score functions
    score_function = np.row_stack(
        (
            mean_field_reference_data.grad_params_logpdf,
            fullrank_reference_data.grad_params_logpdf,
        )
    )

    fisher_information_matrix = scipy.linalg.block_diag(
        mean_field_reference_data.fisher_information_matrix,
        fullrank_reference_data.fisher_information_matrix,
    )
    distribution_parameters = [
        mean_field_reference_data.distribution_parameters,
        fullrank_reference_data.distribution_parameters,
    ]
    return ReferenceData(
        distribution,
        [distribution_parameters],
        variational_parameters,
        input_samples,
        default_variational_parameters,
        score_function,
        fisher_information_matrix,
    )


@pytest.fixture(name="mixture_reference_data")
def fixture_mixture_reference_data(mean_field_reference_data):
    """Reference data for the mixture distribution."""
    distribution_1 = mean_field_reference_data.distribution
    mean_1, cov_1 = mean_field_reference_data.distribution_parameters
    variational_parameters_1 = mean_field_reference_data.variational_parameters
    input_samples = mean_field_reference_data.input_samples

    mean_2 = -mean_1
    cov_2 = cov_1
    distribution_2 = Normal(mean_2, cov_2)
    variational_parameters_2 = np.concatenate([mean_2.flatten(), 0.5 * np.log(np.diag(cov_2))])
    weights = np.array([0.1, 0.9])

    distribution = Mixture(weights, [distribution_1, distribution_2])

    distribution_parameters_components = [(mean_1, cov_1), (mean_2, cov_2)]
    variational_parameters = np.concatenate(
        [variational_parameters_1, variational_parameters_2, np.log(weights)]
    )

    default_variational_parameters = np.zeros(len(variational_parameters))
    default_variational_parameters[-2:] = np.log(0.5)

    score_function = np.array(
        [
            [-0.07576644, -0.11584847],
            [-0.06734795, -0.13032953],
            [-0.05892946, 0.0],
            [-0.10018008, -0.1969424],
            [-0.11449152, -0.1723246],
            [-0.12711926, -0.28962118],
            [0.45739657, 0.42622729],
            [0.49897807, 0.39070835],
            [0.54055958, 0.71037882],
            [-0.3284939, -0.19890607],
            [-0.23285643, -0.28059963],
            [-0.12890267, 0.71037882],
            [0.06836988, 0.18962118],
            [-0.06836988, -0.18962118],
        ]
    )
    return ReferenceData(
        distribution,
        (distribution_parameters_components, weights),
        variational_parameters,
        input_samples,
        default_variational_parameters,
        score_function,
        None,
    )


@pytest.fixture(name="particles_reference_data")
def fixture_particles_reference_data():
    """Reference data for the particles distribution."""
    probabilities = np.array([0.1, 0.9])
    sample_space = np.array([[1], [2]])
    distribution = ParticleDistribution(probabilities, sample_space)
    variational_parameters = np.log(probabilities)
    input_samples = [[1], [1], [2]]

    default_variational_parameters = np.ones(2) * 0.5
    default_variational_parameters = np.log(default_variational_parameters)

    score_function = np.array([[0.9, 0.9, -0.1], [-0.9, -0.9, 0.1]])
    fisher_information_matrix = np.array([[0.09, -0.09], [-0.09, 0.09]])
    return ReferenceData(
        distribution,
        (probabilities, sample_space),
        variational_parameters,
        input_samples,
        default_variational_parameters,
        score_function,
        fisher_information_matrix,
    )


@pytest.fixture(name="distributions")
def fixture_distributions(request):
    """The distributions and reference data."""
    distribution_name = request.param
    distribution = request.getfixturevalue(distribution_name + "_distribution")
    reference_data = request.getfixturevalue(distribution_name + "_reference_data")
    return distribution, reference_data


@pytest.mark.parametrize(
    "distributions",
    DISTRIBUTION_NAMES,
    indirect=True,
)
def test_draw_dimension(distributions):
    """Test draw method."""
    distribution, reference_data = distributions
    variational_parameters = reference_data.variational_parameters
    assert distribution.draw(variational_parameters, 10).shape == (
        10,
        distribution.dimension,
    )


@pytest.mark.parametrize(
    "distributions",
    DISTRIBUTION_NAMES,
    indirect=True,
)
def test_reconstruct_distribution_parameters(distributions):
    """Test reconstruct distribution parameters method."""
    distribution, reference_data = distributions
    reconstructed_parameters = distribution.reconstruct_distribution_parameters(
        reference_data.variational_parameters
    )
    nested_numpy_assertion(reconstructed_parameters, reference_data.distribution_parameters)


@pytest.mark.parametrize(
    "distributions",
    DISTRIBUTION_NAMES,
    indirect=True,
)
def test_pdf(distributions):
    """Test pdf method."""
    distribution, reference_data = distributions
    obtained = distribution.pdf(reference_data.variational_parameters, reference_data.input_samples)
    reference = reference_data.distribution.pdf(reference_data.input_samples)
    np.testing.assert_almost_equal(obtained, reference)


@pytest.mark.parametrize(
    "distributions",
    DISTRIBUTION_NAMES,
    indirect=True,
)
def test_construct_variational_parameters(distributions):
    """Test construct variational parameters method."""
    distribution, reference_data = distributions
    obtained_variational_parameters = distribution.construct_variational_parameters(
        *reference_data.distribution_parameters
    )
    np.testing.assert_almost_equal(
        obtained_variational_parameters, reference_data.variational_parameters
    )


@pytest.mark.parametrize(
    "distributions",
    DISTRIBUTION_NAMES,
    indirect=True,
)
def test_logpdf(distributions):
    """Test logpdf method."""
    distribution, reference_data = distributions

    obtained = distribution.logpdf(
        reference_data.variational_parameters, reference_data.input_samples
    )
    reference = reference_data.distribution.logpdf(reference_data.input_samples)
    np.testing.assert_almost_equal(obtained, reference)


@pytest.mark.parametrize(
    "distributions",
    DISTRIBUTION_NAMES,
    indirect=True,
)
def test_random_initialization_dimension(distributions):
    """Test dimension of random initialization."""
    distribution, _ = distributions
    assert (
        distribution.initialize_variational_parameters().shape
        == distribution.initialize_variational_parameters(random=True).shape
    )


@pytest.mark.parametrize(
    "distributions",
    DISTRIBUTION_NAMES,
    indirect=True,
)
def test_grad_params_logpdf(distributions):
    """Test grad_params_logpdf method."""
    distribution, reference_data = distributions
    np.testing.assert_almost_equal(
        distribution.grad_params_logpdf(
            reference_data.variational_parameters, reference_data.input_samples
        ),
        reference_data.grad_params_logpdf,
    )


@pytest.mark.parametrize(
    "distributions",
    [name for name in DISTRIBUTION_NAMES if name != "mixture"],  # mixtures use MC
    indirect=True,
)
def test_fisher_information_matrix(distributions):
    """Test Fisher information matrix method."""
    distribution, reference_data = distributions
    np.testing.assert_almost_equal(
        distribution.fisher_information_matrix(reference_data.variational_parameters),
        reference_data.fisher_information_matrix,
    )


@pytest.mark.parametrize(
    "distributions",
    ["mixture"],
    indirect=True,
)
def test_fisher_information_matrix_mixture(distributions):
    """Test Fisher information matrix for the mixture."""
    distribution, reference_data = distributions

    # Seed needs to be fixed due to MC
    np.random.seed(42)
    fisher_information_matrix = np.array(
        [
            [
                3.06080000e-02,
                -5.49138154e-04,
                -1.89769539e-03,
                3.11856262e-02,
                -2.55931009e-03,
                -2.13490913e-03,
                7.60080656e-03,
                -1.21100059e-02,
                -1.16372587e-02,
                3.28987288e-02,
                -1.96920920e-02,
                -1.72690930e-02,
                1.29289534e-02,
                -1.29289534e-02,
            ],
            [
                -5.49138154e-04,
                2.96640537e-02,
                -3.22532276e-04,
                -7.13753203e-04,
                2.51165801e-02,
                -2.35472754e-03,
                -1.11138326e-02,
                8.67324526e-03,
                -1.12780780e-02,
                -1.66668787e-02,
                3.65828012e-02,
                -1.76977888e-02,
                1.40427452e-02,
                -1.40427452e-02,
            ],
            [
                -1.89769539e-03,
                -3.22532276e-04,
                2.67459566e-02,
                -4.65171955e-03,
                2.05184722e-03,
                1.80548992e-02,
                -1.18951344e-02,
                -1.25321269e-02,
                7.60485164e-03,
                -1.72694867e-02,
                -1.98390932e-02,
                3.32512867e-02,
                1.06219305e-02,
                -1.06219305e-02,
            ],
            [
                3.11856262e-02,
                -7.13753203e-04,
                -4.65171955e-03,
                1.35727617e-01,
                4.56849721e-04,
                7.12263892e-04,
                -2.25040119e-02,
                4.78128381e-03,
                4.79578497e-03,
                -4.35796329e-03,
                1.48385531e-02,
                1.26454205e-02,
                6.35870624e-04,
                -6.35870624e-04,
            ],
            [
                -2.55931009e-03,
                2.51165801e-02,
                2.05184722e-03,
                4.56849721e-04,
                1.11645726e-01,
                1.97211627e-03,
                3.90094793e-03,
                -2.09648169e-02,
                4.92667954e-03,
                1.10121478e-02,
                -1.33652729e-03,
                1.27732623e-02,
                -1.07848592e-03,
                1.07848592e-03,
            ],
            [
                -2.13490913e-03,
                -2.35472754e-03,
                1.80548992e-02,
                7.12263892e-04,
                1.97211627e-03,
                9.85006673e-02,
                4.80426877e-03,
                4.93119673e-03,
                -2.21595441e-02,
                1.16316727e-02,
                1.20487099e-02,
                -6.31598313e-03,
                -6.36819172e-03,
                6.36819172e-03,
            ],
            [
                7.60080656e-03,
                -1.11138326e-02,
                -1.18951344e-02,
                -2.25040119e-02,
                3.90094793e-03,
                4.80426877e-03,
                4.13357055e-01,
                -1.95187096e-02,
                -1.45207550e-02,
                -5.68741624e-02,
                5.27900629e-03,
                -1.64852750e-02,
                2.69106581e-02,
                -2.69106581e-02,
            ],
            [
                -1.21100059e-02,
                8.67324526e-03,
                -1.25321269e-02,
                4.78128381e-03,
                -2.09648169e-02,
                4.93119673e-03,
                -1.95187096e-02,
                4.24689137e-01,
                -1.99012755e-02,
                -1.52304439e-03,
                -7.23991013e-02,
                -1.42034734e-02,
                2.72208134e-02,
                -2.72208134e-02,
            ],
            [
                -1.16372587e-02,
                -1.12780780e-02,
                7.60485164e-03,
                4.79578497e-03,
                4.92667954e-03,
                -2.21595441e-02,
                -1.45207550e-02,
                -1.99012755e-02,
                4.14962873e-01,
                -9.04637721e-03,
                -1.95904870e-03,
                -5.48844013e-02,
                2.66214795e-02,
                -2.66214795e-02,
            ],
            [
                3.28987288e-02,
                -1.66668787e-02,
                -1.72694867e-02,
                -4.35796329e-03,
                1.10121478e-02,
                1.16316727e-02,
                -5.68741624e-02,
                -1.52304439e-03,
                -9.04637721e-03,
                1.53286058e00,
                -1.84654445e-02,
                2.37789120e-02,
                2.93396976e-02,
                -2.93396976e-02,
            ],
            [
                -1.96920920e-02,
                3.65828012e-02,
                -1.98390932e-02,
                1.48385531e-02,
                -1.33652729e-03,
                1.20487099e-02,
                5.27900629e-03,
                -7.23991013e-02,
                -1.95904870e-03,
                -1.84654445e-02,
                1.68953593e00,
                -2.94930948e-02,
                3.07967827e-02,
                -3.07967827e-02,
            ],
            [
                -1.72690930e-02,
                -1.76977888e-02,
                3.32512867e-02,
                1.26454205e-02,
                1.27732623e-02,
                -6.31598313e-03,
                -1.64852750e-02,
                -1.42034734e-02,
                -5.48844013e-02,
                2.37789120e-02,
                -2.94930948e-02,
                1.60817064e00,
                2.85616389e-02,
                -2.85616389e-02,
            ],
            [
                1.29289534e-02,
                1.40427452e-02,
                1.06219305e-02,
                6.35870624e-04,
                -1.07848592e-03,
                -6.36819172e-03,
                2.69106581e-02,
                2.72208134e-02,
                2.66214795e-02,
                2.93396976e-02,
                3.07967827e-02,
                2.85616389e-02,
                5.02412940e-02,
                -5.02412940e-02,
            ],
            [
                -1.29289534e-02,
                -1.40427452e-02,
                -1.06219305e-02,
                -6.35870624e-04,
                1.07848592e-03,
                6.36819172e-03,
                -2.69106581e-02,
                -2.72208134e-02,
                -2.66214795e-02,
                -2.93396976e-02,
                -3.07967827e-02,
                -2.85616389e-02,
                -5.02412940e-02,
                5.02412940e-02,
            ],
        ]
    )
    np.testing.assert_almost_equal(
        distribution.fisher_information_matrix(
            reference_data.variational_parameters, n_samples=10000
        ),
        fisher_information_matrix,
    )


def test_total_grad_params_logpdf_mean_field(mean_field_distribution):
    """Test total_grad_params_logpdf method for mean field distribution."""
    np.random.seed(42)
    variational_parameters = np.random.randn(mean_field_distribution.n_parameters)
    std_samples = np.random.randn(2, mean_field_distribution.dimension)
    gradient = mean_field_distribution.total_grad_params_logpdf(variational_parameters, std_samples)

    expected_gradient = np.array(
        [[0.0, 0.0, 0.0, -1.0, -1.0, -1.0], [0.0, 0.0, 0.0, -1.0, -1.0, -1.0]]
    )
    np.testing.assert_almost_equal(gradient, expected_gradient)


def test_total_grad_params_logpdf_fullrank(fullrank_distribution):
    """Test total_grad_params_logpdf method for fullrank distribution."""
    np.random.seed(42)
    variational_parameters = np.random.randn(fullrank_distribution.n_parameters)
    std_samples = np.random.randn(2, fullrank_distribution.dimension)
    gradient = fullrank_distribution.total_grad_params_logpdf(variational_parameters, std_samples)

    expected_gradient = np.array(
        [
            [0.0, 0.0, 0.0, -0.656586, 0.0, 4.271005, 0.0, 0.0, 2.130042],
            [0.0, 0.0, 0.0, -0.656586, 0.0, 4.271005, 0.0, 0.0, 2.130042],
        ]
    )
    np.testing.assert_almost_equal(gradient, expected_gradient, decimal=5)
