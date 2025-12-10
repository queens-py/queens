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
"""Test-module for Parameters class."""

import numpy as np
import pytest

from queens.distributions.normal import Normal
from queens.distributions.uniform import Uniform
from queens.distributions.uniform_discrete import UniformDiscrete
from queens.parameters.parameters import Parameters


@pytest.fixture(name="parameters_set_1", scope="module")
def fixture_parameters_set_1():
    """Parameters dict without random field."""
    x1 = Uniform(lower_bound=-5, upper_bound=10)
    x2 = Normal(mean=[0, 1], covariance=np.diag([1, 2]))
    return Parameters(x1=x1, x2=x2)


@pytest.fixture(name="parameters_set_2", scope="module")
def fixture_parameters_set_2():
    """Parameters dict without random field."""
    x1 = Uniform(lower_bound=-5, upper_bound=10)
    x2 = Normal(mean=0, covariance=1)
    return Parameters(x1=x1, x2=x2)


@pytest.fixture(name="parameters_set_3", scope="module")
def fixture_parameters_set_3():
    """Parameters dict without random field."""
    x1 = Uniform(lower_bound=-5, upper_bound=10)
    x2 = UniformDiscrete(sample_space=[[0], [1], [2], [3], [4]])
    return Parameters(x1=x1, x2=x2)


def test_from_config_create_parameters_set_1(parameters_set_1):
    """Test *from_config_create_parameters* method."""
    random_variable_x1 = parameters_set_1.dict["x1"]
    random_variable_x2 = parameters_set_1.dict["x2"]
    expected_covariance = np.array([[1, 0], [0, 2]])
    expected_low_chol = np.array([[1.0, 0.0], [0.0, 1.41421356]])
    expected_precision = np.array([[1.0, 0.0], [0.0, 0.5]])

    assert parameters_set_1.names == ["x1", "x2"]
    assert random_variable_x1.lower_bound == -5
    assert random_variable_x1.upper_bound == 10
    assert random_variable_x1.mean == [2.5]
    assert random_variable_x1.pdf_const == 0.06666666666666667
    assert random_variable_x1.width == [15]
    assert random_variable_x1.logpdf_const == -2.70805020110221

    assert np.array_equal(random_variable_x2.covariance, expected_covariance)
    assert random_variable_x2.dimension == 2
    assert random_variable_x2.logpdf_const == -2.184450656689318
    assert np.array_equal(random_variable_x2.mean, [0, 1])
    assert np.array_equal(random_variable_x2.mean, [0, 1])
    assert np.allclose(random_variable_x2.low_chol, expected_low_chol)
    assert np.allclose(random_variable_x2.precision, expected_precision)


def test_draw_samples(parameters_set_1):
    """Test *draw_samples* method."""
    np.random.seed(41)
    samples = parameters_set_1.draw_samples(1)
    np.testing.assert_almost_equal(samples, np.array([[-1.23615, 0.11724, 0.57436]]), decimal=5)
    samples = parameters_set_1.draw_samples(2)
    np.testing.assert_almost_equal(
        samples, np.array([[-4.34796, -1.23961, 1.84974], [-3.25364, 0.41658, 1.34302]]), decimal=5
    )
    samples = parameters_set_1.draw_samples(1000)
    mean = np.mean(samples, axis=0)
    variance = np.var(samples, axis=0)
    np.testing.assert_almost_equal(mean, np.array([2.42511, 0.02864, 1.03762]), decimal=5)
    np.testing.assert_almost_equal(variance, np.array([19.00948, 1.03104, 2.09257]), decimal=5)


def test_joint_logpdf(parameters_set_1):
    """Test *joint_logpdf* method."""
    samples = np.array([1, 2, 3])
    logpdf = parameters_set_1.joint_logpdf(samples)
    np.testing.assert_almost_equal(logpdf, np.array([-7.89250]), decimal=5)
    samples = np.array([[20, 2, 3], [-2, 4, -2]])
    logpdf = parameters_set_1.joint_logpdf(samples)
    np.testing.assert_almost_equal(logpdf, np.array([-np.inf, -15.14250]), decimal=5)


@pytest.mark.parametrize(
    "parameters_set, samples, expected, expect_error",
    [
        # test case for parameter_set_1, expect Value Error
        ("parameters_set_1", np.array([0.5, 0.1, 0.6]), None, True),
        # test case for parameter_set_2, 1D sample
        ("parameters_set_2", np.array([0.5, 0.1]), np.array([[2.5, -1.28155]]), False),
        # test case for parameter_set_2, 2D sample
        (
            "parameters_set_2",
            np.array([[0.5, 0.1], [1.0, 0.1]]),
            np.array([[2.5, -1.28155], [10.0, -1.28155]]),
            False,
        ),
        # test case for parameter_set_3
        (
            "parameters_set_3",
            np.array([[0.5, 0.1], [0.5, 0.6]]),
            np.array([[2.5, 0], [2.5, 2]]),
            False,
        ),
    ],
)
def test_inverse_cdf_transform(parameters_set, samples, expected, expect_error, request):
    """Test *inverse_cdf_transform* method."""
    parameters = request.getfixturevalue(parameters_set)

    if expect_error:
        with pytest.raises(ValueError):
            parameters.inverse_cdf_transform(samples)
    else:
        transformed_samples = parameters.inverse_cdf_transform(samples)
        np.testing.assert_almost_equal(transformed_samples, expected, decimal=5)


def test_sample_as_dict(parameters_set_1):
    """Test *sample_as_dict* method."""
    sample = np.array([0.5, 0.1, 0.6])
    sample_dict = parameters_set_1.sample_as_dict(sample)
    assert sample_dict == {"x1": 0.5, "x2_0": 0.1, "x2_1": 0.6}


def test_dict_as_sample(parameters_set_1):
    """Test *dict_as_sample* method."""
    sample_dict = {"x2_0": 0.1, "x2_1": 0.6, "x1": 0.5}  # ordering does not matter
    sample_array = parameters_set_1.dict_as_sample(sample_dict)
    np.testing.assert_array_equal(sample_array, np.array([0.5, 0.1, 0.6]))


def test_to_list(parameters_set_1):
    """Test *to_list* method."""
    parameters_list = parameters_set_1.to_list()
    assert isinstance(parameters_list, list)
    assert len(parameters_list) == 2


# -------------------------------------------------------------------------------
# -------------------------   With random field   -------------------------------
# -------------------------------------------------------------------------------
@pytest.fixture(name="parameters_options_3", scope="module")
def fixture_parameters_options_3():
    """Parameters dict with random field."""
    parameters_dict = {
        "x1": {
            "type": "uniform",
            "lower_bound": -5,
            "upper_bound": 10,
        },
        "x2": {
            "type": "normal",
            "mean": [0, 1],
            "covariance": np.diag([1, 2]),
        },
        "random_inflow": {
            "type": "kl",
            "corr_length": 1.0,
            "std": 0.001,
            "mean": 0,
            "explained_variance": 0.98,
        },
    }
    return parameters_dict


@pytest.fixture(name="pre_processor", scope="module")
def fixture_pre_processor():
    """Create basic preprocessor class instance."""

    class PreProcessor:
        """Basic preprocessor class."""

        def __init__(self):
            """Initialize."""
            self.coords_dict = {
                "random_inflow": {
                    "keys": ["random_inflow_0", "random_inflow_1", "random_inflow_2"],
                    "coords": [[0.0], [0.5], [1.0]],
                }
            }

    return PreProcessor()
