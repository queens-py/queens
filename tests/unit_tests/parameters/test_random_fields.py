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
"""Test-module for Random field expansions."""

from unittest.mock import MagicMock

import numpy as np
import pytest

from queens.parameters.parameters import from_config_create_parameters


@pytest.fixture(name="parameters", scope="module")
def fixture_parameters(pre_processor):
    """Parameters dict with random fields."""
    # mock np.linalg.eigh function
    np.linalg.eigh = MagicMock(
        return_value=(
            np.array(
                [
                    0.45871872,
                    0.33733471,
                    0.33733471,
                    0.24807078,
                    0.21987822,
                    0.21987822,
                    0.16169507,
                    0.16169507,
                ]
            ),
            np.array(
                [
                    [
                        -2.51370339e-01,
                        4.93708941e-01,
                        8.73030401e-02,
                        5.00000000e-01,
                        2.75230174e-02,
                        3.52475145e-01,
                        -4.98562964e-01,
                        8.03936995e-03,
                    ],
                    [
                        -3.53548079e-01,
                        4.08591703e-01,
                        -2.85801471e-01,
                        -5.39814040e-16,
                        -4.98695975e-01,
                        3.61916477e-02,
                        3.60190899e-01,
                        3.48759022e-01,
                    ],
                    [
                        -2.51370339e-01,
                        8.73030401e-02,
                        -4.93708941e-01,
                        -5.00000000e-01,
                        2.75230174e-02,
                        3.52475145e-01,
                        -8.03936995e-03,
                        -4.98562964e-01,
                    ],
                    [
                        -3.53548079e-01,
                        2.85801471e-01,
                        4.08591703e-01,
                        2.61003557e-16,
                        4.98269263e-01,
                        -4.16563707e-02,
                        3.48759022e-01,
                        -3.60190899e-01,
                    ],
                    [
                        -4.97259321e-01,
                        -2.40914807e-16,
                        -3.22497934e-16,
                        -2.12697105e-15,
                        -5.50460347e-02,
                        -7.04950290e-01,
                        5.19638692e-16,
                        5.08102538e-16,
                    ],
                    [
                        -3.53548079e-01,
                        -2.85801471e-01,
                        -4.08591703e-01,
                        -1.06257318e-15,
                        4.98269263e-01,
                        -4.16563707e-02,
                        -3.48759022e-01,
                        3.60190899e-01,
                    ],
                    [
                        -2.51370339e-01,
                        -8.73030401e-02,
                        4.93708941e-01,
                        -5.00000000e-01,
                        2.75230174e-02,
                        3.52475145e-01,
                        8.03936995e-03,
                        4.98562964e-01,
                    ],
                    [
                        -3.53548079e-01,
                        -4.08591703e-01,
                        2.85801471e-01,
                        1.30916575e-16,
                        -4.98695975e-01,
                        3.61916477e-02,
                        -3.60190899e-01,
                        -3.48759022e-01,
                    ],
                    [
                        -2.51370339e-01,
                        -4.93708941e-01,
                        -8.73030401e-02,
                        5.00000000e-01,
                        2.75230174e-02,
                        3.52475145e-01,
                        4.98562964e-01,
                        -8.03936995e-03,
                    ],
                ]
            ),
        )
    )
    parameters_dict = {
        "field_1": {"type": "kl", "corr_length": 0.3, "std": 0.5, "explained_variance": 0.9},
        "field_2": {
            "type": "fourier",
            "corr_length": 0.3,
            "std": 0.5,
            "variability": 0.1,
            "trunc_threshold": 1,
        },
        "field_3": {
            "type": "piece-wise",
            "distribution": {
                "type": "normal",
                "mean": [0],
                "covariance": [[1]],
            },
        },
    }
    parameters = from_config_create_parameters(parameters_dict, pre_processor=pre_processor)
    return parameters


def test_draw_samples(parameters):
    """Test *draw_samples* method."""
    np.random.seed(42)
    samples = parameters.draw_samples(2)
    pytest.approx(
        samples,
        np.array(
            [
                [
                    0.49671415,
                    0.64768854,
                    -0.23415337,
                    1.57921282,
                    -0.46947439,
                    -0.46341769,
                    0.24196227,
                    -1.72491783,
                    -1.01283112,
                    -0.90802408,
                    1.46564877,
                    0.0675282,
                    -0.54438272,
                    0.11092259,
                    -1.15099358,
                    0.37569802,
                    -0.60063869,
                    -0.29169375,
                    -0.60170661,
                    1.85227818,
                    -0.01349722,
                ],
                [
                    -0.1382643,
                    1.52302986,
                    -0.23413696,
                    0.76743473,
                    0.54256004,
                    -0.46572975,
                    -1.91328024,
                    -0.56228753,
                    0.31424733,
                    -1.4123037,
                    -0.2257763,
                    -1.42474819,
                    -1.05771093,
                    0.82254491,
                    -1.22084365,
                    0.2088636,
                    -1.95967012,
                    -1.32818605,
                    0.19686124,
                    0.73846658,
                    0.17136828,
                ],
            ]
        ),
    )
    samples = parameters.draw_samples(1000)
    mean = np.mean(samples, axis=0)
    variance = np.var(samples, axis=0)
    pytest.approx(
        mean,
        np.array(
            [
                0.03984161,
                0.050958,
                0.00090938,
                -0.01797671,
                -0.03360751,
                -0.05189412,
                -0.02305717,
                0.01389105,
                0.0189942,
                -0.00594287,
                -0.04837384,
                -0.00693197,
                0.02457807,
                0.01085603,
                -0.02857602,
                0.05354919,
                0.03507602,
                0.0589912,
                -0.0418968,
                0.07301034,
                -0.03276676,
            ]
        ),
    )
    pytest.approx(
        variance,
        np.array(
            [
                0.95960407,
                1.00661311,
                0.96384865,
                1.05149678,
                0.9929845,
                1.00313217,
                1.05255451,
                1.09532161,
                0.99872863,
                0.93329299,
                0.93220725,
                1.03669406,
                0.997278,
                1.00399071,
                1.01084993,
                1.02404699,
                0.98220271,
                0.98474804,
                1.0062898,
                1.06838389,
                0.98432677,
            ]
        ),
    )


def test_joint_logpdf(parameters):
    """Test *joint_logpdf* method."""
    samples = np.ones(shape=(2, 29))
    logpdf = parameters.joint_logpdf(samples)
    pytest.approx(logpdf, np.array([-29.7977092, -29.7977092]))


def test_grad_joint_logpdf(parameters):
    """Test *joint_logpdf* method."""
    samples = np.ones(shape=(2, 29))
    grad_logpdf = parameters.grad_joint_logpdf(samples)
    pytest.approx(grad_logpdf, -np.ones(shape=(2, 21)))


def test_sample_as_dict(parameters):
    """Test *sample_as_dict* method."""
    sample = np.ones(shape=(1, 29))
    sample_dict = parameters.sample_as_dict(sample)

    np.testing.assert_almost_equal(
        np.array(list(sample_dict.values())),
        np.array(
            [
                0.39717782125387063,
                -0.09993212915495203,
                -0.6808521625113999,
                0.37336763150663344,
                -0.6931588544685283,
                -0.42405291800614264,
                0.1986562065280319,
                -0.8127228669345008,
                0.11676031012249019,
                0.34550880227254777,
                0.3882699167000142,
                0.24364631605094017,
                0.3882699167000142,
                0.43103103112748065,
                0.2864074304784066,
                0.24364631605094017,
                0.2864074304784066,
                0.14178382982933257,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
            ]
        ),
    )


def test_to_list(parameters):
    """Test *to_list* method."""
    parameters_list = parameters.to_list()
    assert isinstance(parameters_list, list)
    assert len(parameters_list) == 3


@pytest.fixture(name="pre_processor", scope="module")
def fixture_pre_processor():
    """Create basic preprocessor class instance."""

    class PreProcessor:
        """Basic preprocessor class."""

        def __init__(self):
            """Initialize."""
            g = np.meshgrid(np.linspace(0, 1, 3), np.linspace(0, 1, 3))
            grid = np.append(g[0].reshape(-1, 1), g[1].reshape(-1, 1), axis=1)
            key_list_0 = []
            key_list_1 = []
            key_list_2 = []
            for i in range(grid.shape[0]):
                key_list_0.append("x_" + str(i))
                key_list_1.append("y_" + str(i))
                key_list_2.append("z_" + str(i))
            self.coords_dict = {
                "field_1": {
                    "keys": key_list_0,
                    "coords": grid,
                },
                "field_2": {
                    "keys": key_list_1,
                    "coords": grid,
                },
                "field_3": {
                    "keys": key_list_2,
                    "coords": grid,
                },
            }

    return PreProcessor()


def test_from_config_create_parameters(parameters):
    """Test from_config_create_parameters method with random fields."""
    assert parameters.num_parameters == 29
    assert parameters.dict["field_1"].dimension == 8
    assert parameters.dict["field_2"].dimension == 12
    assert parameters.dict["field_3"].dimension == 9
    assert parameters.parameters_keys == [
        "x_0",
        "x_1",
        "x_2",
        "x_3",
        "x_4",
        "x_5",
        "x_6",
        "x_7",
        "x_8",
        "y_0",
        "y_1",
        "y_2",
        "y_3",
        "y_4",
        "y_5",
        "y_6",
        "y_7",
        "y_8",
        "z_0",
        "z_1",
        "z_2",
        "z_3",
        "z_4",
        "z_5",
        "z_6",
        "z_7",
        "z_8",
    ]
    assert parameters.random_field_flag
    assert parameters.names == ["field_1", "field_2", "field_3"]
    rf_1 = parameters.dict["field_1"]
    rf_2 = parameters.dict["field_2"]
    rf_3 = parameters.dict["field_3"]
    assert rf_1.corr_length == 0.3
    assert rf_2.corr_length == 0.3
    assert rf_3.distribution.mean[0] == 0.0

    sample_1 = np.ones(shape=(2, 9))
    sample_2 = np.ones(shape=(2, 9))
    sample_3 = np.ones(shape=(2, 9))
    grad_field_1 = rf_1.latent_gradient(sample_1)
    grad_field_2 = rf_2.latent_gradient(sample_2)
    grad_field_3 = rf_3.latent_gradient(sample_3)
    grad_dict = {
        "field_1": np.array(
            [
                [
                    -1.97560304e00,
                    1.11022302e-16,
                    -2.81025203e-16,
                    7.21644966e-16,
                    -5.73388332e-02,
                    -3.21349704e-01,
                    -1.11022302e-16,
                    3.05311332e-16,
                ],
                [
                    -1.97560304e00,
                    1.11022302e-16,
                    -2.81025203e-16,
                    7.21644966e-16,
                    -5.73388332e-02,
                    -3.21349704e-01,
                    -1.11022302e-16,
                    3.05311332e-16,
                ],
            ]
        ),
        "field_2": np.array([[0.84598704, 0.0, 0.0, 0.0], [0.84598704, 0.0, 0.0, 0.0]]),
        "field_3": np.array(
            [
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            ]
        ),
    }
    pytest.approx(grad_field_1, grad_dict["field_1"])
    pytest.approx(grad_field_2, grad_dict["field_2"])
    pytest.approx(grad_field_3, grad_dict["field_3"])

    sample_joint = np.concatenate((sample_1, sample_2, sample_3), axis=1)
    latent_grad = parameters.latent_grad(sample_joint)
    pytest.approx(latent_grad, np.concatenate((grad_field_1, grad_field_2, grad_field_3), axis=1))
