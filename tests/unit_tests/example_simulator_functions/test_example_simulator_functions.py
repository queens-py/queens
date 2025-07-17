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
"""Test module for example simulator functions."""

import numpy as np
import pytest

from example_simulator_functions import (
    VALID_EXAMPLE_SIMULATOR_FUNCTIONS,
    example_simulator_function_by_name,
)

# Dictionary of the functions to be tested
TEST_DICT = {
    "agawal09a": {
        "input": {"x1": 0.4, "x2": 0.4},
        "desired_result": 0.90450849718747361,
    },
    "borehole83_lofi": {
        "input": {
            "rw": 0.1,
            "r": 500,
            "tu": 70000,
            "hu": 1000,
            "tl": 80,
            "hl": 750,
            "l": 1550,
            "kw": 11100,
        },
        "desired_result": 44.58779860979928,
    },
    "borehole83_hifi": {
        "input": {
            "rw": 0.1,
            "r": 500,
            "tu": 70000,
            "hu": 1000,
            "tl": 80,
            "hl": 750,
            "l": 1550,
            "kw": 11100,
        },
        "desired_result": 56.03080181188316,
    },
    "branin78_lofi": {
        "input": {"x1": -4, "x2": 5},
        "desired_result": 1.4307273110713652,
    },
    "branin78_medfi": {
        "input": {"x1": -4, "x2": 5},
        "desired_result": 125.49860898539086,
    },
    "branin78_hifi": {
        "input": {"x1": -4, "x2": 5},
        "desired_result": 92.70795679406056,
    },
    "currin88_lofi": {
        "input": {"x1": 0.6, "x2": 0.1},
        "desired_result": 10.964538831722423,
    },
    "currin88_hifi": {
        "input": {"x1": 0.6, "x2": 0.1},
        "desired_result": 11.06777716201019,
    },
    "gardner14a": {
        "input": {"x1": 1.234, "x2": 0.666},
        "desired_result": np.array([0.32925795427636007, -0.32328956686350346]),
    },
    "ishigami90": {
        "input": {"x1": 0.1, "x2": 0.23, "x3": 0.4, "p1": 7, "p2": 0.1},
        "desired_result": 0.4639052488541057,
    },
    "ma09": {
        "input": {"x1": 0.25, "x2": 0.5},
        "desired_result": 8.8888888888888875,
    },
    "oakley_ohagan04": {
        "input": {
            "x1": 0.3,
            "x2": 0.6,
            "x3": 0.5,
            "x4": 0.1,
            "x5": 0.9,
            "x6": 0.3,
            "x7": 0.6,
            "x8": 0.5,
            "x9": 0.1,
            "x10": 0.9,
            "x11": 0.3,
            "x12": 0.6,
            "x13": 0.5,
            "x14": 0.1,
            "x15": 0.9,
        },
        "desired_result": 24.496726490699082,
    },
    "paraboloid": {
        "input": {"x1": 0.6, "x2": 0.9},
        "desired_result": 2.7200000000000006,
    },
    "parabola_residual": {"input": {"x1": 0.6}, "desired_result": np.array([3.0])},
    "park91a_lofi_on_grid": {
        "input": {"x1": 0.6, "x2": 0.9},
        "desired_result": np.array(
            [
                1.94143388,
                2.73503973,
                3.91802798,
                5.27245202,
                4.86331549,
                6.71854136,
                9.23991222,
                11.89896363,
                7.85237331,
                10.79803021,
                14.68682775,
                18.67089934,
                10.73924667,
                14.74096618,
                19.95148794,
                25.21413295,
            ]
        ),
    },
    "park91a_hifi_on_grid": {
        "input": {"x1": 0.6, "x2": 0.9},
        "desired_result": np.array(
            [
                1.73345561,
                2.37956809,
                3.18381199,
                3.95882022,
                4.49917307,
                6.15016577,
                8.22126049,
                10.23116841,
                7.32847641,
                10.01162049,
                13.37705781,
                16.6411683,
                10.06105667,
                13.74382059,
                18.36034068,
                22.8346894,
            ]
        ),
    },
    "park91a_hifi_on_grid_with_gradients": {
        "input": {"x1": 0.6, "x2": 0.9},
        "desired_result": (
            np.array(
                [
                    1.73345561,
                    2.37956809,
                    3.18381199,
                    3.95882022,
                    4.49917307,
                    6.15016577,
                    8.22126049,
                    10.23116841,
                    7.32847641,
                    10.01162049,
                    13.37705781,
                    16.6411683,
                    10.06105667,
                    13.74382059,
                    18.36034068,
                    22.8346894,
                ]
            ),
            (
                np.array(
                    [
                        2.73946469,
                        3.7635742,
                        5.0357767,
                        6.2589992,
                        2.61486571,
                        3.62978115,
                        4.87857452,
                        6.07358046,
                        2.55177614,
                        3.56546718,
                        4.81251441,
                        6.0077492,
                        2.51381084,
                        3.52760463,
                        4.77556048,
                        5.97285327,
                    ]
                ),
                np.array(
                    [
                        0.00411553,
                        0.00410936,
                        0.00409098,
                        0.00406195,
                        0.10257365,
                        0.09981385,
                        0.09269862,
                        0.08389309,
                        0.17009755,
                        0.16389772,
                        0.14869717,
                        0.13121326,
                        0.22127348,
                        0.21215152,
                        0.19032831,
                        0.16606255,
                    ]
                ),
            ),
        ),
    },
    "park91a_lofi_on_grid_with_gradients": {
        "input": {"x1": 0.6, "x2": 0.9},
        "desired_result": (
            np.array(
                [
                    1.941434,
                    2.73504,
                    3.918028,
                    5.272452,
                    4.863315,
                    6.718541,
                    9.239912,
                    11.898964,
                    7.852373,
                    10.79803,
                    14.686828,
                    18.670899,
                    10.739247,
                    14.740966,
                    19.951488,
                    25.214133,
                ]
            ),
            (
                np.array(
                    [
                        1.03721476,
                        2.17247582,
                        3.58288938,
                        4.93914441,
                        1.13384491,
                        2.3423291,
                        3.83256947,
                        5.26093538,
                        1.30070552,
                        2.5930833,
                        4.18830563,
                        5.72042713,
                        1.48612611,
                        2.86111463,
                        4.56055322,
                        6.19473418,
                    ]
                ),
                np.array(
                    [
                        1.80434791,
                        1.80434139,
                        1.80432197,
                        1.8042913,
                        1.9083654,
                        1.90544977,
                        1.89793278,
                        1.88863005,
                        1.97970198,
                        1.97315209,
                        1.95709325,
                        1.93862212,
                        2.03376752,
                        2.02413049,
                        2.00107506,
                        1.97543914,
                    ]
                ),
            ),
        ),
    },
    "park91a_lofi": {
        "input": {"x1": 0.3, "x2": 0.6, "x3": 0.5, "x4": 0.1},
        "desired_result": 3.2830146685714103,
    },
    "park91a_hifi": {
        "input": {"x1": 0.3, "x2": 0.6, "x3": 0.5, "x4": 0.1},
        "desired_result": 2.6934187033863846,
    },
    "park91b_lofi": {
        "input": {"x1": 0.3, "x2": 0.6, "x3": 0.5, "x4": 0.1},
        "desired_result": 1.510151424293055,
    },
    "park91b_hifi": {
        "input": {"x1": 0.3, "x2": 0.6, "x3": 0.5, "x4": 0.1},
        "desired_result": 2.091792853577546,
    },
    "perdikaris17_hifi": {"input": {"x": 0.6}, "desired_result": -0.2813038672746218},
    "perdikaris17_lofi": {"input": {"x": 0.6}, "desired_result": 0.5877852522924737},
    "rosenbrock60": {
        "input": {"x1": 0.6, "x2": 0.9},
        "desired_result": 29.320000000000004,
    },
    "rosenbrock60_residual": {
        "input": {"x1": 0.6, "x2": 0.9},
        "desired_result": np.array([5.4, 0.4]),
    },
    "rosenbrock60_residual_1d": {
        "input": {"x1": 0.6},
        "desired_result": np.array([6.4, 0.4]),
    },
    "rosenbrock60_residual_3d": {
        "input": {"x1": 0.3, "x2": 0.6, "x3": 0.5},
        "desired_result": np.array([5.1, 0.7, 0.25]),
    },
    "sinus_test_fun": {"input": {"x1": 0.6}, "desired_result": 0.5646424733950354},
    "sobol_g_function": {
        "input": {
            "x1": 0.1,
            "x2": 0.23,
            "x3": 0.4,
            "x4": 0.6,
            "x5": 0.1,
            "x6": 0.25,
            "x7": 0.98,
            "x8": 0.7,
            "a": np.array([0, 1, 4.5, 9, 99, 99, 99, 99]),
            "alpha": np.array([1.0] * 8),
            "delta": np.array([0.0] * 8),
        },
        "desired_result": 1.4119532907954928,
    },
    "patch_for_likelihood": {
        "input": {
            "x": 0,
        },
        "desired_result": 42,
    },
}


@pytest.mark.unit_tests
@pytest.mark.parametrize(
    "function_name,test_dict",
    TEST_DICT.items(),
)
def test_example_simulator_functions(function_name, test_dict):
    """Test the example simulator functions."""
    function = example_simulator_function_by_name(function_name)
    inputs = test_dict["input"]
    desired_result = test_dict["desired_result"]
    if isinstance(desired_result, tuple):
        np.testing.assert_allclose(function(**inputs)[0], desired_result[0], atol=1e-6)
        np.testing.assert_allclose(function(**inputs)[1], desired_result[1], atol=1e-6)
    else:
        np.testing.assert_allclose(function(**inputs), desired_result)


@pytest.mark.unit_tests
@pytest.mark.parametrize(
    "function",
    list(VALID_EXAMPLE_SIMULATOR_FUNCTIONS.keys()),
)
def test_if_all_example_simulator_functions_are_tested(function):
    """Test if all the example simulator functions are being tested."""
    assert function in TEST_DICT
