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
"""Utility methods used by the integration tests.."""

import numpy as np


def assert_monte_carlo_iterator_results(results, expected_mean, expected_var):
    """Assert the equality of the results with the expected values.

    Args:
        results (dict): Results dictionary from pickle file
        expected_mean (np.ndarray): Expected mean of the results
        expected_var (np.ndarray): Expected variance of the results
    """
    np.testing.assert_array_almost_equal(
        results["raw_output_data"]["result"], expected_mean, decimal=4
    )
    np.testing.assert_array_almost_equal(
        results["raw_output_data"]["variance"], expected_var, decimal=2
    )


def assert_surrogate_model_output(
    output, mean_ref, var_ref, grad_mean_ref=None, grad_var_ref=None, decimals=(2, 2, 2, 2)
):
    """Assert the equality of the output with the provided reference values.

    Args:
        output (dict): surrogate model output
        mean_ref (np.ndarray): reference mean
        var_ref (np.ndarray): reference variance
        grad_mean_ref (np.ndarray): reference gradient of the mean
        grad_var_ref (np.ndarray): reference gradient of the variance
        decimals (lst): list of desired decimal precisions
    """
    mean = output["result"]
    variance = output["variance"]

    np.testing.assert_array_almost_equal(mean, mean_ref, decimal=decimals[0])
    np.testing.assert_array_almost_equal(variance, var_ref, decimal=decimals[1])

    if grad_mean_ref is not None:
        gradient_mean = output["grad_mean"]
        np.testing.assert_array_almost_equal(gradient_mean, grad_mean_ref, decimal=decimals[2])

    if grad_var_ref is not None:
        gradient_variance = output["grad_var"]
        np.testing.assert_array_almost_equal(gradient_variance, grad_var_ref, decimal=decimals[3])


def get_input_park91a(n_inputs):
    """Get inputs for the park91a benchmark function.

    Args:
        n_inputs (int): Number of inputs along x_1 and x_2

    Returns:
        np.ndarray, float, float: [x_1, x_2].T, x_3, x_4
    """
    x_3, x_4 = 0.5, 0.5
    x_1 = np.linspace(0.001, 0.999, n_inputs)
    x_2 = np.linspace(0.001, 0.999, n_inputs)
    xx_1, xx_2 = np.meshgrid(x_1, x_2)
    x_1_and_2 = np.vstack((xx_1.flatten(), xx_2.flatten())).T

    return x_1_and_2, x_3, x_4


def fourc_build_paths_from_home(home):
    """Paths of 4C executables from home on testing machine."""
    fourc_build_path = home / "workspace/fourc_build"
    fourc = fourc_build_path / "4C"
    post_ensight = fourc_build_path / "post_ensight"
    post_processor = fourc_build_path / "post_processor"
    return fourc, post_ensight, post_processor
