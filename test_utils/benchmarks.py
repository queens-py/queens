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
"""Utility methods used by the benchmark tests."""

import numpy as np


def assert_weights_and_samples(results, expected_weights, expected_samples):
    """Assert the equality of some SMC results and the expected values.

    Args:
        results (dict): Results dictionary from pickle file
        expected_weights (np.array): Expected weights of the posterior samples. One weight for each
                                     sample row.
        expected_samples (np.array): Expected samples of the posterior. Each row is a different
                                     sample-vector. Different columns represent the different
                                     dimensions of the posterior.
    """
    samples = results["raw_output_data"]["particles"].squeeze()
    weights = results["raw_output_data"]["weights"].squeeze()

    np.testing.assert_array_almost_equal(weights, expected_weights, decimal=5)
    np.testing.assert_array_almost_equal(samples, expected_samples, decimal=5)
