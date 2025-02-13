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
"""Test-module for distributions."""

import numpy as np

from queens.distributions.normal import Normal


def test_create_export_dict():
    """Test creation routine of distribution dict."""
    distribution = Normal(mean=0.0, covariance=1.0)
    exported_dict = distribution.export_dict()
    ref_dict = {
        "type": "Normal",
        "mean": np.array([0.0]),
        "covariance": np.array([[1.0]]),
        "dimension": 1,
        "low_chol": np.array([[1.0]]),
        "precision": np.array([[1.0]]),
        "logpdf_const": np.array([-0.9189385332046728]),
    }
    for (key, value), (key_ref, value_ref) in zip(exported_dict.items(), ref_dict.items()):
        assert key == key_ref
        if key == "type":
            assert value == value_ref
        else:
            np.testing.assert_allclose(value, value_ref)
