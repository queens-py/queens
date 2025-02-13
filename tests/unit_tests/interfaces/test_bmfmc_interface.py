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
"""Test BMFMC interface."""

import numpy as np
import pytest

from queens.models.bmfm import BmfmcInterface

# pylint: disable=invalid-name


# -------- fixtures -----------------------------------
class FakeRegression:
    """FakeRegression class."""

    def __init__(self, map_output_dict):
        """Initialize FakeRegression."""
        self.map_output_dict = map_output_dict

    def predict(self, *_args, **_kwargs):
        """Predict output."""
        output = self.map_output_dict
        return output

    def train(self):
        """Train surrogate model."""

    def setup(self, *args, **kwargs):
        """Setup surrogate model."""


@pytest.fixture(name="config")
def fixture_config(approx_name):
    """Create dummy config for testing."""
    config = {
        approx_name: {"surrogate_model_name": "gp"},
        "gp": {
            "type": "gp_approximation_gpflow",
            "features_config": "opt_features",
            "num_features": 1,
            "X_cols": 1,
        },
    }
    return config


@pytest.fixture(name="default_interface")
def fixture_default_interface(probabilistic_mapping_obj):
    """Create default interface."""
    interface = BmfmcInterface(probabilistic_mapping_obj)
    return interface


@pytest.fixture(name="probabilistic_mapping_obj")
def fixture_probabilistic_mapping_obj(map_output_dict):
    """Create probabilistic mapping object."""
    probabilistic_mapping_obj = FakeRegression(map_output_dict)
    return probabilistic_mapping_obj


@pytest.fixture(name="map_output_dict")
def fixture_map_output_dict():
    """Map output dictionary."""
    output = {"result": np.linspace(1.0, 5.0, 5), "variance": np.linspace(5.0, 10.0, 5)}
    return output


@pytest.fixture(name="approx_name")
def fixture_approx_name():
    """Create approximation name."""
    name = "some_name"
    return name


# --------- actual unit_tests ---------------------------
def test_init():
    """Test initialization."""
    approx = "dummy_approx"
    interface = BmfmcInterface(approx)

    # asserts / tests
    assert interface.probabilistic_mapping == approx


def test_map(default_interface, map_output_dict):
    """Test mapping."""
    Z_LF = 1.0
    expected_Y_HF_mean = map_output_dict["result"]
    expected_Y_HF_var = map_output_dict["variance"]

    mean_Y_HF_given_Z_LF, var_Y_HF_given_Z_LF = default_interface.evaluate(Z_LF)

    with pytest.raises(RuntimeError):
        default_interface.probabilistic_mapping = None
        default_interface.evaluate(Z_LF)

    np.testing.assert_array_almost_equal(mean_Y_HF_given_Z_LF, expected_Y_HF_mean, decimal=6)
    np.testing.assert_array_almost_equal(var_Y_HF_given_Z_LF, expected_Y_HF_var, decimal=6)


def test_build_approximation(mocker, default_interface):
    """Test training of surrogate model."""
    Z = np.atleast_2d(np.linspace(0.0, 1.0, 10))
    Y = np.atleast_2d(np.linspace(1.0, 2.0, 10))
    mp1 = mocker.patch.object(FakeRegression, "setup")
    mp2 = mocker.patch.object(FakeRegression, "train")

    default_interface.build_approximation(Z, Y)
    mp1.assert_called_once()
    mp2.assert_called_once()
