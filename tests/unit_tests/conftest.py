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
"""Fixtures needed across unit_tests."""

from unittest.mock import Mock, patch

import numpy as np
import pytest

from queens.iterators.bmfia import BMFIA
from queens.models.simulation import Simulation


@pytest.fixture(name="dummy_simulation_model")
def fixture_dummy_simulation_model():
    """Dummy model."""
    model = Simulation(scheduler=Mock(), driver=Mock())
    return model


@pytest.fixture(name="experiment_name")
def fixture_experiment_name():
    """Fixture for the experiment_name."""
    return "test_experiment"


@pytest.fixture(name="files_to_copy")
def fixture_files_to_copy():
    """Files to copy."""
    return ["fileA", "fileB"]


@pytest.fixture(name="get_patched_bmfia_iterator")
def fixture_get_patched_bmfia_iterator(global_settings):
    """Function that returns a dummy BMFIA iterator for testing."""

    def get_patched_bmfia_iterator(parameters, hf_model, lf_model):
        x_train = np.array([[1, 2], [3, 4]])
        features_config = "no_features"
        x_cols = None
        num_features = None
        coord_cols = None

        with patch.object(BMFIA, "calculate_initial_x_train", lambda *args: x_train):
            iterator = BMFIA(
                parameters=parameters,
                global_settings=global_settings,
                features_config=features_config,
                hf_model=hf_model,
                lf_model=lf_model,
                initial_design={},
                X_cols=x_cols,
                num_features=num_features,
                coord_cols=coord_cols,
            )

        iterator.Y_LF_train = np.array([[2], [3]])
        iterator.Y_HF_train = np.array([[2.2], [3.3]])
        iterator.Z_train = np.array([[4], [5]])
        iterator.coords_experimental_data = np.array([[1, 2], [3, 4]])
        iterator.time_vec = np.array([1, 3])
        iterator.y_obs_vec = np.array([[2.1], [3.1]])

        return iterator

    return get_patched_bmfia_iterator


@pytest.fixture(name="result_description")
def fixture_result_description():
    """A dummy result description."""
    description = {"write_results": True}
    return description
