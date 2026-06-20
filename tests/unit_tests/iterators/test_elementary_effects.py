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
"""Unit tests for the elementary effects iterator."""

import numpy as np
import pytest

from queens.iterators.elementary_effects import ElementaryEffects


@pytest.fixture(name="default_elementary_effects_iterator")
def fixture_default_elementary_effects_iterator(
    global_settings, default_simulation_model, default_parameters_uniform_3d
):
    """Default elementary effects iterator."""
    default_simulation_model.driver.parameters = default_parameters_uniform_3d

    my_iterator = ElementaryEffects(
        model=default_simulation_model,
        parameters=default_parameters_uniform_3d,
        global_settings=global_settings,
        num_trajectories=20,
        local_optimization=True,
        num_optimal_trajectories=4,
        number_of_levels=4,
        seed=42,
        confidence_level=0.95,
        num_bootstrap_samples=1000,
        result_description={},
    )
    return my_iterator


def test_correct_sampling(default_elementary_effects_iterator):
    """Test if sampling works correctly."""
    default_elementary_effects_iterator.pre_run()

    ref_vals = np.array(
        [
            [1.04719755, -1.04719755, 3.14159265],
            [1.04719755, 3.14159265, 3.14159265],
            [-3.14159265, 3.14159265, 3.14159265],
            [-3.14159265, 3.14159265, -1.04719755],
            [3.14159265, -3.14159265, -3.14159265],
            [3.14159265, 1.04719755, -3.14159265],
            [-1.04719755, 1.04719755, -3.14159265],
            [-1.04719755, 1.04719755, 1.04719755],
            [-1.04719755, -3.14159265, 3.14159265],
            [3.14159265, -3.14159265, 3.14159265],
            [3.14159265, -3.14159265, -1.04719755],
            [3.14159265, 1.04719755, -1.04719755],
            [1.04719755, 1.04719755, 3.14159265],
            [1.04719755, -3.14159265, 3.14159265],
            [-3.14159265, -3.14159265, 3.14159265],
            [-3.14159265, -3.14159265, -1.04719755],
        ]
    )

    np.testing.assert_allclose(default_elementary_effects_iterator.samples, ref_vals, 1e-07, 1e-07)


def test_correct_sensitivity_indices(default_elementary_effects_iterator):
    """Test correct results."""
    default_elementary_effects_iterator.pre_run()
    default_elementary_effects_iterator.core_run()
    si = default_elementary_effects_iterator.si

    ref_mu = np.array([13.952850214926777, 3.9375000000003153, 3.1243980516682517])
    ref_mu_star = np.array([13.952850214926777, 7.875000000000625, 3.124398051669744])
    ref_mu_star_conf = np.array(
        [6.8719813972395294e-15, 3.0060819460414646e-15, 5.3587332489494415]
    )
    ref_sigma = np.array([3.24316904e-15, 7.87500000e00, 6.24879610e00])

    np.testing.assert_allclose(si["mu"], ref_mu, 1e-07, 1e-07)
    np.testing.assert_allclose(si["mu_star"], ref_mu_star, 1e-07, 1e-07)
    np.testing.assert_allclose(si["mu_star_conf"], ref_mu_star_conf, 1e-07, 1e-07)
    np.testing.assert_allclose(si["sigma"], ref_sigma, 1e-07, 1e-07)
