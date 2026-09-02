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
"""Unit test for Sobol index iterator."""

import numpy as np
import pytest

from queens.iterators.sobol_index import SobolIndex


@pytest.fixture(name="default_sobol_index_iterator")
def fixture_default_sobol_index_iterator(
    global_settings, ishigami_90_uniform, default_parameters_uniform_3d
):
    """Default sobol index iterator."""
    my_iterator = SobolIndex(
        ishigami_90_uniform,
        parameters=default_parameters_uniform_3d,
        global_settings=global_settings,
        seed=42,
        num_samples=2,
        calc_second_order=True,
        num_bootstrap_samples=1000,
        confidence_level=0.95,
        result_description={},
        skip_values=1024,
    )
    return my_iterator


@pytest.fixture(name="default_sobol_index_iterator_mixed")
def fixture_default_sobol_index_iterator_mixed(
    global_settings, ishigami_90_mixed, default_parameters_mixed
):
    """Default sobol index iterator with different distributions."""
    my_iterator = SobolIndex(
        model=ishigami_90_mixed,
        parameters=default_parameters_mixed,
        global_settings=global_settings,
        seed=42,
        num_samples=2,
        calc_second_order=True,
        num_bootstrap_samples=1000,
        confidence_level=0.95,
        result_description={},
        skip_values=1024,
    )
    return my_iterator


@pytest.mark.parametrize(
    "fixture_sobol_index_iterator, ref_vals",
    [
        # Test case for fixture_default_sobol_index_iterator
        (
            "default_sobol_index_iterator",
            np.array(
                [
                    [-3.1323887689, -0.7761942787, -0.3282718886],
                    [-0.0828349625, -0.7761942787, -0.3282718886],
                    [-3.1323887689, 0.3589515044, -0.3282718886],
                    [-3.1323887689, -0.7761942787, 2.1629129109],
                    [-3.1323887689, 0.3589515044, 2.1629129109],
                    [-0.0828349625, -0.7761942787, 2.1629129109],
                    [-0.0828349625, 0.3589515044, -0.3282718886],
                    [-0.0828349625, 0.3589515044, 2.1629129109],
                    [0.0092038847, 2.3653983749, 2.8133207650],
                    [3.0587576910, 2.3653983749, 2.8133207650],
                    [0.0092038847, -2.7826411492, 2.8133207650],
                    [0.0092038847, 2.3653983749, -0.9786797427],
                    [0.0092038847, -2.7826411492, -0.9786797427],
                    [3.0587576910, 2.3653983749, -0.9786797427],
                    [3.0587576910, -2.7826411492, 2.8133207650],
                    [3.0587576910, -2.7826411492, -0.9786797427],
                ]
            ),
        ),
        # Test case for fixture_default_sobol_index_iterator_mixed with mixed distributions
        (
            "default_sobol_index_iterator_mixed",
            np.array(
                [
                    [-3.1323887688626857, -0.6295573938365613, 1.183719794407078],
                    [-0.08283496254583023, -0.6295573938365613, 1.183719794407078],
                    [-3.1323887688626857, 0.28738780203495984, 1.183719794407078],
                    [-3.1323887688626857, -0.6295573938365613, 3.71370699996047],
                    [-3.1323887688626857, 0.28738780203495984, 3.71370699996047],
                    [-0.08283496254583023, -0.6295573938365613, 3.71370699996047],
                    [-0.08283496254583023, 0.28738780203495984, 1.183719794407078],
                    [-0.08283496254583023, 0.28738780203495984, 3.71370699996047],
                    [0.009203884727314371, 2.314989355551835, 6.84456525569755],
                    [3.0587576910441703, 2.314989355551835, 6.84456525569755],
                    [0.009203884727314371, -3.158682435228621, 6.84456525569755],
                    [0.009203884727314371, 2.314989355551835, 0.9040023275536164],
                    [0.009203884727314371, -3.158682435228621, 0.9040023275536164],
                    [3.0587576910441703, 2.314989355551835, 0.9040023275536164],
                    [3.0587576910441703, -3.158682435228621, 6.84456525569755],
                    [3.0587576910441703, -3.158682435228621, 0.9040023275536164],
                ]
            ),
        ),
    ],
)
def test_correct_sampling(fixture_sobol_index_iterator, ref_vals, request):
    """Test correct sampling."""
    sobol_index_iterator = request.getfixturevalue(fixture_sobol_index_iterator)
    sobol_index_iterator.pre_run()
    np.testing.assert_allclose(sobol_index_iterator.samples, ref_vals, 1e-07, 1e-07)


def test_correct_sensitivity_indices(default_sobol_index_iterator):
    """Test Sobol indices results."""
    default_sobol_index_iterator.pre_run()
    default_sobol_index_iterator.core_run()
    si = default_sobol_index_iterator.sensitivity_indices

    ref_s1 = np.array([-0.1500205418, 2.0052373431, 0.0282147215])
    ref_s1_conf = np.array([0.3189669552, 0.0599864170, 0.0186884010])

    ref_st = np.array([0.0399037376, 1.8151878529, 0.0004985237])
    ref_st_conf = np.array([0.0589769471, 0.2607858609, 0.0006083827])

    ref_s2 = np.array(
        [
            [np.nan, 0.0915118400, 0.2003003441],
            [np.nan, np.nan, -0.2712666971],
            [np.nan, np.nan, np.nan],
        ]
    )
    ref_s2_conf = np.array(
        [
            [np.nan, 0.4690527908, 0.1802601772],
            [np.nan, np.nan, 0.3460941042],
            [np.nan, np.nan, np.nan],
        ]
    )

    np.testing.assert_allclose(si["S1"], ref_s1, 1e-07, 1e-07)
    np.testing.assert_allclose(si["S1_conf"], ref_s1_conf, 1e-07, 1e-07)

    np.testing.assert_allclose(si["ST"], ref_st, 1e-07, 1e-07)
    np.testing.assert_allclose(si["ST_conf"], ref_st_conf, 1e-07, 1e-07)

    np.testing.assert_allclose(si["S2"], ref_s2, 1e-07, 1e-07)
    np.testing.assert_allclose(si["S2_conf"], ref_s2_conf, 1e-07, 1e-07)
