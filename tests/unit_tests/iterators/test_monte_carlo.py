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
"""Unit tests for Monte Carlo iterator."""

import numpy as np
import pytest

from queens.iterators.monte_carlo import MonteCarlo


@pytest.fixture(name="default_mc_iterator")
def fixture_default_mc_iterator(global_settings, ishigami_90_mixed, default_parameters_mixed):
    """Default monte carlo iterator."""
    # create LHS iterator
    my_iterator = MonteCarlo(
        model=ishigami_90_mixed,
        parameters=default_parameters_mixed,
        global_settings=global_settings,
        seed=42,
        num_samples=100,
        result_description=None,
    )
    return my_iterator


def test_correct_sampling(default_mc_iterator):
    """Test if we get correct samples."""
    default_mc_iterator.pre_run()

    # check if mean and std match
    means_ref = np.array([-1.8735991508e-01, -2.1607203347e-03, 2.8955130234e00])

    np.testing.assert_allclose(
        np.mean(default_mc_iterator.samples, axis=0), means_ref, 1e-09, 1e-09
    )

    std_ref = np.array([1.8598117085, 1.8167064845, 6.7786919771])
    np.testing.assert_allclose(np.std(default_mc_iterator.samples, axis=0), std_ref, 1e-09, 1e-09)

    # check if samples are identical too
    ref_sample_first_row = np.array([-0.7882876819, 0.1740941365, 1.3675241182])

    np.testing.assert_allclose(
        default_mc_iterator.samples[0, :], ref_sample_first_row, 1e-07, 1e-07
    )


def test_correct_results(default_mc_iterator):
    """Test if we get correct results."""
    default_mc_iterator.pre_run()
    default_mc_iterator.core_run()

    ref_results = np.array(
        [
            [-7.4713449052e-01],
            [3.6418728120e01],
            [1.3411821745e00],
            [1.0254005782e04],
            [-2.9330095397e00],
            [2.1639496168e00],
            [-1.1964201899e-01],
            [7.6345947125e00],
            [7.6591139616e00],
            [1.1519434320e01],
        ]
    )

    np.testing.assert_allclose(
        default_mc_iterator.output["result"][0:10], ref_results, 1e-09, 1e-09
    )
