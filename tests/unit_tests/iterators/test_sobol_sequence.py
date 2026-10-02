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
"""Unit tests for sobol sequence iterator."""

import numpy as np
import pytest

from queens.iterators.sobol_sequence import SobolSequence


@pytest.fixture(name="default_qmc_iterator")
def fixture_default_qmc_iterator(global_settings, ishigami_90_mixed, default_parameters_mixed):
    """Sobol sequence iterator."""
    my_iterator = SobolSequence(
        model=ishigami_90_mixed,
        parameters=default_parameters_mixed,
        global_settings=global_settings,
        seed=42,
        number_of_samples=100,
        randomize=True,
        result_description={},
    )
    return my_iterator


def test_correct_sampling(default_qmc_iterator):
    """Test if we get correct samples."""
    default_qmc_iterator.pre_run()

    # check if mean and std match
    means_ref = np.array([-0.000981751215250255, 0.002827691955533891, 2.211574656721489])

    np.testing.assert_allclose(
        np.mean(default_qmc_iterator.samples, axis=0), means_ref, 1e-09, 1e-09
    )

    std_ref = np.array([1.8030798500938032, 2.0254290820900027, 2.617797964759257])
    np.testing.assert_allclose(np.std(default_qmc_iterator.samples, axis=0), std_ref, 1e-09, 1e-09)

    # check if samples are identical too
    ref_sample_first_row = np.array([-0.4333545933125702, 1.788216201875851, 3.205130570125655])

    np.testing.assert_allclose(
        default_qmc_iterator.samples[0, :], ref_sample_first_row, 1e-07, 1e-07
    )


def test_correct_results(default_qmc_iterator):
    """Test if we get correct results."""
    default_qmc_iterator.pre_run()
    default_qmc_iterator.core_run()

    # check if results are identical too
    ref_results = np.array(
        [
            1.8229016260009976,
            1.639467613025844,
            2.2243805849936695,
            6.085953170625794,
            5.468098968335089,
            88.62846222355823,
            2.457108989824364,
            1.2278235961901531,
            0.46305177903124134,
            6.734923724227602,
        ]
    )

    np.testing.assert_allclose(
        default_qmc_iterator.output["result"][0:10].flatten(), ref_results, 1e-09, 1e-09
    )
