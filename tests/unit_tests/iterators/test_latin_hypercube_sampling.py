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
"""Unit tests for LHS iterator."""

import numpy as np
import pytest

from queens.iterators.latin_hypercube_sampling import LatinHypercubeSampling


@pytest.fixture(name="default_lhs_iterator")
def fixture_default_lhs_iterator(global_settings, ishigami_90_mixed, default_parameters_mixed):
    """Default latin hypercube sampling iterator."""
    # create LHS iterator
    # pylint: disable=duplicate-code
    my_iterator = LatinHypercubeSampling(
        model=ishigami_90_mixed,
        parameters=default_parameters_mixed,
        global_settings=global_settings,
        seed=42,
        num_samples=100,
        num_iterations=1,
        result_description=None,
        criterion="maximin",
    )
    return my_iterator


@pytest.fixture(name="ref_lhs_result_iterator")
def fixture_ref_lhs_result_iterator():
    """Reference results for the default LHS iterator."""
    return np.array(
        [
            [0.5401856501906925],
            [4.305418063074407],
            [6.218626875992211],
            [-7.341084114985484],
            [6.2263184418260975],
            [5.992809151839016],
            [8.676959923988381],
            [2.187257549384814],
            [6.521135693744268],
            [6.0112492830277136],
        ]
    )


def test_correct_sampling(default_lhs_iterator):
    """Test if we get correct samples."""
    # np.set_printoptions(precision=10)
    default_lhs_iterator.pre_run()

    # check if mean and std match
    means_ref = np.array([3.5708636985e-04, 9.4366122545e-03, 2.2065845877e00])

    np.testing.assert_allclose(
        np.mean(default_lhs_iterator.samples, axis=0), means_ref, 1e-09, 1e-09
    )

    std_ref = np.array([1.8152930704, 1.9637665236, 2.6752150172])
    np.testing.assert_allclose(np.std(default_lhs_iterator.samples, axis=0), std_ref, 1e-09, 1e-09)

    # check if samples are identical too
    ref_sample_first_row = np.array([-1.9557761949, 0.5130876432, 1.2407952238])

    np.testing.assert_allclose(
        default_lhs_iterator.samples[0, :], ref_sample_first_row, 1e-07, 1e-07
    )


def test_correct_results(default_lhs_iterator, ref_lhs_result_iterator):
    """Test if we get correct results."""
    default_lhs_iterator.pre_run()
    default_lhs_iterator.core_run()

    # np.set_printoptions(precision=10)

    # check if samples are identical too

    np.testing.assert_allclose(
        default_lhs_iterator.output["result"][0:10], ref_lhs_result_iterator, 1e-09, 1e-09
    )
