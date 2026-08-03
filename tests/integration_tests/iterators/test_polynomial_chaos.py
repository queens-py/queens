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
"""Integration test for the Polynomial Chaos iterator."""

import pytest

from queens.iterators.polynomial_chaos import (
    PolynomialChaos,
    has_macos_numpoly_reshape_mismatch,
)
from queens.main import run_iterator
from queens.utils.io import load_result

pytestmark = pytest.mark.skipif(
    has_macos_numpoly_reshape_mismatch(),
    reason=(
        "Skipped on macOS only for the known downstream numpoly/NumPy mismatch: "
        "numpoly < 1.3.9 calls numpy.reshape(..., newshape=...), which this NumPy rejects."
    ),
)


def test_polynomial_chaos_pseudo_spectral_borehole(
    global_settings, borehole_parameters, borehole83_lofi_model
):
    """Test case for the PC iterator using a pseudo spectral approach."""
    iterator = PolynomialChaos(
        approach="pseudo_spectral",
        seed=42,
        num_collocation_points=50,
        sampling_rule="gaussian",
        sparse=True,
        polynomial_order=2,
        result_description={"write_results": True},
        model=borehole83_lofi_model,
        parameters=borehole_parameters,
        global_settings=global_settings,
    )

    # Actual analysis
    run_iterator(iterator, global_settings=global_settings)

    # Load results
    results = load_result(global_settings.result_file(".pickle"))
    assert results["mean"] == pytest.approx(61.78966587)
    assert results["covariance"] == pytest.approx([1312.23414971])


def test_polynomial_chaos_collocation_borehole(
    global_settings, borehole_parameters, borehole83_lofi_model
):
    """Test for the PC iterator using a collocation approach."""
    iterator = PolynomialChaos(
        approach="collocation",
        seed=42,
        num_collocation_points=50,
        sampling_rule="sobol",
        polynomial_order=2,
        result_description={"write_results": True},
        model=borehole83_lofi_model,
        parameters=borehole_parameters,
        global_settings=global_settings,
    )

    # Actual analysis
    run_iterator(iterator, global_settings=global_settings)

    # Load results
    results = load_result(global_settings.result_file(".pickle"))
    assert results["mean"] == pytest.approx(62.05018243)
    assert results["covariance"] == pytest.approx([1273.81372103])
