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
"""Integration test for the data iterator."""

import pytest

from queens.iterators.data import Data
from queens.main import run_iterator
from queens.parameters.parameters import Parameters
from queens.utils.io import load_result


def test_data(mocker, ref_result_iterator, global_settings):
    """Test case for data iterator."""
    output = {}
    output["result"] = ref_result_iterator

    samples = ref_result_iterator

    mocker.patch("queens.iterators.data.Data.read_pickle_file", return_value=[samples, output])

    parameters = Parameters()

    # Setup iterator
    iterator = Data(
        path_to_data="/path_to_data/some_data.pickle",
        result_description={
            "write_results": True,
            "plot_results": False,
            "num_support_points": 5,
        },
        parameters=parameters,
        global_settings=global_settings,
    )

    # Actual analysis
    run_iterator(iterator, global_settings=global_settings)
    # Load results
    results = load_result(global_settings.result_file(".pickle"))

    assert results["mean"] == pytest.approx(1.3273452195599997)
    assert results["var"] == pytest.approx(44.82468751096612)
