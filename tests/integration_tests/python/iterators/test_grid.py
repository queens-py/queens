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
"""Integration test for the grid iterator."""

import numpy as np
import pytest

from queens.distributions.uniform import Uniform
from queens.drivers.function import Function
from queens.iterators.grid import Grid
from queens.main import run_iterator
from queens.models.simulation import Simulation
from queens.parameters.parameters import Parameters
from queens.schedulers.pool import Pool
from queens.utils.io import load_result


def test_grid_rosenbrock60(expected_response, expected_grid, global_settings, tmp_path):
    """Integration test for the grid iterator."""
    # Parameters
    x1 = Uniform(lower_bound=-2.0, upper_bound=2.0)
    x2 = Uniform(lower_bound=-2.0, upper_bound=2.0)
    parameters = Parameters(x1=x1, x2=x2)

    # Setup iterator
    driver = Function(parameters=parameters, function="rosenbrock60")
    scheduler = Pool(experiment_name=global_settings.experiment_name)
    model = Simulation(scheduler=scheduler, driver=driver)
    iterator = Grid(
        grid_design={
            "x1": {"num_grid_points": 5, "axis_type": "lin", "data_type": "FLOAT"},
            "x2": {"num_grid_points": 5, "axis_type": "lin", "data_type": "FLOAT"},
        },
        result_description={
            "write_results": True,
            "plotting_options": {
                "plot_booleans": [True],
                "plotting_dir": tmp_path,
                "plot_names": ["grid_plot.eps"],
                "save_bool": [True],
            },
        },
        model=model,
        parameters=parameters,
        global_settings=global_settings,
    )

    # Actual analysis
    run_iterator(iterator, global_settings=global_settings)

    # Load results
    results = load_result(global_settings.result_file(".pickle"))

    np.testing.assert_array_equal(
        results["raw_output_data"]["result"],
        expected_response,
    )

    np.testing.assert_allclose(results["input_data"], expected_grid, rtol=1.0e-3)


@pytest.fixture(name="expected_grid")
def fixture_expected_grid():
    """Expected grid coordinates."""
    input_data = np.array(
        [
            [-2.000, -2.000],
            [-1.000, -2.000],
            [0.000, -2.000],
            [1.000, -2.000],
            [2.000, -2.000],
            [-2.000, -1.000],
            [-1.000, -1.000],
            [0.000, -1.000],
            [1.000, -1.000],
            [2.000, -1.000],
            [-2.000, 0.000],
            [-1.000, 0.000],
            [0.000, 0.000],
            [1.000, 0.000],
            [2.000, 0.000],
            [-2.000, 1.000],
            [-1.000, 1.000],
            [0.000, 1.000],
            [1.000, 1.000],
            [2.000, 1.000],
            [-2.000, 2.000],
            [-1.000, 2.000],
            [0.000, 2.000],
            [1.000, 2.000],
            [2.000, 2.000],
        ]
    )
    return input_data


@pytest.fixture(name="expected_response")
def fixture_expected_response():
    """Expected response values."""
    expected_response = np.atleast_2d(
        np.array(
            [
                3.609e03,
                9.040e02,
                4.010e02,
                9.000e02,
                3.601e03,
                2.509e03,
                4.040e02,
                1.010e02,
                4.000e02,
                2.501e03,
                1.609e03,
                1.040e02,
                1.000e00,
                1.000e02,
                1.601e03,
                9.090e02,
                4.000e00,
                1.010e02,
                0.000e00,
                9.010e02,
                4.090e02,
                1.040e02,
                4.010e02,
                1.000e02,
                4.010e02,
            ]
        )
    ).T

    return expected_response
