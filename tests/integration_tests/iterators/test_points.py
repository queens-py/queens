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
"""Integration test for the points iterator."""

import numpy as np
import pytest

from example_simulator_functions.rosenbrock60 import rosenbrock60
from queens.distributions.free_variable import FreeVariable
from queens.drivers.function import Function
from queens.iterators.points import Points
from queens.main import run_iterator
from queens.models.simulation import Simulation
from queens.parameters.parameters import Parameters
from queens.schedulers.pool import Pool
from queens.utils.io import load_result


def test_points(inputs, expected_results, global_settings):
    """Integration test for the points iterator."""
    # Parameters
    x1 = FreeVariable(dimension=1)
    x2 = FreeVariable(dimension=1)
    parameters = Parameters(x1=x1, x2=x2)

    # Setup iterator
    driver = Function(parameters=parameters, function="rosenbrock60")
    scheduler = Pool(experiment_name=global_settings.experiment_name)
    model = Simulation(scheduler=scheduler, driver=driver)
    iterator = Points(
        points=inputs,
        result_description={"write_results": True},
        model=model,
        parameters=parameters,
        global_settings=global_settings,
    )

    # Actual analysis
    run_iterator(iterator, global_settings=global_settings)

    # Load results
    results = load_result(global_settings.result_file(".pickle"))

    np.testing.assert_array_equal(
        results["output"]["result"],
        expected_results,
    )


def test_points_failure(global_settings):
    """Test failure of the points iterator."""
    inputs = {"x1": [1], "x2": [1, 2]}
    # Parameters
    x1 = FreeVariable(dimension=1)
    x2 = FreeVariable(dimension=1)
    parameters = Parameters(x1=x1, x2=x2)

    # Setup iterator
    driver = Function(parameters=parameters, function="rosenbrock60")
    scheduler = Pool(experiment_name=global_settings.experiment_name)
    model = Simulation(scheduler=scheduler, driver=driver)
    iterator = Points(
        points=inputs,
        result_description={"write_results": True},
        model=model,
        parameters=parameters,
        global_settings=global_settings,
    )

    with pytest.raises(
        ValueError, match="Non-matching number of points for the different parameters: x1: 1, x2: 2"
    ):
        run_iterator(iterator, global_settings=global_settings)


@pytest.fixture(name="inputs")
def fixture_inputs():
    """Input points."""
    return {"x1": [1, 2], "x2": [3, 4]}


@pytest.fixture(name="expected_results")
def fixture_expected_results(inputs):
    """Expected results for given input points."""
    input_as_array = inputs.copy()
    input_as_array["x1"] = np.array(input_as_array["x1"]).reshape(-1, 1)
    input_as_array["x2"] = np.array(input_as_array["x2"]).reshape(-1, 1)
    return rosenbrock60(**input_as_array)
