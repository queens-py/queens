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
"""Integration tests for the Optimization iterator."""

import numpy as np
import pytest

from queens.distributions.free_variable import FreeVariable
from queens.drivers.function import Function
from queens.iterators.optimization import Optimization
from queens.main import run_iterator
from queens.models.simulation import Simulation
from queens.parameters.parameters import Parameters
from queens.schedulers.pool import Pool
from queens.utils.io import load_result


@pytest.mark.parametrize(
    "algorithm",
    ["NELDER-MEAD", "POWELL", "CG", "BFGS", "L-BFGS-B", "TNC", "SLSQP"],
)
def test_optimization_rosenbrock(algorithm, global_settings):
    """Test different solution algorithms in optimization iterator."""
    # Parameters
    parameters = Parameters("x1", "x2")

    # Setup iterator
    driver = Function(parameters=parameters, function="rosenbrock60")
    scheduler = Pool(experiment_name=global_settings.experiment_name)
    model = Simulation(scheduler=scheduler, driver=driver)
    iterator = Optimization(
        algorithm=algorithm,
        initial_guess=[-3.0, -4.0],
        result_description={"write_results": True},
        bounds=[float("-inf"), float("inf")],
        model=model,
        parameters=parameters,
        global_settings=global_settings,
    )

    # Actual analysis
    run_iterator(iterator, global_settings=global_settings)

    # Load results
    results = load_result(global_settings.result_file(".pickle"))

    np.testing.assert_allclose(results.x, np.array([+1.0, +1.0]), rtol=1.0e-3)
    np.testing.assert_allclose(results.fun, np.array(+0.0), atol=5.0e-07)


@pytest.mark.parametrize(
    "algorithm",
    ["COBYLA", "SLSQP"],
)
def test_optimization_paraboloid_constrained(algorithm, global_settings):
    """Test different solution algorithms in optimization iterator.

    COBYLA: constrained but unbounded

    SLSQP:  constrained and bounded
    """
    # Parameters
    x1 = FreeVariable(dimension=1)
    x2 = FreeVariable(dimension=1)
    parameters = Parameters(x1=x1, x2=x2)

    # Setup iterator
    driver = Function(parameters=parameters, function="paraboloid")
    scheduler = Pool(experiment_name=global_settings.experiment_name)
    model = Simulation(scheduler=scheduler, driver=driver)
    iterator = Optimization(
        initial_guess=[2.0, 0.0],
        algorithm=algorithm,
        result_description={"write_results": True, "plot_results": True},
        bounds=[[0.0, 0.0], float("inf")],
        constraints={
            "cons1": {"type": "ineq", "fun": "lambda x:  x[0] - 2 * x[1] + 2"},
            "cons2": {"type": "ineq", "fun": "lambda x: -x[0] - 2 * x[1] + 6"},
            "cons3": {"type": "ineq", "fun": "lambda x: -x[0] + 2 * x[1] + 2"},
        },
        model=model,
        parameters=parameters,
        global_settings=global_settings,
    )

    # Actual analysis
    run_iterator(iterator, global_settings=global_settings)

    # Load results
    results = load_result(global_settings.result_file(".pickle"))

    np.testing.assert_allclose(results.x, np.array([+1.4, +1.7]), rtol=1.0e-4)
    np.testing.assert_allclose(results.fun, np.array(+0.8), atol=1.0e-07)
