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
"""Integration test for Sobol indices estimation for Ishigami function."""

import numpy as np

from queens.distributions.uniform import Uniform
from queens.drivers.function import Function
from queens.iterators.sobol_index_iterator import SobolIndexIterator
from queens.main import run_iterator
from queens.models.simulation_model import SimulationModel
from queens.parameters.parameters import Parameters
from queens.schedulers.pool_scheduler import PoolScheduler
from queens.utils.io_utils import load_result
from test_utils.integration_tests import assert_sobol_index_iterator_results


def test_sobol_indices_ishigami(global_settings):
    """Test case for Salib based Saltelli iterator."""
    # Parameters
    x1 = Uniform(lower_bound=-3.14159265359, upper_bound=3.14159265359)
    x2 = Uniform(lower_bound=-3.14159265359, upper_bound=3.14159265359)
    x3 = Uniform(lower_bound=-3.14159265359, upper_bound=3.14159265359)
    parameters = Parameters(x1=x1, x2=x2, x3=x3)

    # Setup iterator
    driver = Function(parameters=parameters, function="ishigami90")
    scheduler = PoolScheduler(experiment_name=global_settings.experiment_name, verbose=True)
    model = SimulationModel(scheduler=scheduler, driver=driver)
    iterator = SobolIndexIterator(
        seed=42,
        calc_second_order=True,
        num_samples=16,
        confidence_level=0.95,
        num_bootstrap_samples=1000,
        result_description={"write_results": True, "plot_results": False},
        model=model,
        parameters=parameters,
        global_settings=global_settings,
        skip_values=1024,
    )

    # Actual analysis
    run_iterator(iterator, global_settings=global_settings)

    # Load results
    results = load_result(global_settings.result_file(".pickle"))

    expected_result = {}

    expected_result["S1"] = np.array([0.12572757495660558, 0.3888444532476749, -0.1701023677236496])

    expected_result["S1_conf"] = np.array(
        [0.3935803586836114, 0.6623091120357786, 0.2372589075839736]
    )

    expected_result["ST"] = np.array([0.32520201992825987, 0.5263552164769918, 0.1289289258091274])

    expected_result["ST_conf"] = np.array(
        [0.24575185898081872, 0.5535870474744364, 0.15792828597131078]
    )

    expected_result["S2"] = np.array(
        [
            [np.nan, 0.6350854922111611, 1.0749774123116016],
            [np.nan, np.nan, 0.32907368546743065],
            [np.nan, np.nan, np.nan],
        ]
    )

    expected_result["S2_conf"] = np.array(
        [
            [np.nan, 0.840605849268133, 1.2064077218919202],
            [np.nan, np.nan, 0.5803799668636836],
            [np.nan, np.nan, np.nan],
        ]
    )

    assert_sobol_index_iterator_results(results, expected_result)
