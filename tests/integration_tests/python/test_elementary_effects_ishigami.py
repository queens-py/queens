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
"""Integration test for the elementary effects iterator.

This test is based on the Ishigami function.
"""

import logging

import pytest

from queens.distributions.uniform import Uniform
from queens.drivers.function_driver import FunctionDriver
from queens.iterators.elementary_effects_iterator import ElementaryEffectsIterator
from queens.main import run_iterator
from queens.models.simulation_model import SimulationModel
from queens.parameters.parameters import Parameters
from queens.schedulers.pool_scheduler import PoolScheduler
from queens.utils.io_utils import load_result

_logger = logging.getLogger(__name__)


def test_elementary_effects_ishigami(global_settings):
    """Test case for elementary effects iterator."""
    # Parameters
    x1 = Uniform(lower_bound=-3.14159265359, upper_bound=3.14159265359)
    x2 = Uniform(lower_bound=-3.14159265359, upper_bound=3.14159265359)
    x3 = Uniform(lower_bound=-3.14159265359, upper_bound=3.14159265359)
    parameters = Parameters(x1=x1, x2=x2, x3=x3)

    # Setup iterator
    driver = FunctionDriver(parameters=parameters, function="ishigami90")
    scheduler = PoolScheduler(experiment_name=global_settings.experiment_name)
    model = SimulationModel(scheduler=scheduler, driver=driver)
    iterator = ElementaryEffectsIterator(
        seed=2,
        num_trajectories=100,
        num_optimal_trajectories=4,
        number_of_levels=10,
        confidence_level=0.95,
        local_optimization=False,
        num_bootstrap_samples=1000,
        result_description={
            "write_results": True,
            "plotting_options": {
                "plot_booleans": [False, False],
                "plotting_dir": "dummy",
                "plot_names": ["bars", "scatter"],
                "save_bool": [False, False],
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
    _logger.info(results)

    assert results["sensitivity_indices"]["mu"][0] == pytest.approx(15.46038594, abs=1e-7)
    assert results["sensitivity_indices"]["mu"][1] == pytest.approx(0.0, abs=1e-7)
    assert results["sensitivity_indices"]["mu"][2] == pytest.approx(0.0, abs=1e-7)

    assert results["sensitivity_indices"]["mu_star"][0] == pytest.approx(15.460385940, abs=1e-7)
    assert results["sensitivity_indices"]["mu_star"][1] == pytest.approx(1.47392000, abs=1e-7)
    assert results["sensitivity_indices"]["mu_star"][2] == pytest.approx(5.63434321, abs=1e-7)

    assert results["sensitivity_indices"]["sigma"][0] == pytest.approx(15.85512257, abs=1e-7)
    assert results["sensitivity_indices"]["sigma"][1] == pytest.approx(1.70193622, abs=1e-7)
    assert results["sensitivity_indices"]["sigma"][2] == pytest.approx(9.20084394, abs=1e-7)

    assert results["sensitivity_indices"]["mu_star_conf"][0] == pytest.approx(13.53414548, abs=1e-7)
    assert results["sensitivity_indices"]["mu_star_conf"][1] == pytest.approx(0.0, abs=1e-7)
    assert results["sensitivity_indices"]["mu_star_conf"][2] == pytest.approx(5.51108773, abs=1e-7)
