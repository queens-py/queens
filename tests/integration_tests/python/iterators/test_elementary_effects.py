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
"""Integration tests for the elementary effects iterator."""

import logging

import numpy as np
import pytest

from queens.distributions.uniform import Uniform
from queens.drivers.function import Function
from queens.iterators.elementary_effects import ElementaryEffects
from queens.main import run_iterator
from queens.models.simulation import Simulation
from queens.parameters.parameters import Parameters
from queens.schedulers.pool import Pool
from queens.utils.io import load_result

_logger = logging.getLogger(__name__)


def test_elementary_effects_ishigami90(global_settings):
    """Test case for the elementary effects iterator.

    This test is based on the Ishigami function.
    """
    # Parameters
    x1 = Uniform(lower_bound=-3.14159265359, upper_bound=3.14159265359)
    x2 = Uniform(lower_bound=-3.14159265359, upper_bound=3.14159265359)
    x3 = Uniform(lower_bound=-3.14159265359, upper_bound=3.14159265359)
    parameters = Parameters(x1=x1, x2=x2, x3=x3)

    # Setup iterator
    driver = Function(parameters=parameters, function="ishigami90")
    scheduler = Pool(experiment_name=global_settings.experiment_name)
    model = Simulation(scheduler=scheduler, driver=driver)
    iterator = ElementaryEffects(
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


def test_elementary_effects_sobol(
    expected_result_mu,
    expected_result_mu_star,
    expected_result_sigma,
    global_settings,
):
    """Test case for the elementary effects iterator.

    This test is based on Sobol's G function.
    """
    # Parameters
    x1 = Uniform(lower_bound=0, upper_bound=1)
    x2 = Uniform(lower_bound=0, upper_bound=1)
    x3 = Uniform(lower_bound=0, upper_bound=1)
    x4 = Uniform(lower_bound=0, upper_bound=1)
    x5 = Uniform(lower_bound=0, upper_bound=1)
    x6 = Uniform(lower_bound=0, upper_bound=1)
    x7 = Uniform(lower_bound=0, upper_bound=1)
    x8 = Uniform(lower_bound=0, upper_bound=1)
    x9 = Uniform(lower_bound=0, upper_bound=1)
    x10 = Uniform(lower_bound=0, upper_bound=1)
    parameters = Parameters(x1=x1, x2=x2, x3=x3, x4=x4, x5=x5, x6=x6, x7=x7, x8=x8, x9=x9, x10=x10)

    # Setup iterator
    driver = Function(parameters=parameters, function="sobol_g_function")
    scheduler = Pool(experiment_name=global_settings.experiment_name)
    model = Simulation(scheduler=scheduler, driver=driver)
    iterator = ElementaryEffects(
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

    np.testing.assert_allclose(results["sensitivity_indices"]["mu"], expected_result_mu)
    np.testing.assert_allclose(results["sensitivity_indices"]["mu_star"], expected_result_mu_star)
    np.testing.assert_allclose(results["sensitivity_indices"]["sigma"], expected_result_sigma)
