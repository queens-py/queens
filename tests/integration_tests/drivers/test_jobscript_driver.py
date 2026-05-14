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
"""Integration test for the jobscript driver."""

import numpy as np
import pytest

from queens.distributions import Uniform
from queens.drivers import Jobscript
from queens.iterators import MonteCarlo
from queens.main import run_iterator
from queens.models import Simulation
from queens.parameters import Parameters
from queens.schedulers import Pool
from queens.utils.io import load_result


@pytest.mark.parametrize("reuse_existing_jobs", [True, False])
def test_resusing_existing_results(
    tmp_path,
    global_settings,
    current_time_jobscript_template,
    time_data_processor,
    reuse_existing_jobs,
):
    """Test that jobscript results are reused or overwritten in a second QUEENS run.

    The results will be different if the jobscript is executed again because they capture the
    current time at execution (see current_time_jobscript_template and time_data_processor).
    """
    input_template = tmp_path / "input_template.txt"
    input_template.write_text("{{ x1 }}")
    num_samples = 3

    # First iterator run
    with global_settings as gs:
        x1 = Uniform(lower_bound=0.0, upper_bound=1.0)
        parameters = Parameters(x1=x1)
        scheduler = Pool(
            experiment_name=gs.experiment_name,
            num_jobs=1,
        )
        driver = Jobscript(
            parameters=parameters,
            input_templates=input_template,
            jobscript_template=current_time_jobscript_template,
            executable="",
            data_processor=time_data_processor,
        )
        model = Simulation(scheduler=scheduler, driver=driver)
        iterator = MonteCarlo(
            model=model,
            parameters=parameters,
            global_settings=gs,
            seed=123,
            num_samples=num_samples,
            result_description={"write_results": True, "plot_results": False},
        )

        run_iterator(iterator, global_settings=gs)
        first_results = load_result(gs.result_file(".pickle"))

    # Second iterator run with the same experiment name and inputs
    with global_settings as gs:
        x1 = Uniform(lower_bound=0.0, upper_bound=1.0)
        parameters = Parameters(x1=x1)
        scheduler = Pool(
            experiment_name=gs.experiment_name,
            num_jobs=1,
            overwrite_existing_experiment=True,
        )
        driver = Jobscript(
            parameters=parameters,
            input_templates=input_template,
            jobscript_template=current_time_jobscript_template,
            executable="",
            data_processor=time_data_processor,
            reuse_existing_jobs=reuse_existing_jobs,
        )
        model = Simulation(scheduler=scheduler, driver=driver)
        iterator = MonteCarlo(
            model=model,
            parameters=parameters,
            global_settings=gs,
            seed=123,
            num_samples=num_samples,
            result_description={"write_results": True, "plot_results": False},
        )

        run_iterator(iterator, global_settings=gs)
        second_results = load_result(gs.result_file(".pickle"))

    np.testing.assert_array_equal(first_results["input_data"], second_results["input_data"])

    if reuse_existing_jobs:
        np.testing.assert_array_equal(
            first_results["raw_output_data"]["result"],
            second_results["raw_output_data"]["result"],
        )
    else:
        assert not np.array_equal(
            first_results["raw_output_data"]["result"],
            second_results["raw_output_data"]["result"],
        )
