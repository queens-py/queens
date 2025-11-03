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
"""Test 4C run."""

import logging

import numpy as np

from queens.data_processors.pvd_file import PvdFile
from queens.distributions.uniform import Uniform
from queens.iterators.monte_carlo import MonteCarlo
from queens.main import run_iterator
from queens.models.simulation import Simulation
from queens.parameters.parameters import Parameters
from queens.schedulers.local import Local
from queens.utils.config_directories import experiment_directory
from queens.utils.io import load_result, read_file
from queens_interfaces.fourc.driver import Fourc

_logger = logging.getLogger(__name__)


def test_fourc_mc(
    third_party_inputs,
    fourc_link_paths,
    fourc_example_expected_output,
    global_settings,
):
    """Test simple 4C run."""
    # generate json input file from template
    fourc_input_file_template = third_party_inputs / "fourc" / "solid_runtime_hex8.4C.yaml"
    fourc_executable, _, _ = fourc_link_paths

    # Parameters
    parameter_1 = Uniform(lower_bound=0.0, upper_bound=1.0)
    parameter_2 = Uniform(lower_bound=0.0, upper_bound=1.0)
    parameters = Parameters(parameter_1=parameter_1, parameter_2=parameter_2)

    data_processor = PvdFile(
        field_name="displacement",
        file_name_identifier="output-structure.pvd",
        file_options_dict={},
    )

    scheduler = Local(
        experiment_name=global_settings.experiment_name,
        num_procs=2,
        num_jobs=2,
    )
    driver = Fourc(
        parameters=parameters,
        input_templates=fourc_input_file_template,
        executable=fourc_executable,
        data_processor=data_processor,
    )
    model = Simulation(scheduler=scheduler, driver=driver)
    iterator = MonteCarlo(
        seed=42,
        num_samples=2,
        result_description={"write_results": True, "plot_results": False},
        model=model,
        parameters=parameters,
        global_settings=global_settings,
    )

    try:
        # Actual analysis
        run_iterator(iterator, global_settings=global_settings)

        # Load results
        results = load_result(global_settings.result_file(".pickle"))

        # assert statements
        np.testing.assert_array_almost_equal(
            results["raw_output_data"]["result"], fourc_example_expected_output, decimal=6
        )
    except Exception as error:
        experiment_dir, _ = experiment_directory(global_settings.experiment_name)
        job_dir = experiment_dir / "0"
        _logger.info(list(job_dir.iterdir()))
        output_dir = job_dir / "output"
        _logger.info(list(output_dir.iterdir()))
        _logger.info(read_file(output_dir / "output.log"))
        raise error
