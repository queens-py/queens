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
"""Test 4C with RF materials."""

import logging

import numpy as np
import pytest

from queens.iterators.monte_carlo import MonteCarlo
from queens.main import run_iterator
from queens.models.simulation import Simulation
from queens.parameters.parameters import Parameters
from queens.parameters.random_fields.karhunen_loeve import KarhunenLoeve
from queens.schedulers.local import Local
from queens.utils.config_directories import experiment_directory
from queens.utils.io import load_result, read_file
from queens_interfaces.fourc.driver import Fourc
from queens_interfaces.fourc.random_material_preprocessor import (
    create_random_elemenentwise_material_field,
    extract_by_material_id,
)

_logger = logging.getLogger(__name__)


class DummyKLField(KarhunenLoeve):
    """Dummy Karhunen-Loeve random field."""

    def expanded_representation(self, samples):
        """Dummy method for expansion."""
        return self.mean + self.std**2 * np.linalg.norm(self.coords["coords"], axis=1) * samples[0]


def test_write_random_elementwise_material(
    tmp_path,
    third_party_inputs,
    fourc_link_paths,
    expected_mean,
    global_settings,
):
    """Test 4C with random field for material parameters."""
    fourc_input_template = third_party_inputs / "fourc" / "coarse_plate_dirichlet_template.4C.yaml"

    material_file_template = tmp_path / "material.json"

    fourc_executable, post_ensight, _ = fourc_link_paths

    mue_rf_parameters = create_random_elemenentwise_material_field(
        fourc_input_template,
        "STRUCTURE ELEMENTS",
        extract_by_material_id(10),
        "mue",
        material_file_template,
    )

    # Parameters
    mue = DummyKLField(
        corr_length=5.0,
        std=0.03,
        mean=0.25,
        explained_variance=0.95,
        coords=mue_rf_parameters,
    )
    parameters = Parameters(mue=mue)

    # Setup iterator
    data_processor = None

    scheduler = Local(
        num_procs=1,
        num_jobs=1,
        experiment_name=global_settings.experiment_name,
    )
    driver = Fourc(
        parameters=parameters,
        input_templates={
            "input_file": fourc_input_template,
            "material_file": material_file_template,
        },
        executable=fourc_executable,
        post_processor=post_ensight,
        data_processor=data_processor,
    )
    model = Simulation(scheduler=scheduler, driver=driver)
    iterator = MonteCarlo(
        seed=1,
        num_samples=1,
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

        # Check if we got the expected results
        np.testing.assert_array_almost_equal(results["mean"], expected_mean, decimal=8)
    except Exception as error:
        experiment_dir = experiment_directory(global_settings.experiment_name)
        job_dir = experiment_dir / "0"
        _logger.info(list(job_dir.iterdir()))
        output_dir = job_dir / "output"
        _logger.info(list(output_dir.iterdir()))
        _logger.info(read_file(output_dir / "test_write_random_material_to_dat_0.log"))
        raise error


@pytest.fixture(name="expected_mean")
def fixture_expected_mean():
    """Reference samples mean."""
    result = np.array([None])

    return result
