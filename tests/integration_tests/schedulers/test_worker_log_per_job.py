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
"""Integration test: each job writes its own worker log."""

import numpy as np
import pytest

from queens.data_processors import NumpyFile
from queens.distributions import FreeVariable
from queens.drivers import Jobscript
from queens.parameters import Parameters
from queens.schedulers import Local, Pool

NUM_JOBS = 8


@pytest.fixture(name="driver")
def fixture_driver(tmp_path):
    """Jobscript driver whose data processor looks for a missing file."""
    input_template = tmp_path / "input.yaml"
    input_template.write_text("x: {{ x }}")
    return Jobscript(
        parameters=Parameters(x=FreeVariable(1)),
        input_templates=input_template,
        jobscript_template="echo dummy",
        executable="",
        data_processor=NumpyFile(file_name_identifier="missing.npy"),
        worker_log_level="DEBUG",
    )


@pytest.mark.parametrize("scheduler_class", [Pool, Local])
def test_worker_log_per_job(scheduler_class, driver, tmp_path):
    """Test that each job logs into its own worker log file."""
    scheduler = scheduler_class(
        experiment_name="worker_log",
        num_jobs=4,
        experiment_base_dir=tmp_path,
        overwrite_existing_experiment=True,
    )
    scheduler.copy_files_to_experiment_dir(driver.files_to_copy)

    try:
        scheduler.evaluate(np.arange(NUM_JOBS, dtype=float).reshape(-1, 1), driver)
    finally:
        scheduler.cleanup()

    for job_id in range(NUM_JOBS):
        job_dir = scheduler.experiment_dir / str(job_id)
        log = (job_dir / "worker.log").read_text()
        assert log.count("does not exist!") == 1  # one message of the data processor
        assert log.count("Got result") == 1  # one message of the driver
        assert str(job_dir / "output" / "missing.npy") in log  # it refers to this job
