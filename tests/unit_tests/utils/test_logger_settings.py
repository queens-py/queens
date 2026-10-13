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
"""Unit tests for the logger on the workers."""

import logging

import numpy as np
import pytest

from queens.data_processors import NumpyFile
from queens.distributions import FreeVariable
from queens.drivers import Function, Jobscript
from queens.parameters import Parameters
from queens.utils.exceptions import SubprocessError
from queens.utils.logger_settings import (
    get_worker_logger,
    reset_logger_on_worker,
    setup_logger_on_worker,
)


@pytest.fixture(name="parameters")
def fixture_parameters():
    """Parameters for the driver tests."""
    return Parameters(parameter_1=FreeVariable(1))


@pytest.fixture(name="input_template")
def fixture_input_template(tmp_path):
    """Input template for the jobscript driver."""
    input_template = tmp_path / "input_template.yaml"
    input_template.write_text("parameter_1: {{ parameter_1 }}")
    return input_template


def has_log_file(logger):
    """Check if a log file is attached to the logger."""
    return any(isinstance(handler, logging.FileHandler) for handler in logger.handlers)


def test_setup_logger_on_worker_one_file_per_job(tmp_path):
    """Test that each job gets its own log file."""
    logger = get_worker_logger("test")
    job_dirs = [tmp_path / "1", tmp_path / "2"]
    for job_dir in job_dirs:
        job_dir.mkdir()
        setup_logger_on_worker(log_dir=job_dir, level=logging.INFO)
        logger.info("message of job %s", job_dir.name)
    reset_logger_on_worker()

    assert not has_log_file(get_worker_logger())
    for job_dir in job_dirs:
        log = (job_dir / "worker.log").read_text()
        assert log.count("message of job") == 1
        assert f"message of job {job_dir.name}" in log


def test_setup_logger_on_worker_overwrites_log_file(tmp_path):
    """Test that a rerun of a job starts a new log file."""
    logger = get_worker_logger("test")
    for run in range(2):
        setup_logger_on_worker(log_dir=tmp_path, level=logging.INFO)
        logger.info("run %s", run)
    reset_logger_on_worker()

    assert (tmp_path / "worker.log").read_text().count("run") == 1


def test_file_level_is_kept_in_debug_mode(tmp_path):
    """Test that the file keeps its level in debug mode."""
    parent = get_worker_logger().parent
    parent.setLevel(logging.DEBUG)
    logger = get_worker_logger("test")
    setup_logger_on_worker(log_dir=tmp_path, level=logging.INFO)
    logger.debug("debug message")
    logger.info("info message")
    reset_logger_on_worker()

    log = (tmp_path / "worker.log").read_text()
    assert "info message" in log
    assert "debug message" not in log


def test_jobscript_driver_writes_worker_log(tmp_path, parameters, input_template):
    """Test that driver and data processor log to the job directory."""
    driver = Jobscript(
        parameters=parameters,
        input_templates=input_template,
        jobscript_template="echo dummy",
        executable="",
        data_processor=NumpyFile(
            file_name_identifier="missing.npy", files_to_be_deleted_regex_lst=["*.log"]
        ),
        worker_log_level="DEBUG",
    )

    driver(np.array([1.0]), 7, 1, tmp_path, "experiment")

    job_dir = tmp_path / "7"
    log = (job_dir / "worker.log").read_text()
    assert "missing.npy' does not exist!" in log
    assert "Got result: None" in log
    # the worker log is not in the directory the data processor searches and cleans up
    assert not list((job_dir / "output").iterdir())
    # the log file is released and the level is reset after the job
    assert not has_log_file(get_worker_logger())
    assert get_worker_logger().level == logging.NOTSET


def test_failed_job_is_logged(tmp_path, parameters, input_template):
    """Test that the error of a failed job is written to the worker log."""
    driver = Jobscript(
        parameters=parameters,
        input_templates=input_template,
        jobscript_template="exit 1",
        executable="",
    )

    with pytest.raises(SubprocessError):
        driver(np.array([1.0]), 7, 1, tmp_path, "experiment")

    log = (tmp_path / "7" / "worker.log").read_text()
    assert "Job 7 failed." in log
    assert "SubprocessError" in log
    assert not has_log_file(get_worker_logger())


@pytest.mark.parametrize("level", ["debug", "DEBUG", logging.DEBUG])
def test_worker_log_level(level, parameters):
    """Test that the worker log level can be a number or a name."""
    driver = Function(
        parameters=parameters, function=lambda parameter_1: 1.0, worker_log_level=level
    )

    assert driver.worker_log_level == logging.DEBUG


def test_invalid_worker_log_level(parameters):
    """Test that an invalid worker log level fails on initialization."""
    with pytest.raises(ValueError, match="Unknown logging level"):
        Function(parameters=parameters, function=lambda parameter_1: 1.0, worker_log_level="loud")


def test_worker_log_level_none_writes_no_file(tmp_path, parameters, input_template):
    """Test that worker_log_level=None switches the log file off."""
    driver = Jobscript(
        parameters=parameters,
        input_templates=input_template,
        jobscript_template="echo dummy",
        executable="",
        worker_log_level=None,
    )

    driver(np.array([1.0]), 7, 1, tmp_path, "experiment")

    assert driver.worker_log_level is None
    assert not (tmp_path / "7" / "worker.log").exists()


def test_function_driver_writes_no_worker_log_by_default(tmp_path):
    """Test that the function driver creates no job directory by default."""
    parameters = Parameters(x1=FreeVariable(1), x2=FreeVariable(1))
    driver = Function(parameters=parameters, function="rosenbrock60")

    driver(np.array([1.0, 1.0]), 1, 1, tmp_path, "experiment")

    assert not list(tmp_path.iterdir())


def test_data_processor_without_job_writes_no_file(tmp_path):
    """Test that a data processor outside of a job only reads."""
    data_processor = NumpyFile(file_name_identifier="missing.npy")

    assert data_processor.get_data_from_file(tmp_path) is None
    assert data_processor.get_data_from_file(tmp_path / "not_existing") is None
    assert not list(tmp_path.iterdir())
