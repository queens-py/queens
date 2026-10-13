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
"""QUEENS driver module base class."""

import abc
import logging
from pathlib import Path

import numpy as np

from queens.utils.config_directories import create_directory, current_job_directory
from queens.utils.logger_settings import (
    get_logging_level,
    get_worker_logger,
    reset_logger_on_worker,
    setup_logger_on_worker,
)


class Driver(metaclass=abc.ABCMeta):
    """Abstract base class for drivers in QUEENS.

    Attributes:
        parameters (Parameters): Parameters object
        files_to_copy (list): files or directories to copy to experiment_dir
        worker_log_level (int | None): Logging level of the job log files, None switches them off
        logger_on_worker (logging.Logger): Logger instance used on the worker
    """

    def __init__(
        self,
        parameters,
        files_to_copy=None,
        worker_log_level=logging.INFO,
    ):
        """Initialize Driver object.

        Args:
            parameters (Parameters): Parameters object
            files_to_copy (list): files or directories to copy to experiment_dir
            worker_log_level (int | str | None): Logging level of the log file written for each
                                                 job (default: logging.INFO). None switches the
                                                 log files off.
        """
        self.parameters = parameters
        if files_to_copy is None:
            files_to_copy = []
        if not isinstance(files_to_copy, list):
            raise TypeError("files_to_copy must be a list")
        for file_to_copy in files_to_copy:
            if not isinstance(file_to_copy, (str, Path)):
                raise TypeError("files_to_copy must be a list of strings or Path objects")
        self.files_to_copy = files_to_copy

        self.worker_log_level = (
            None if worker_log_level is None else get_logging_level(worker_log_level)
        )
        self.logger_on_worker = get_worker_logger(type(self).__name__)

    @abc.abstractmethod
    def run(
        self,
        sample: np.ndarray,
        job_id: int,
        num_procs: int,
        experiment_dir: Path,
        experiment_name: str,
    ) -> dict:
        """Abstract method for driver run.

        The log file of the job is set up by __call__, i.e., when a scheduler runs the driver.

        Args:
            sample (np.ndarray): Input sample
            job_id (int): Job ID
            num_procs (int): number of processors
            experiment_dir (Path): Path to QUEENS experiment directory.
            experiment_name (str): name of QUEENS experiment.

        Returns:
            Results
        """

    def __call__(
        self,
        sample: np.ndarray,
        job_id: int,
        num_procs: int,
        experiment_dir: Path,
        experiment_name: str,
    ) -> dict:
        """Run the driver with the log file of the job.

        Args:
            sample (np.ndarray): Input sample
            job_id (int): Job ID
            num_procs (int): number of processors
            experiment_dir (Path): Path to QUEENS experiment directory.
            experiment_name (str): name of QUEENS experiment.

        Returns:
            Results
        """
        worker_log_dir = None
        if self.worker_log_level is not None:
            worker_log_dir = current_job_directory(experiment_dir, job_id)
            create_directory(worker_log_dir)
        setup_logger_on_worker(log_dir=worker_log_dir, level=self.worker_log_level)

        try:
            return self.run(sample, job_id, num_procs, experiment_dir, experiment_name)
        except Exception:
            self.logger_on_worker.exception("Job %s failed.", job_id)
            raise
        finally:
            reset_logger_on_worker()
