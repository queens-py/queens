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
"""QUEENS scheduler parent class."""

import abc
import logging
import select
import sys

import numpy as np

from queens.utils.config_directories import create_directory, experiment_directory
from queens.utils.rsync import rsync

_logger = logging.getLogger(__name__)


class Scheduler(metaclass=abc.ABCMeta):
    """Abstract base class for schedulers in QUEENS.

    Attributes:
        experiment_name (str): name of the current experiment
        experiment_dir (Path): Path to QUEENS experiment directory.
        num_jobs (int): Maximum number of parallel jobs
        next_job_id (int): Next job ID.
        verbose (bool): Verbosity of evaluations
    """

    def __init__(self, experiment_name, experiment_dir, num_jobs, verbose=True):
        """Initialize scheduler.

        Args:
            experiment_name (str): name of QUEENS experiment.
            experiment_dir (Path): Path to QUEENS experiment directory.
            num_jobs (int): Maximum number of parallel jobs
            verbose (bool, opt): Verbosity of evaluations. Defaults to True.
        """
        self.experiment_name = experiment_name
        self.experiment_dir = experiment_dir
        self.num_jobs = num_jobs
        self.next_job_id = 0
        self.verbose = verbose

    @abc.abstractmethod
    def evaluate(self, samples, driver, job_ids=None):
        """Submit jobs to driver.

        Args:
            samples (np.array): Array of samples
            driver (Driver): Driver object that runs simulation
            job_ids (lst, opt): List of job IDs corresponding to samples

        Returns:
            result_dict (dict): Dictionary containing results
        """

    def local_experiment_dir(
        self, experiment_name, experiment_base_dir, overwrite_existing_experiment
    ):
        """Get the local experiment directory.

        Args:
            experiment_name (str): name of the current experiment
            experiment_base_dir (str, Path): Base directory for the simulation outputs
            overwrite_existing_experiment (bool): If true, continue and overwrite experiment
                directory. If false, prompt user for confirmation before continuing and overwriting.

        Returns:
            experiment_dir (Path): Path to local experiment directory.
        """
        experiment_dir = experiment_directory(experiment_name, experiment_base_dir)
        if not overwrite_existing_experiment and experiment_dir.exists():
            self.get_user_confirmation_to_overwrite(experiment_dir)
        create_directory(experiment_dir)

        return experiment_dir

    def get_user_confirmation_to_overwrite(self, experiment_dir):
        """Prompt the user to confirm overwriting the experiment directory.

        Args:
            experiment_dir (Path): Directory where experiments are stored.

        Raises:
            FileExistsError: If an experiment with the same name already exists.
        """
        _logger.warning(
            "An experiment already exists in the experiment directory '%s'.\nIf you still want "
            "to continue and overwrite the existing experiment, please enter 'y' within 10 "
            "seconds. Otherwise, press enter or wait to abort the QUEENS run.",
            experiment_dir,
        )
        # Wait for user input for 10 seconds
        input_entered, _, _ = select.select([sys.stdin], [], [], 10)
        if input_entered:
            user_input = sys.stdin.readline().strip()
            if user_input != "y":
                _logger.info("Aborting QUEENS run.")
                sys.exit(2)
            else:
                _logger.info("Overwriting existing experiment and continuing QUEENS run.")
        else:
            _logger.info("No input received. Aborting QUEENS run.")
            sys.exit(2)

    def copy_files_to_experiment_dir(self, paths):
        """Copy file to experiment directory.

        Args:
            paths (str, Path, list): paths to files or directories that should be copied to
                                     experiment directory
        """
        destination = f"{self.experiment_dir}/"
        rsync(paths, destination)

    def get_job_ids(self, num_samples):
        """Get job ids and update next_job_id.

        Args:
            num_samples (int): Number of samples

        Returns:
            job_ids (np.array): Array of job ids
        """
        job_ids = self.next_job_id + np.arange(num_samples)
        self.next_job_id += num_samples
        return job_ids
