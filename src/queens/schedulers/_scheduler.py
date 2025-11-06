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
from collections.abc import Iterable
from pathlib import Path
from typing import Protocol

import numpy as np

from queens.utils.rsync import rsync


class SchedulerCallableSignature(Protocol):
    """Signature for callables which can be used with QUEENS schedulers."""

    def __call__(
        self,
        sample: np.ndarray,
        job_id: int,
        num_procs: int,
        experiment_dir: Path,
        experiment_name: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Signature for callables which can be used with QUEENS schedulers.

        Args:
            sample (np.array): Input ample.
            job_id (int): Job ID.
            num_procs (int): Number of processors.
            experiment_dir (Path): Path to QUEENS experiment directory.
            experiment_name (str): Name of QUEENS experiment.

        Returns:
            Result and potentially the gradient.
        """


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
    def evaluate(
        self, samples: Iterable, function: SchedulerCallableSignature, job_ids: Iterable = None
    ) -> dict:
        """Submit jobs to driver.

        Args:
            samples (np.array): Array of samples
            function (Callable): Callable to evaluate in the scheduler
            job_ids (lst, opt): List of job IDs corresponding to samples

        Returns:
            result_dict (dict): Dictionary containing results
        """

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
