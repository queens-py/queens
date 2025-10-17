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
"""Configuration of folder structure of QUEENS experiments."""

import logging
from pathlib import Path

from queens.utils.path import create_folder_if_not_existent

_logger = logging.getLogger(__name__)

BASE_DATA_DIR = "queens-experiments"


def base_directory() -> Path:
    """Holds all queens experiments.

    The base directory holds individual folders for each queens experiment on the compute machine.
    Per default, it is located and structured as follows::

        $HOME/queens-experiments
          ├── experiment_name_1
          ├── experiment_name_2

    For remote cluster test runs, a separate base directory structure is used::

        $HOME/queens-tests
          ├── pytest-0
          │   ├── test_name_1
          │   └── test_name_2
          ├── pytest-1
              ├── test_name_1
              └── test_name_2
    """
    base_dir = Path().home() / BASE_DATA_DIR
    create_directory(base_dir)
    return base_dir


def experiment_directory(
    experiment_name: str, experiment_base_directory: str | Path | None = None
) -> Path:
    """Directory for data of a specific experiment on the computing machine.

    If no experiment_base_directory is provided, base_directory() is used as default.

    Args:
        experiment_name: Experiment name
        experiment_base_directory: Base directory for the experiment directory

    Returns:
        Experiment directory
    """
    if experiment_base_directory is None:
        experiment_base_directory = base_directory()
    else:
        experiment_base_directory = Path(experiment_base_directory)

    experiment_dir = experiment_base_directory / experiment_name
    create_directory(experiment_dir)
    return experiment_dir


def create_directory(dir_path: str | Path) -> None:
    """Create a directory either local or remote.

    Args:
        dir_path: Directory to create
    """
    _logger.debug("Creating folder %s.", dir_path)
    create_folder_if_not_existent(dir_path)


def current_job_directory(experiment_dir: Path, job_id: int) -> Path:
    """Directory of the latest submitted job.

    Args:
        experiment_dir: Experiment directory
        job_id: Job ID of the current job

    Returns:
        Path to the current job directory.
    """
    job_dir = experiment_dir / str(job_id)
    return job_dir


def job_dirs_in_experiment_dir(experiment_dir: Path | str) -> list[Path]:
    """Get job directories in experiment_dir.

    Args:
        experiment_dir: Path with the job dirs

    Returns:
        List with job_dir paths
    """
    experiment_dir = Path(experiment_dir)
    job_directories = []
    for job_directory in experiment_dir.iterdir():
        if job_directory.is_dir() and job_directory.name.isdigit():
            job_directories.append(job_directory)

    # Sort the jobs directories
    return sorted(job_directories, key=lambda x: int(x.name))
