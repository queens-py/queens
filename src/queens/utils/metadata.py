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
"""Metadata objects."""

from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Iterator

import pandas as pd
import yaml
from pandas.io.json._normalize import nested_to_record

from queens.utils.config_directories import job_dirs_in_experiment_dir
from queens.utils.io import to_dict_with_standard_types
from queens.utils.printing import get_str_table

METADATA_FILENAME = "metadata"
METADATA_FILETYPE = ".yaml"


class SimulationMetadata:
    """Simulation metadata object.

    This objects holds metadata, times code sections and exports them to yaml.

    Attributes:
        job_id: Id of the job
        inputs: Parameters for this job
        file_path (pathlib.Path): Path to export the metadata
        timestamp (str): Timestamp of the object creation
        outputs (tuple): Results obtain by the simulation
        times (dict): Wall times of code sections
    """

    def __init__(self, job_id: int, inputs: dict, job_dir: Path) -> None:
        """Init simulation metadata object.

        Args:
            job_id: Id of the job
            inputs: Parameters for this job
            job_dir: Directory in which to write the metadata
        """
        self.job_id = job_id
        self.timestamp: str | None = None
        self.inputs = inputs
        self.file_path = (Path(job_dir) / METADATA_FILENAME).with_suffix(METADATA_FILETYPE)
        self.outputs = None
        self.times: dict = {}
        self._create_timestamp()

    def _create_timestamp(self) -> None:
        """Create timestamp in a nice format."""
        self.timestamp = datetime.now().strftime("%d-%m-%Y, %H:%M:%S")

    def to_dict(self) -> dict[str, Any]:
        """Create dictionary from object.

        Returns:
            Dictionary of the metadata object
        """
        dictionary = self.__dict__.copy()
        dictionary.pop("file_path")
        return dictionary

    def export(self) -> None:
        """Export the object to human readable format."""
        yaml_string = yaml.safe_dump(
            to_dict_with_standard_types(self.to_dict()), sort_keys=False, default_flow_style=False
        )
        self.file_path.write_text(yaml_string, encoding="utf-8")

    @contextmanager
    def time_code(self, code_section_name: str) -> Iterator[None]:
        """Timer some code section.

        This method allows us to time not only the runtime of the simulation, but also subparts.

        Args:
            code_section_name: Name for this code section
        """
        # Start timer
        start = perf_counter()
        self.times[code_section_name] = {"status": "running"}

        # Export metadata
        self.export()
        try:
            # Call the code within the context
            yield

            # If we are here the job was successful
            self.times[code_section_name]["status"] = "successful"

        # Something goes wrong
        except Exception as exception:
            # Set the status to failed
            self.times[code_section_name]["status"] = "failed"

            # Raise the original exception
            raise exception
        # Call this no matter if job was successful or failed
        finally:
            run_time = perf_counter() - start
            # Add the runtime of this code section
            self.times[code_section_name]["time"] = run_time
            # Export since the job is either finished or failed
            self.export()

    def __str__(self) -> str:
        """Create function string.

        Returns:
            Table of the metdata object
        """
        return get_str_table("Simulation Metadata", self.to_dict())


def get_metadata_from_experiment_dir(experiment_dir: Path | str) -> Iterator[Any]:
    """Get metadata from experiment_dir.

    To keep memory usage limited, this is implemented as a generator.

    Args:
        experiment_dir: Path with the job dirs

    Yields:
        Metadata of a job
    """
    for job_dir in job_dirs_in_experiment_dir(experiment_dir):
        metadata_path = (job_dir / METADATA_FILENAME).with_suffix(METADATA_FILETYPE)
        yield yaml.safe_load(metadata_path.read_text())


def write_metadata_to_csv(experiment_dir: Path | str, csv_path: Path | None = None) -> None:
    """Gather and write job metadata to csv.

    Args:
        experiment_dir: Path with the job dirs
        csv_path: Path to export the csv file
    """
    experiment_dir = Path(experiment_dir)
    if csv_path is None:
        csv_path = experiment_dir / "metadata_gathered.csv"

    data = []
    for job_metadata in get_metadata_from_experiment_dir(experiment_dir):
        # For proper csv the dictionary can not be nested
        job_metadata = nested_to_record(job_metadata)
        data.append(job_metadata)
    df = pd.DataFrame.from_dict(data)
    df.to_csv(csv_path)
