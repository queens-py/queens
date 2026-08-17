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
"""Data processor class for pvd data extraction."""

import logging
from pathlib import Path

import numpy as np
import pyvista as pv

from queens.data_processors._data_processor import DataProcessor
from queens.utils.logger_settings import log_init_args

_logger = logging.getLogger(__name__)


class PvdFile(DataProcessor):
    """Class for extracting data from pvd.

    Attributes:
        field_name (str): Name of the field to extract data from
        time_steps (lst): Considered time steps (last time step by default)
        block (int): Considered block of MultiBlock data set (first block by default)
        data_attribute (str): 'point_data' or 'cell_data'
    """

    @log_init_args
    def __init__(
        self,
        field_name: str,
        file_name_identifier: str,
        file_options_dict: dict | None = None,
        files_to_be_deleted_regex_lst: list[str] | None = None,
        time_steps: list[int] | None = None,
        block: int = 0,
        point_data: bool = True,
    ) -> None:
        """Instantiate data processor class for pvd data extraction.

        Args:
            field_name: Name of the field to extract data from
            file_name_identifier: Identifier of file name. The file prefix can contain regex
                expression and subdirectories.
            file_options_dict: Dictionary with read-in options for the file
            files_to_be_deleted_regex_lst: List with paths to files that should be deleted. The
                paths can contain regex expressions.
            time_steps: Considered time steps (last time step by default)
            block: Considered block of MultiBlock data set (first block by default)
            point_data: Whether to extract point data (True) or cell data (False). Defaults to
                point data.
        """
        super().__init__(
            file_name_identifier=file_name_identifier,
            file_options_dict=file_options_dict,
            files_to_be_deleted_regex_lst=files_to_be_deleted_regex_lst,
        )
        self.field_name = field_name
        if time_steps is None:
            time_steps = [-1]
        self.time_steps = time_steps
        self.block = block
        self.data_attribute = "point_data"
        if not point_data:
            self.data_attribute = "cell_data"

    def get_raw_data_from_file(self, file_path: str | Path) -> pv.PVDReader:
        """Get the raw data from the files of interest.

        Args:
            file_path: Actual path to the file of interest.

        Returns:
            PVDReader object.
        """
        raw_data_reader = pv.get_reader(file_path)
        return raw_data_reader

    def filter_and_manipulate_raw_data(self, raw_data: pv.PVDReader) -> np.ndarray:
        """Filter and manipulate the raw data.

        Args:
            raw_data: PVDReader object.

        Returns:
            Cleaned, filtered or manipulated *data_processor* data.
        """
        field_data = []
        for time_step in self.time_steps:
            raw_data.set_active_time_value(raw_data.time_values[time_step])
            field_data.append(
                getattr(raw_data.read()[self.block], self.data_attribute)[self.field_name]
            )
        processed_data = np.vstack(field_data)

        return processed_data
