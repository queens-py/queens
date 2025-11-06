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
        field_name,
        file_name_identifier=None,
        file_options_dict=None,
        files_to_be_deleted_regex_lst=None,
        time_steps=None,
        block=0,
        point_data=True,
        worker_log_level=logging.INFO,
        write_worker_log_files=True,
    ):
        """Instantiate data processor class for pvd data extraction.

        Args:
            field_name (str): Name of the field to extract data from
            file_name_identifier (str): Identifier of file name.
                                        The file prefix can contain regex expression
                                        and subdirectories.
            file_options_dict (dict): Dictionary with read-in options for the file
            files_to_be_deleted_regex_lst (lst): List with paths to files that should be deleted.
                                                 The paths can contain regex expressions.
            time_steps (lst, optional): Considered time steps (last time step by default)
            block (int, optional): Considered block of MultiBlock data set (first block by default)
            point_data (bool, optional): Whether to extract point data (True) or cell data (False).
                                         Defaults to point data.
            worker_log_level (int | str): Logging level used on the worker (default: "INFO")
            write_worker_log_files (bool): Control writing of worker logs to files (one per job)
                                           (default: True)
        """
        # pylint: disable=duplicate-code
        super().__init__(
            file_name_identifier=file_name_identifier,
            file_options_dict=file_options_dict,
            files_to_be_deleted_regex_lst=files_to_be_deleted_regex_lst,
            worker_log_level=worker_log_level,
            write_worker_log_files=write_worker_log_files,
        )
        self.field_name = field_name
        if time_steps is None:
            time_steps = [-1]
        self.time_steps = time_steps
        self.block = block
        self.data_attribute = "point_data"
        if not point_data:
            self.data_attribute = "cell_data"

    def get_raw_data_from_file(self, file_path):
        """Get the raw data from the files of interest.

        Args:
            file_path (str): Actual path to the file of interest.

        Returns:
            raw_data (pv.PVDReader): PVDReader object.
        """
        raw_data = pv.get_reader(file_path)
        return raw_data

    def filter_and_manipulate_raw_data(self, raw_data):
        """Filter and manipulate the raw data.

        Args:
            raw_data (pv.PVDReader): PVDReader object.

        Returns:
            processed_data (np.array): Cleaned, filtered or manipulated *data_processor* data.
        """
        processed_data = []
        for time_step in self.time_steps:
            raw_data.set_active_time_value(raw_data.time_values[time_step])
            processed_data.append(
                getattr(raw_data.read()[self.block], self.data_attribute)[self.field_name]
            )
        processed_data = np.vstack(processed_data)

        return processed_data
