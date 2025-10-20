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
"""Data processor class for numpy data extraction."""

import logging

import numpy as np

from queens.data_processors._data_processor import DataProcessor
from queens.utils.logger_settings import log_init_args

_logger = logging.getLogger(__name__)


class NumpyFile(DataProcessor):
    """Class for extracting data from numpy binaries."""

    @log_init_args
    def __init__(
        self,
        file_name_identifier=None,
        file_options_dict=None,
        files_to_be_deleted_regex_lst=None,
    ):
        """Instantiate data processor class for numpy binary data.

        Args:
            file_name_identifier (str): Identifier of file name.
                                        The file prefix can contain regex expression
                                        and subdirectories.
            file_options_dict (dict): Dictionary with read-in options for the file
            files_to_be_deleted_regex_lst (lst): List with paths to files that should be deleted.
                                                 The paths can contain regex expressions.
        """
        super().__init__(
            file_name_identifier=file_name_identifier,
            file_options_dict=file_options_dict,
            files_to_be_deleted_regex_lst=files_to_be_deleted_regex_lst,
        )

    def get_raw_data_from_file(self, file_path):
        """Get the raw data from the files of interest.

        This method loads the numpy binary data from the file.

        Args:
            file_path (str): Actual path to the file of interest.

        Returns:
            raw_data (np.array): Raw data from file.
        """
        try:
            raw_data = np.load(file_path)
            _logger.info("Successfully read-in data from %s.", file_path)
            return raw_data
        except FileNotFoundError as error:
            _logger.warning(
                "Could not find the file: %s. The following FileNotFoundError was raised: %s. "
                "Skipping the file and continuing.",
                file_path,
                error,
            )
        except ValueError as error:
            _logger.warning(
                "Could not read the file: %s. The following ValueError was raised: %s. "
                "Skipping the file and continuing.",
                file_path,
                error,
            )
        return None
