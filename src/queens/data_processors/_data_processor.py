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
"""Module for data processing of simulation results."""

import abc
import logging
from pathlib import Path

from queens.utils.logger_settings import setup_logger_on_worker

_logger = logging.getLogger(__name__)


class DataProcessor(metaclass=abc.ABCMeta):
    """Base class for data processing.

    Attributes:
        files_to_be_deleted_regex_lst (lst): List with paths to files that should be deleted.
                                             The paths can contain regex expressions. The
                                             paths are relative to the particular simulation output
                                             folder: *experiment_dir/<job_id>/output/<here
                                             comes your regex>* .
        file_options_dict (dict): Dictionary with read-in options for
                                  the file.
        file_name_identifier (str): Identifier for files.
                                    The file prefix can contain BASIC regex expression
                                    and subdirectories. Examples are wildcards `*` or
                                    expressions like `[ab]`.
    """

    def __init__(
        self,
        file_name_identifier=None,
        file_options_dict=None,
        files_to_be_deleted_regex_lst=None,
    ):
        """Init data processor class.

        Args:
            file_name_identifier (str): Identifier for files.
                                        The file prefix can contain regex expression and
                                        subdirectories.
            file_options_dict (dict): Dictionary with read-in options for
                                      the file. The respective child class will
                                       implement valid options for this dictionary.
            files_to_be_deleted_regex_lst (lst): List with paths to files that should be deleted.
                                                 The paths can contain regex expressions.
            logger_on_dask_worker (logging.Logger): Logger instance used on the dask worker
        """
        if not file_name_identifier:
            raise ValueError(
                f"No option 'file_name_identifier' was provided in '{self.__class__.__name__}'! "
                "DataProcessor object cannot be instantiated! Abort..."
            )
        if not isinstance(file_name_identifier, str):
            raise TypeError(
                "The option 'file_name_identifier' must be of type 'str' "
                f"but is of type {type(file_name_identifier)}. Abort..."
            )

        if file_options_dict is None:
            raise ValueError(
                f"No option 'file_options_dict' was provided in '{self.__class__.__name__}'! "
                "DataProcessor object cannot be instantiated! Abort..."
            )
        if not isinstance(file_options_dict, dict):
            raise TypeError(
                "The option 'file_options_dict' must be of type 'dict' "
                f"but is of type {type(file_options_dict)}. Abort..."
            )

        if files_to_be_deleted_regex_lst is None:
            files_to_be_deleted_regex_lst = []
        if not isinstance(files_to_be_deleted_regex_lst, list):
            raise TypeError(
                "The option 'files_to_be_deleted_regex_lst' must be of type 'list' "
                f"but is of type {type(files_to_be_deleted_regex_lst)}. Abort..."
            )

        self.files_to_be_deleted_regex_lst = files_to_be_deleted_regex_lst
        self.file_options_dict = file_options_dict
        self.file_name_identifier = file_name_identifier

        self.logger_on_worker = None

    def get_data_from_file(self, base_dir):
        """Get data of interest from file.

        Args:
            base_dir (Path): Path of the base directory that contains the file of interest

        Returns:
            processed_data (np.array): Final data from data processor module
        """
        if not base_dir:
            raise ValueError(
                "The data processor requires a base_directory for the "
                "files to operate on! Your input was empty! Abort..."
            )
        if not isinstance(base_dir, Path):
            raise TypeError(
                "The argument 'base_dir' must be of type 'Path' "
                f"but is of type {type(base_dir)}. Abort..."
            )

        self.logger_on_worker = setup_logger_on_worker(
            name=type(self).__name__,
            log_dir=base_dir,
            level=logging.INFO,
        )

        file_path = self._check_file_exist_and_is_unique(base_dir)
        self.logger_on_worker.info("\nProcessing the following files:")
        self.logger_on_worker.info(str(file_path))
        processed_data = None
        if file_path:
            raw_data = self.get_raw_data_from_file(file_path)
            filtered_data = self.filter_and_manipulate_raw_data(raw_data)
            processed_data = self._subsequent_data_manipulation(filtered_data)

        self._clean_up(base_dir)
        return processed_data

    def _check_file_exist_and_is_unique(self, base_dir):
        """Check if file exists.

        Args:
            base_dir (Path): Path to base directory that contains file of interest

        Returns:
            file_path (str): Actual path to the file of interest.
        """
        file_list = list(base_dir.glob(self.file_name_identifier))

        if len(file_list) > 1:
            raise RuntimeError(
                "The data_processor module found several files for the "
                "provided 'file_name_prefix'!"
                "The files are: {file_list}."
                "The file prefix must lead to a unique file. Abort..."
            )
        if len(file_list) == 1:
            file_path = file_list[0]
        else:
            self.logger_on_worker.warning(
                "The file '%s' does not exist!", base_dir / self.file_name_identifier
            )
            file_path = None

        return file_path

    @abc.abstractmethod
    def get_raw_data_from_file(self, file_path):
        """Get the raw data from the files of interest.

        Args:
            file_path (str): Actual path to the file of interest.

        Returns:
            raw_data (obj): Raw data from file.
        """

    def filter_and_manipulate_raw_data(self, raw_data):
        """Filter or clean the raw data for given criteria.

        Args:
            raw_data (obj): Raw data from file.

        Returns:
            processed_data (np.array): Cleaned, filtered or manipulated *data_processor* data.
        """
        return raw_data

    def _subsequent_data_manipulation(self, processed_data):
        """Subsequent manipulate the data_processor data.

        This method can be easily implemented by overloading the empty
        method in a custom inheritance of a desired child class. Make
        sure to add the module to the `from_config_create` method in
        this file.

        Args:
            processed_data (np.array): Cleaned, filtered or manipulated *data_processor* data.

        Returns:
            processed_data (np.array): Cleaned, filtered or manipulated *data_processor* data.
        """
        return processed_data

    def _clean_up(self, base_dir_file):
        """Clean-up files in the output directory.

        Args:
            base_dir_file (Path): Path of the base directory that
                                    contains the file of interest.
        """
        for regex in self.files_to_be_deleted_regex_lst:
            for file in sorted(base_dir_file.glob(regex)):
                file.unlink(missing_ok=True)
