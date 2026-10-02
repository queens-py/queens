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
"""Data processor class for csv data extraction."""

import logging
from pathlib import Path
from typing import override

import numpy as np
import pandas as pd

from queens.data_processors._data_processor import DataProcessor
from queens.utils.logger_settings import log_init_args
from queens.utils.valid_options import get_option

_logger = logging.getLogger(__name__)


class CsvFile(DataProcessor):
    """Class for extracting data from csv files.

    Attributes:
        use_cols_lst (lst): List with column numbers that should be read-in.
        filter_type (str): Filter type to use.
        header_row (int): Integer that determines which csv-row contains labels/headers of
                          the columns. Default is 'None', meaning no header used.
        skip_rows (int): Number of rows that should be skipped to be read-in in csv file.
        index_column (int, str): Column to use as the row labels of the DataFrame, either
                                 given as string name or column index.

                                 **Note:** *index_column=False* can be used to force pandas
                                 to not use the first column as the index. *index_column* is
                                 used for filtering the remaining columns.
        use_rows_lst (lst): In case this options is used, the list contains the indices of rows
                            in the csv file that should be used as data.
        filter_range (lst): After data is selected by *use_cols_lst* and a filter column is
                            specified by *index_column*, this option selects which data range
                            shall be filtered by providing a minimum and maximum value pair
                            in list format.
        filter_target_values (list): Target values to filter.
        filter_tol (float): Tolerance for the filter range.
        returned_filter_format (str): Returned data format after filtering.
    """

    expected_filter_entire_file = {"type": "entire_file"}
    expected_filter_by_row_index = {"type": "by_row_index", "rows": [1, 2]}
    expected_filter_by_target_values = {
        "type": "by_target_values",
        "target_values": [1.0, 2.0, 3.0],
        "tolerance": 0.0,
    }
    expected_filter_by_range = {"type": "by_range", "range": [1.0, 2.0], "tolerance": 0.0}

    @log_init_args
    def __init__(
        self,
        file_name_identifier: str,
        file_options_dict: dict,
        files_to_be_deleted_regex_lst: list[str] | None = None,
    ) -> None:
        """Instantiate data processor class for csv data.

        Args:
            file_name_identifier: Identifier of file name. The file prefix can contain regex
                expression and subdirectories.
            file_options_dict: Dictionary with read-in options for the file:

                - header_row (int):
                    Integer that determines which csv-row contains labels/headers of
                    the columns. Default is 'None', meaning no header used.
                - use_cols_lst (lst):
                    (optional) list with column numbers that should be read-in.
                - skip_rows (int):
                    Number of rows that should be skipped to be read-in in csv file.
                - index_column (int, str):
                    Column to use as the row labels of the DataFrame, either
                    given as string name or column index.
                    Note: ``index_column=False`` can be used to force pandas to not use the first
                    column as the index. ``index_column`` is used for filtering the remaining
                    columns.
                - returned_filter_format (str):
                    Returned data format after filtering
                - filter (dict):
                    Dictionary with filter options:

                    -- type (str):
                        Filter type to use
                    -- rows (lst):
                        In case this options is used, the list contains the indices of
                        rows in the csv file that should be used as data
                    -- range (lst):
                        After data is selected by `use_cols_lst` and a filter column is specified
                        by `index_column`, this option selects which data range shall be filtered
                        by providing a minimum and maximum value pair in list format
                    -- target_values (list):
                        target values to filter
                    -- tolerance (float):
                        Tolerance for the filter range

            files_to_be_deleted_regex_lst: List with paths to files that should be deleted.
                The paths can contain regex expressions.

        Returns:
            Instance of CsvFile class
        """
        super().__init__(
            file_name_identifier=file_name_identifier,
            file_options_dict=file_options_dict,
            files_to_be_deleted_regex_lst=files_to_be_deleted_regex_lst,
        )

        header_row = self.file_options_dict.get("header_row")
        if header_row and not isinstance(header_row, int):
            raise ValueError(
                "The option 'header_row' in the data_processor settings must be of type 'int'! "
                f"You provided type '{type(header_row)}'. Abort..."
            )

        use_cols_lst = self.file_options_dict.get("use_cols_lst")
        if use_cols_lst and not isinstance(use_cols_lst, list):
            raise TypeError(
                "The option 'use_cols_lst' must be of type 'list' "
                f"but you provided type {type(use_cols_lst)}. Abort..."
            )

        skip_rows = self.file_options_dict.get("skip_rows", 0)
        if not isinstance(skip_rows, int):
            raise ValueError(
                "The option 'skip_rows' in the data_processor settings must be of type 'int'! "
                f"You provided type '{type(skip_rows)}'. Abort..."
            )

        index_column = self.file_options_dict.get("index_column", False)
        if index_column and not isinstance(index_column, (int, str)):
            raise TypeError(
                "The option 'index_column' must be either of type 'int' or 'str', "
                f"but you provided type {type(index_column)}! Either your original data "
                "type is wrong or the column does not exist in the csv-data file! "
                "Abort..."
            )

        returned_filter_format = self.file_options_dict.get("returned_filter_format", "numpy")

        filter_options_dict = self.file_options_dict.get("filter")
        if not isinstance(filter_options_dict, dict):
            raise TypeError(
                "The option 'filter' must be of type 'dict' "
                f"but you provided type {type(filter_options_dict)}. Abort..."
            )
        self.check_valid_filter_options(filter_options_dict)

        filter_type = filter_options_dict.get("type")
        if not isinstance(filter_type, str):
            raise ValueError(
                "The option 'type' in the data_processor settings must be of type 'str'! "
                f"You provided type '{type(filter_type)}'. Abort..."
            )

        use_rows_lst: list[int] = filter_options_dict.get("rows", [])
        if not isinstance(use_rows_lst, list):
            raise TypeError(
                "The option 'rows' must be of type 'list' "
                f"but you provided type {type(use_rows_lst)}. Abort..."
            )
        if not all(isinstance(row_idx, int) for row_idx in use_rows_lst):
            raise TypeError(
                "The option 'rows' must be a list of `int` "
                f"but you provided type {[type(row_idx) for row_idx in use_rows_lst]}. Abort..."
            )

        filter_range: list[float] = filter_options_dict.get("range", [])
        if filter_range and not isinstance(filter_range, list):
            raise TypeError(
                "The option 'range' has to be of type 'list', "
                f"but you provided type {type(filter_range)}. Abort..."
            )

        filter_target_values: list[float] = filter_options_dict.get("target_values", [])
        if not isinstance(filter_target_values, list):
            raise TypeError(
                "The option 'target_values' has to be of type 'list', "
                f"but you provided type {type(filter_target_values)}. Abort..."
            )

        filter_tol = filter_options_dict.get("tolerance", 0.0)
        if not isinstance(filter_tol, float):
            raise TypeError(
                "The option 'tolerance' has to be of type 'float', "
                f"but you provided type {type(filter_tol)}. Abort..."
            )

        self.use_cols_lst = use_cols_lst
        self.filter_type = filter_type
        self.header_row = header_row
        self.skip_rows = skip_rows
        self.index_column = index_column
        self.use_rows_lst = use_rows_lst
        self.filter_range = filter_range
        self.filter_target_values = filter_target_values
        self.filter_tol = filter_tol
        self.returned_filter_format = returned_filter_format

    @classmethod
    def check_valid_filter_options(cls, filter_options_dict: dict) -> None:
        """Check valid filter input options.

        Args:
            filter_options_dict: dictionary with filter options
        """
        if filter_options_dict["type"] == "entire_file":
            if not filter_options_dict.keys() == cls.expected_filter_entire_file.keys():
                raise TypeError(
                    "For the filter type `entire_file`, you have to provide "
                    f"a dictionary of type {cls.expected_filter_entire_file}."
                )
            return
        if filter_options_dict["type"] == "by_range":
            if not filter_options_dict.keys() == cls.expected_filter_by_range.keys():
                raise TypeError(
                    "For the filter type `by_range`, you have to provide "
                    f"a dictionary of type {cls.expected_filter_by_range}."
                )
            return
        if filter_options_dict["type"] == "by_row_index":
            if not filter_options_dict.keys() == cls.expected_filter_by_row_index.keys():
                raise TypeError(
                    "For the filter type `by_row_index`, you have to provide "
                    f"a dictionary of type {cls.expected_filter_by_row_index}."
                )
            return
        if filter_options_dict["type"] == "by_target_values":
            if not filter_options_dict.keys() == cls.expected_filter_by_target_values.keys():
                raise TypeError(
                    "For the filter type `by_target_values`, you have to provide "
                    f"a dictionary of type {cls.expected_filter_by_target_values}."
                )
        else:
            raise TypeError("You provided an invalid 'filter_type'!")

    @override
    def get_raw_data_from_file(self, file_path: str | Path) -> pd.DataFrame | None:
        """Get the raw data from the files of interest.

        This method loads the desired parts of the csv file as a pandas
        dataframe.

        Args:
            file_path: Actual path to the file of interest.

        Returns:
            Raw data from file, or *None* if the file could not be read.
        """
        try:
            raw_data = pd.read_csv(
                file_path,
                sep=r",|\s+",
                usecols=self.use_cols_lst,
                skiprows=self.skip_rows,
                header=self.header_row,
                engine="python",
                index_col=self.index_column,
            )
            _logger.info("Successfully read-in data from %s.", file_path)
            return raw_data
        except IOError as error:
            # pylint: disable=duplicate-code
            _logger.warning(
                "Could not read the file: %s. The following IOError was raised: %s. "
                "Skipping the file and continuing.",
                file_path,
                error,
            )
            return None

    @override
    def filter_and_manipulate_raw_data(self, raw_data: pd.DataFrame) -> np.ndarray | dict:
        """Filter the pandas data-frame based on filter type.

        Args:
            raw_data: Raw data from file.

        Returns:
            Filtered data. The returned data type is determined by
                file_options_dict['returned_filter_format'].
        """
        valid_filter_types = {
            "entire_file": self._filter_entire_file,
            "by_range": self._filter_by_range,
            "by_row_index": self._filter_by_row_index,
            "by_target_values": self._filter_by_target_values,
        }

        error_message = "You provided an invalid 'filter_type'!"
        filter_method = get_option(
            valid_filter_types, self.filter_type, error_message=error_message
        )
        filtered_data = filter_method(raw_data)
        if filtered_data is None:
            raise RuntimeError(
                "The filtered data was empty! Adjust your filter tolerance or filter range!"
            )
        filter_formats_dict = {
            "numpy": filtered_data.to_numpy(),
            "dict": filtered_data.to_dict("list"),
        }

        processed_data = get_option(
            filter_formats_dict,
            self.returned_filter_format,
            error_message="The returned filter format you provided is not a current option.",
        )

        if not np.any(processed_data):
            raise RuntimeError(
                "The filtered data was empty! Adjust your filter tolerance or filter range!"
            )
        return processed_data

    def _filter_entire_file(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """Keep entire csv file data.

        Args:
            raw_data: Raw data from file.

        Returns:
            Raw data from file.
        """
        return raw_data

    def _filter_by_row_index(self, raw_data: pd.DataFrame) -> pd.DataFrame | None:
        """Filter the csv file based on given data rows.

        Args:
            raw_data: Raw data from file.

        Returns:
            Filtered data.
        """
        if not raw_data.empty:
            try:
                return raw_data.iloc[self.use_rows_lst]
            except IndexError as exception:
                raise IndexError(
                    f"Index list {self.use_rows_lst} are not contained in raw_file_data. "
                ) from exception
        return None

    def _filter_by_target_values(self, raw_data: pd.DataFrame) -> pd.DataFrame | None:
        """Filter the pandas data frame based on target values.

        Args:
            raw_data: Raw data from file.

        Returns:
            Filtered data.
        """
        if not raw_data.empty:
            target_indices = raw_data.index.get_indexer(
                self.filter_target_values,
                method="nearest",
                tolerance=self.filter_tol,
            )

            if (target_indices == -1).any():
                missing_targets = np.asarray(self.filter_target_values)[target_indices == -1]
                raise RuntimeError(
                    f"No index values found within tolerance {self.filter_tol} "
                    f"for target values {missing_targets.tolist()}."
                )

            return raw_data.iloc[target_indices]
        return None

    def _filter_by_range(self, raw_data: pd.DataFrame) -> pd.DataFrame | None:
        """Filter the pandas data frame based on values in a data column.

        Args:
            raw_data: Raw data from file.

        Returns:
            Filtered data.
        """
        if not raw_data.empty:
            range_start, range_end = raw_data.index.get_indexer(
                self.filter_range,
                method="nearest",
                tolerance=self.filter_tol,
            )

            if -1 in (range_start, range_end):
                missing_targets = np.asarray(self.filter_range)[
                    np.asarray([range_start, range_end]) == -1
                ]
                raise RuntimeError(
                    f"No index values found within tolerance {self.filter_tol} "
                    f"for range values {missing_targets.tolist()}."
                )

            return raw_data.iloc[range_start : range_end + 1]
        return None
