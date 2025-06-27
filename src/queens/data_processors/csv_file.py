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

import abc
import logging
from functools import partial

import numpy as np
import pandas as pd

from queens.data_processors._data_processor import DataProcessor
from queens.utils.logger_settings import log_init_args

_logger = logging.getLogger(__name__)


class PandasDataFrameFilter:
    """Filter for pandas dataframe."""

    def __init__(self, result_format="numpy"):
        """Initialiase pandas dataframe filter.

        Args:
            result_format (str): Either numpy or dict
        """
        if result_format == "numpy":
            self.return_function = pd.DataFrame.to_numpy
        elif result_format == "dict":
            self.return_function = partial(pd.DataFrame.to_dict, orient="list")
        else:
            raise ValueError(f"Expected result format dict or numpy not {result_format}")

    @abc.abstractmethod
    def filter(self, data):
        """Filter the pandas dataframe.

        Args:
            data (pandas.DataFrame): Dataframe to filter.

        Returns:
            pandas.DataFrame: Filtered data
        """

    def __call__(self, raw_data):
        """Do the filtering.

        Args:
            raw_data (pandas.DataFrame): Frame to filter

        Returns:
            np.ndarray: Result
        """
        if len(raw_data) > 0:
            return self.return_function(self.filter(raw_data))
        return None


class EntireFileFilter(PandasDataFrameFilter):
    """No filter."""

    def filter(self, data):
        """Nothing to filter.

        Args:
            data (pandas.DataFrame): Dataframe to filter.

        Returns:
            pandas.DataFrame: Filtered data
        """
        return data


class RowFilter(PandasDataFrameFilter):
    """Extract rows from dataframe."""

    def __init__(self, row_ids, result_format="numpy"):
        """Initialise row filter.

        Args:
            row_ids (list): List of row ids.
            result_format (str): Either numpy or dict
        """
        super().__init__(result_format)
        self.row_ids = row_ids

    def filter(self, data):
        """Extract rows.

        Args:
            data (pandas.DataFrame): Dataframe to filter.

        Returns:
            pandas.DataFrame: Filtered data
        """
        return data.iloc[self.row_ids]


class TargetFilter(PandasDataFrameFilter):
    """Target value filter."""

    def __init__(self, target_values, tolerance=0, result_format="numpy"):
        """Initialise target filter.

        Args:
            target_values (list): List of target index values
            tolerance (float): Tolerance to be within
            result_format (str): Either numpy or dict
        """
        super().__init__(result_format)
        self.target_values = target_values
        self.tolerance = tolerance

    def filter(self, data):
        """Extract by target.

        Args:
            data (pandas.DataFrame): Dataframe to filter.

        Returns:
            pandas.DataFrame: Filtered data
        """
        target_indices = []
        for target_value in self.target_values:
            target_indices.append(
                int(np.where(np.abs(data.index - target_value) <= self.tolerance)[0])
            )

        return data.iloc[target_indices]


class RangeFilter(PandasDataFrameFilter):
    """Range filter."""

    def __init__(self, target_range, tolerance=0, result_format="numpy"):
        """Initialise range filter.

        Args:
            target_range (list): Range for the target values
            tolerance (float): Tolerance to be within
            result_format (str): Either numpy or dict
        """
        super().__init__(result_format)
        self.target_range = target_range
        self.tolerance = tolerance

    def filter(self, data):
        """Extract by range.

        Args:
            data (pandas.DataFrame): Dataframe to filter.

        Returns:
            pandas.DataFrame: Filtered data
        """
        range_start = int(np.where(np.abs(data.index - self.target_range[0]) <= self.tolerance)[0])
        range_end = int(np.where(np.abs(data.index - self.target_range[-1]) <= self.tolerance)[-1])

        return data.iloc[range_start : range_end + 1]


class CsvFile(DataProcessor):
    """Class for extracting data from csv files.

    Attributes:
        header_row (int): Integer that determines which csv-row contains labels/headers of
                          the columns. Default is 'None', meaning no header used.
        use_cols_lst (lst): List with column numbers that should be read-in.
        skip_rows (int): Number of rows that should be skipped to be read-in in csv file.
        index_column (int, str): Column to use as the row labels of the DataFrame, either
                                 given as string name or column index.

                                 **Note:** *index_column=False* can be used to force pandas
                                 to not use the first column as the index. *index_column* is
                                 used for filtering the remaining columns.
        pandas_filter (callable): Filter to apply
    """

    @log_init_args
    def __init__(
        self,
        file_name_identifier,
        header_row=None,
        use_cols_lst=None,
        skip_rows=0,
        index_column=False,
        pandas_filter=EntireFileFilter(),
        files_to_be_deleted_regex_lst=None,
    ):
        """Instantiate data processor class for csv data.

        Args:
            file_name_identifier (str): Identifier of file name
                                             The file prefix can contain regex expression
                                             and subdirectories.
            header_row (int): Integer that determines which csv-row contains labels/headers of
                              the columns. Default is 'None', meaning no header used.
            use_cols_lst (lst): (optional) list with column numbers that should be read-in.
            skip_rows (int): Number of rows that should be skipped to be read-in in csv file.
            index_column (int, str): Column to use as the row labels of the DataFrame, either
                                     given as string name or column index.
                                     Note: index_column=False can be used to force pandas to
                                     not use the first column as the index. Index_column is
                                     used for filtering the remaining columns.
            pandas_filter (callable): Filter to apply
            files_to_be_deleted_regex_lst (lst): List with paths to files that should be deleted.
                                                 The paths can contain regex expressions.

        Returns:
            Instance of CsvFile class
        """
        super().__init__(
            file_name_identifier=file_name_identifier,
            files_to_be_deleted_regex_lst=files_to_be_deleted_regex_lst,
        )

        self.header_row = header_row
        self.use_cols_lst = use_cols_lst
        self.skip_rows = skip_rows
        self.index_column = index_column

        self.pandas_filter = pandas_filter

    def get_raw_data_from_file(self, file_path):
        """Get the raw data from the files of interest.

        This method loads the desired parts of the csv file as a pandas
        dataframe.

        Args:
            file_path (str): Actual path to the file of interest.

        Returns:
            raw_data (DataFrame): Raw data from file.
        """
        try:
            kwargs = {
                "skiprows": self.skip_rows,
                "header": self.header_row,
                "index_col": self.index_column,
            }
            if self.use_cols_lst is not None:
                kwargs["usecols"] = self.use_cols_lst
            raw_data = pd.read_csv(file_path, sep=r",|\s+", engine="python", **kwargs)

            if self.pandas_filter is not None:
                return self.pandas_filter(raw_data)
            _logger.info("Successfully read-in data from %s.", file_path)
            return raw_data

        except IOError as error:
            _logger.warning(
                "Could not read the file: %s. The following IOError was raised: %s. "
                "Skipping the file and continuing.",
                file_path,
                error,
            )
            return None
