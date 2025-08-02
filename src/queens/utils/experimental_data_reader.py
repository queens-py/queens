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
"""Module to read experimental data."""

from pathlib import Path

import numpy as np

from queens.data_processors.csv_file import CsvFile
from queens.utils.logger_settings import log_init_args


class ExperimentalDataReader:
    """Reader for experimental data.

    Attributes:
        output_label (str): Label that marks the output quantity in the csv file
        coordinate_labels (lst): List of column-wise coordinate labels in csv files
        time_label (str): Name of the time variable in csv file
        file_name (str): File name of experimental data
        base_dir (Path): Path to base directory containing experimental data
        data_processor (DataProcessor): data processor for experimental data
    """

    @log_init_args
    def __init__(
        self,
        data_processor=None,
        output_label=None,
        coordinate_labels=None,
        time_label=None,
        file_name_identifier=None,
        csv_data_base_dir=None,
    ):
        """Initialize ExperimentalDataReader.

        Args:
            data_processor (DataProcessor): data processor for experimental data
            output_label (str): Label that marks the output quantity in the csv file
            coordinate_labels (lst): List of column-wise coordinate labels in csv files
            time_label (str): Name of the time variable in csv file
            file_name_identifier (str): File name of experimental data
            csv_data_base_dir (Path): Path to base directory containing experimental data
        """
        self.output_label = output_label
        self.coordinate_labels = coordinate_labels
        self.time_label = time_label
        self.file_name = file_name_identifier
        self.base_dir = Path(csv_data_base_dir)

        if data_processor is None:
            self.data_processor = CsvFile(
                file_name_identifier=self.file_name,
                file_options_dict={
                    "header_row": 0,
                    "index_column": False,
                    "returned_filter_format": "dict",
                    "filter": {"type": "entire_file"},
                },
            )

    def get_experimental_data(self):
        """Load experimental data.

        Returns:
            y_obs_vec (np.array): Column-vector of model outputs which correspond row-wise to
                                  observation coordinates
            experimental_coordinates (np.array): Matrix with observation coordinates. One row
                                                 corresponds to one coordinate point
            time_vec (np.array): Unique vector of observation times
            experimental_data_dict (dict): Dictionary containing the experimental data
            time_label (str): Name of the time variable in csv file
            coordinate_labels (lst): List of column-wise coordinate labels in csv files
        """
        experimental_data_dict = self.data_processor.get_data_from_file(self.base_dir)

        # arrange the experimental data coordinates
        experimental_coordinates = None
        if self.coordinate_labels:
            experimental_coordinates = (
                np.array(
                    [experimental_data_dict[coordinate] for coordinate in self.coordinate_labels]
                ),
            )[0].T

        # get a unique vector of observation times
        time_vec = None
        if self.time_label:
            time_vec = np.sort(list(set(experimental_data_dict[self.time_label])))

        # get the experimental outputs
        y_obs_vec = np.array(experimental_data_dict[self.output_label]).reshape(
            -1,
        )

        return (
            y_obs_vec,
            experimental_coordinates,
            time_vec,
            experimental_data_dict,
            self.time_label,
            self.coordinate_labels,
            self.output_label,
        )


VALID_TYPES = {"experimental_data_reader": ExperimentalDataReader}
