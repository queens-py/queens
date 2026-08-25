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
"""Unit tests for the experimental data reader."""

from pathlib import Path
from typing import Any, override

import numpy as np
import pytest

from queens.data_processors._data_processor import DataProcessor
from queens.data_processors.csv_file import CsvFile
from queens.utils.experimental_data_reader import ExperimentalDataReader

# ------ test data ----------
FILE_NAME = "experimental_data.csv"
BASE_DIR = Path("/some/base/dir")

OUTPUT_LABEL = "output"
COORDINATE_LABELS = ["x", "y"]
TIME_LABEL = "time"

# one row per observation point, two coordinates and two observation times
COORDINATES = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
TIMES = np.array([0.5, 0.5, 0.1, 0.1])
OUTPUTS = np.array([1.0, 2.0, 3.0, 4.0])
UNIQUE_TIMES = np.unique(TIMES)


# ------ fixtures ----------
@pytest.fixture(name="experimental_data")
def fixture_experimental_data():
    """Column-wise experimental data as returned by a data processor."""
    data = dict(zip(COORDINATE_LABELS, COORDINATES.T.tolist()))
    data[TIME_LABEL] = TIMES.tolist()
    data[OUTPUT_LABEL] = OUTPUTS.tolist()
    return data


class DummyDataProcessor(DataProcessor):
    """Data processor returning a fixed dictionary instead of a file."""

    def __init__(self, data):
        """Initialize with the data that should be returned."""
        super().__init__(file_name_identifier=FILE_NAME, file_options_dict={})
        self.data = data
        self.base_dir_file = None

    @override
    def get_data_from_file(self, base_dir_file: Path) -> Any:
        """Return the stored data and record the base directory."""
        self.base_dir_file = base_dir_file
        return self.data

    @override
    def get_raw_data_from_file(self, file_path: str | Path) -> Any:
        """Not used, since `get_data_from_file` is overridden."""
        raise NotImplementedError


@pytest.fixture(name="csv_file_path")
def fixture_csv_file_path(tmp_path, experimental_data):
    """Write the experimental data to a csv file."""
    file_path = tmp_path / FILE_NAME
    rows = zip(*experimental_data.values())
    lines = [",".join(experimental_data.keys())] + [
        ",".join(str(value) for value in row) for row in rows
    ]
    file_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return file_path


# ------ actual tests ----------
def test_init_default_data_processor():
    """Test that a csv data processor is created by default."""
    reader = ExperimentalDataReader(
        file_name_identifier=FILE_NAME,
        output_label=OUTPUT_LABEL,
        coordinate_labels=COORDINATE_LABELS,
        time_label=TIME_LABEL,
        csv_data_base_dir=BASE_DIR,
    )

    assert reader.output_label == OUTPUT_LABEL
    assert reader.coordinate_labels == COORDINATE_LABELS
    assert reader.time_label == TIME_LABEL
    assert reader.file_name == FILE_NAME
    assert reader.base_dir == BASE_DIR
    assert isinstance(reader.data_processor, CsvFile)
    assert reader.data_processor.file_name_identifier == FILE_NAME
    assert reader.data_processor.returned_filter_format == "dict"


def test_init_given_data_processor(experimental_data):
    """Test that a provided data processor is not overwritten."""
    data_processor = DummyDataProcessor(experimental_data)

    reader = ExperimentalDataReader(file_name_identifier=FILE_NAME, data_processor=data_processor)

    assert reader.data_processor is data_processor


def test_get_experimental_data(experimental_data):
    """Test that outputs, coordinates and times are extracted correctly."""
    data_processor = DummyDataProcessor(experimental_data)
    reader = ExperimentalDataReader(
        file_name_identifier=FILE_NAME,
        data_processor=data_processor,
        output_label=OUTPUT_LABEL,
        coordinate_labels=COORDINATE_LABELS,
        time_label=TIME_LABEL,
        csv_data_base_dir=BASE_DIR,
    )

    (
        y_obs_vec,
        coordinates,
        time_vec,
        data_dict,
        time_label,
        coordinate_labels,
        output_label,
    ) = reader.get_experimental_data()

    np.testing.assert_array_equal(y_obs_vec, OUTPUTS)
    np.testing.assert_array_equal(coordinates, COORDINATES)
    np.testing.assert_array_equal(time_vec, UNIQUE_TIMES)
    assert data_dict == experimental_data
    assert time_label == TIME_LABEL
    assert coordinate_labels == COORDINATE_LABELS
    assert output_label == OUTPUT_LABEL
    assert data_processor.base_dir_file == BASE_DIR


def test_get_experimental_data_from_csv_file(csv_file_path, experimental_data):
    """Test reading experimental data from an actual csv file."""
    reader = ExperimentalDataReader(
        file_name_identifier=csv_file_path.name,
        output_label=OUTPUT_LABEL,
        coordinate_labels=COORDINATE_LABELS,
        time_label=TIME_LABEL,
        csv_data_base_dir=csv_file_path.parent,
    )

    y_obs_vec, coordinates, time_vec, data_dict, _, _, _ = reader.get_experimental_data()

    np.testing.assert_array_equal(y_obs_vec, OUTPUTS)
    np.testing.assert_array_equal(coordinates, COORDINATES)
    np.testing.assert_array_equal(time_vec, UNIQUE_TIMES)
    assert data_dict == experimental_data
