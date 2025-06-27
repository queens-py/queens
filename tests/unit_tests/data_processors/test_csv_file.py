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
"""Tests for data processor csv routine."""

import numpy as np
import pandas as pd
import pytest

from queens.data_processors.csv_file import (
    CsvFile,
    EntireFileFilter,
    RangeFilter,
    RowFilter,
    TargetFilter,
)


@pytest.fixture(name="dummy_data", scope="session")
def fixture_dummy_data():
    """Create dummy data for tests."""
    data = {
        "step": np.arange(10),
        "time": np.linspace(0, 1, 10),
        "x": np.linspace(10, 20, 10),
        "y": np.linspace(20, 30, 10),
    }

    data = pd.DataFrame(data)
    return data


@pytest.fixture(name="dummy_csv_file")
def fixture_dummy_csv_file(tmp_path, dummy_data):
    """Create dummy csv-file for tests."""
    dummy_data_path = tmp_path / "file.csv"
    dummy_data.to_csv(dummy_data_path, index=False)

    dummy_data_path.write_text("#These\n#are\n#rows are empty\n" + dummy_data_path.read_text())
    return dummy_data_path


def test_entire_file_filter(dummy_data):
    """Test entire file filter."""
    pandas_filter = EntireFileFilter()
    data = pandas_filter(dummy_data)
    np.testing.assert_allclose(data, dummy_data.to_numpy())


def test_entire_file_filter_dict(dummy_data):
    """Test entire file filter dict."""
    pandas_filter = EntireFileFilter(result_format="dict")
    data = pandas_filter(dummy_data)
    assert data == dummy_data.to_dict(orient="list")


def test_row_filter(dummy_data):
    """Test row filter."""
    pandas_filter = RowFilter([1, 9])
    data = pandas_filter(dummy_data)
    np.testing.assert_allclose(data, dummy_data.to_numpy()[[1, 9]])


def test_target_filter(dummy_data):
    """Test target filter."""
    pandas_filter = TargetFilter(target_values=[1, 9], tolerance=0.01)
    data = pandas_filter(dummy_data)
    np.testing.assert_allclose(data, dummy_data.to_numpy()[[1, 9]])


def test_range_filter(dummy_data):
    """Test range filter."""
    pandas_filter = RangeFilter(target_range=[2, 8], tolerance=0.01)
    data = pandas_filter(dummy_data)
    np.testing.assert_allclose(data, dummy_data.to_numpy()[2:9])


def test_csv_dataprocessor(dummy_csv_file, dummy_data):
    """Test dataprocessing reading."""
    dp = CsvFile(
        "no name",
        header_row=3,
        use_cols_lst=[2, 3],
        skip_rows=3,
        index_column=0,
        pandas_filter=TargetFilter(target_values=[13.3333, 18.88888], tolerance=0.01),
    )
    data = dp.get_data_from_file(dummy_csv_file)
    np.testing.assert_allclose(data, dummy_data.to_numpy()[[3, 8]][:, -1].reshape(-1, 1))
