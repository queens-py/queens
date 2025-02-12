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
"""Tests for numpy data processor."""

from pathlib import Path

import numpy as np
import pytest

from queens.data_processors.numpy import Numpy


# ------ fixtures ----------
@pytest.fixture(name="data_path")
def fixture_data_path(tmp_path):
    """Create a dummy data path."""
    my_data_path = tmp_path / "my_data.npy"
    return my_data_path


@pytest.fixture(name="wrong_data_path")
def fixture_wrong_data_path(tmp_path):
    """Create a dummy data path."""
    my_data_path = tmp_path / "my_data.csv"
    return my_data_path


@pytest.fixture(name="dummy_data")
def fixture_dummy_data():
    """Create some dummy data."""
    data = np.array([[1, 2], [3, 4]])
    return data


@pytest.fixture(name="_write_dummy_data")
def fixture_write_dummy_data(data_path, dummy_data):
    """Write the dummy data."""
    with open(data_path, "wb") as f:
        np.save(f, dummy_data)


@pytest.fixture(name="_write_wrong_dummy_data")
def fixture_write_wrong_dummy_data(wrong_data_path, dummy_data):
    """Write the wrong dummy data."""
    with open(wrong_data_path, "w", encoding="utf-8") as f:
        np.savetxt(f, dummy_data)


@pytest.fixture(name="default_data_processor_npy")
def fixture_default_data_processor_npy():
    """Dummy data processor npy."""
    file_name_identifier = "dummy"
    file_options_dict = {}
    files_to_be_deleted_regex_lst = []

    data_processor = Numpy(
        file_name_identifier,
        file_options_dict,
        files_to_be_deleted_regex_lst,
    )

    return data_processor


# ------ actual tests ------
def test_init():
    """Test initialization of npy data processor."""
    file_name_identifier = "dummy"
    file_options_dict = {"dummy": "dummy"}
    files_to_be_deleted_regex_lst = ["abc"]

    data_processor = Numpy(
        file_name_identifier,
        file_options_dict,
        files_to_be_deleted_regex_lst,
    )

    assert data_processor.files_to_be_deleted_regex_lst == files_to_be_deleted_regex_lst
    assert data_processor.file_options_dict == file_options_dict
    assert data_processor.file_name_identifier == file_name_identifier


def test_get_raw_data_from_file(
    default_data_processor_npy,
    data_path,
    dummy_data,
    _write_dummy_data,
):
    """Test get raw data from file."""
    raw_data = default_data_processor_npy.get_raw_data_from_file(data_path)
    np.testing.assert_array_equal(raw_data, dummy_data)


def test_non_existing_file(default_data_processor_npy, data_path):
    """Test non-existing raw data file."""
    default_data_processor_npy.file_path = Path("non_existing_file.npy")
    raw_data = default_data_processor_npy.get_raw_data_from_file(data_path)
    assert raw_data is None


def test_wrong_file_type(default_data_processor_npy, wrong_data_path, _write_wrong_dummy_data):
    """Test with wrong file type."""
    raw_data = default_data_processor_npy.get_raw_data_from_file(wrong_data_path)
    assert raw_data is None


def test_filter_and_manipulate_raw_data(default_data_processor_npy, dummy_data):
    """Test filter and manipulate raw data."""
    processed_data = default_data_processor_npy.filter_and_manipulate_raw_data(dummy_data)
    np.testing.assert_array_equal(processed_data, dummy_data)
