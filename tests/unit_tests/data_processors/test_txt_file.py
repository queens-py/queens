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
"""Tests for data processor txt routine."""

import pytest

from queens.data_processors.txt_file import TxtFile
from queens.utils.path import relative_path_from_queens


@pytest.fixture(name="dummy_txt_file", scope="session")
def fixture_dummy_txt_file():
    """Create dummy txt-file for tests."""
    txt_file_path = relative_path_from_queens(
        "tests/unit_tests/data_processors/queens_example_log.txt"
    )
    return txt_file_path


@pytest.fixture(name="default_data_processor")
def fixture_default_data_processor():
    """Default data processor txt class for unit tests."""
    file_name_identifier = "queens_example_log.txt"
    files_to_be_deleted_regex_lst = []

    file_options_dict = {}

    txt_instance = TxtFile(
        file_name_identifier,
        file_options_dict,
        files_to_be_deleted_regex_lst,
        remove_logger_prefix_from_raw_data=True,
    )
    return txt_instance


@pytest.fixture(name="default_raw_data")
def fixture_default_raw_data(default_data_processor, dummy_txt_file):
    """Default raw data for tests."""
    raw_data = default_data_processor.get_raw_data_from_file(dummy_txt_file)
    return raw_data


def test_check_file_size_exceeds_limit(default_data_processor, dummy_txt_file):
    """Throw a MemoryError for a file exceeding the size limit."""
    default_data_processor.max_file_size_in_mega_byte = 0.04
    with pytest.raises(MemoryError):
        default_data_processor._check_file_size(dummy_txt_file)  # pylint: disable=protected-access


def test_get_raw_data_from_file_remove_logger_prefix(default_raw_data):
    """Test the get_raw_data_from_file.

    This Test checks the removal of the leading regex in the log file
    inserted by the queens logger. The "filtered" queens log files is
    compared to the original 4C log file.
    """
    file_path_fourc_log = relative_path_from_queens(
        "tests/unit_tests/data_processors/fourc_example_log.txt"
    )
    with open(file_path_fourc_log, "r", encoding="utf-8") as file:
        raw_data_fourc = file.readlines()
    # Remove leading and trailing whitespaces from each string in the list
    raw_data_fourc = [s.strip() for s in raw_data_fourc]
    assert raw_data_fourc == default_raw_data
    assert len(default_raw_data) == 850
    assert default_raw_data[0] == ""
    assert (
        default_raw_data[363]
        == "Finalised step 1 / 10 | time 2.000e-02 | dt 2.000e-02 | numiter 2 | wct 1.86e+00"
    )
    assert default_raw_data[849] == "processor 2 finished normally"


def test_extract_lines_with_regex(default_data_processor, default_raw_data):
    """This test checks the extraction of an entire line."""
    regex = r"CORE::LINALG::Solver:  1\)   Setup"
    expected_val = (
        "CORE::LINALG::Solver:  1)   Setup                                   2.2577e+00 (19)      "
        "2.2580e+00 (19)      2.2583e+00 (19)      1.1884e-01 (19)"
    )
    matches = default_data_processor._extract_lines_with_regex(  # pylint: disable=protected-access
        default_raw_data, regex
    )

    assert len(matches) == 1
    assert matches[0][0] == 815
    assert matches[0][1] == expected_val


def test_extract_quantities_from_line(default_data_processor, default_raw_data):
    """This test checks the extraction quantities from a line."""
    regex_global = r"CORE::LINALG::Solver:  1\)   Setup"
    regex_numeric_vals = r"\b\d+\.\d+[eE][+-]?\d+\b"
    matches_global = (
        default_data_processor._extract_lines_with_regex(  # pylint: disable=protected-access
            default_raw_data, regex_global
        )
    )
    numeric_vals = (
        default_data_processor._extract_quantities_from_line(  # pylint: disable=protected-access
            matches_global[0][1], regex_numeric_vals
        )
    )

    assert len(numeric_vals) == 4
    assert numeric_vals[0] == "2.2577e+00"
    assert numeric_vals[1] == "2.2580e+00"
    assert numeric_vals[2] == "2.2583e+00"
    assert numeric_vals[3] == "1.1884e-01"


def test_extract_section_from_raw_data_start_and_end_marker(
    default_data_processor, default_raw_data
):
    """Test the extraction of a section with a start and end marker."""
    # Remove the simulation settings and the post-simulation section from raw data
    regex_simulation_start = r"^=+ Standard Lagrange multiplier strategy =+$"
    regex_simulation_end = r"TimeMonitor results over \d+ processors"
    simulation_raw_data = (
        default_data_processor._extract_section_from_raw_data(  # pylint: disable=protected-access
            default_raw_data,
            "start_end",
            regex_start=regex_simulation_start,
            regex_end=regex_simulation_end,
        )
    )
    assert len(simulation_raw_data) == 1
    assert len(simulation_raw_data[0]) == 632


def test_extract_section_from_raw_data_end_marker(default_data_processor, default_raw_data):
    """Test the extraction of a section with an end marker."""
    # Remove the simulation settings and the post-simulation section from raw data
    regex_simulation_start = "Parallel balance: t=0/restart"
    regex_simulation_end = r"TimeMonitor results over \d+ processors"
    simulation_raw_data = (
        default_data_processor._extract_section_from_raw_data(  # pylint: disable=protected-access
            default_raw_data,
            "start_end",
            regex_start=regex_simulation_start,
            regex_end=regex_simulation_end,
        )
    )
    assert len(simulation_raw_data) == 1

    regex_timestep_end = r"^Parallel balance \(eles\): \d+\.\d+e[+-]\d+ \(limit \d+\.\d+\)$"
    timestep_raw_data = (
        default_data_processor._extract_section_from_raw_data(  # pylint: disable=protected-access
            simulation_raw_data[0], "end", regex_end=regex_timestep_end
        )
    )
    assert len(timestep_raw_data) == 5
    assert len(timestep_raw_data[0]) == 180
    assert len(timestep_raw_data[1]) == 98
    assert len(timestep_raw_data[2]) == 80
    assert len(timestep_raw_data[3]) == 120
    assert len(timestep_raw_data[4]) == 130


def test_extract_section_from_raw_data_start_marker(default_data_processor, default_raw_data):
    """Test the extraction of a section with a start marker."""
    # Remove the simulation settings and the post-simulation section from raw data
    regex_simulation_start = r"^=+ Standard Lagrange multiplier strategy =+$"
    regex_simulation_end = r"TimeMonitor results over \d+ processors"
    simulation_raw_data = (
        default_data_processor._extract_section_from_raw_data(  # pylint: disable=protected-access
            default_raw_data,
            "start_end",
            regex_start=regex_simulation_start,
            regex_end=regex_simulation_end,
        )
    )
    assert len(simulation_raw_data) == 1

    regex_timestep_start = r"\*{58}"
    timestep_raw_data = (
        default_data_processor._extract_section_from_raw_data(  # pylint: disable=protected-access
            simulation_raw_data[0], "start", regex_start=regex_timestep_start
        )
    )
    assert len(timestep_raw_data) == 5
    assert len(timestep_raw_data[0]) == 179
    assert len(timestep_raw_data[1]) == 98
    assert len(timestep_raw_data[2]) == 80
    assert len(timestep_raw_data[3]) == 120
    assert len(timestep_raw_data[4]) == 130
