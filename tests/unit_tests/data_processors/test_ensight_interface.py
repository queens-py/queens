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
"""Tests for distance to surface measurement data_processors evaluation."""

import numpy as np
import pytest

from queens.data_processors.ensight_interface import EnsightInterfaceDiscrepancy


############## fixtures
@pytest.fixture(name="all_dimensions", scope="module", params=["2d", "3d"])
def fixture_all_dimensions(request):
    """Parameterized fixture to select problem dimension."""
    return request.param


@pytest.fixture(name="default_data_processor")
def fixture_default_data_processor(mocker):
    """Default ensight class for upcoming tests."""
    file_name_identifier = "dummy_prefix*dummyfix"
    file_options_dict = {
        "path_to_ref_data": "dummy_path",
        "time_tol": 1e-03,
        "visualization": False,
        "displacement_fields": ["first_disp", "second_disp"],
        "problem_dimension": "5d",
    }
    file_to_be_deleted_regex_lst = []

    mocker.patch(
        "queens.data_processors.ensight_interface.EnsightInterfaceDiscrepancy.read_monitorfile",
        return_value="None",
    )
    pp = EnsightInterfaceDiscrepancy(
        file_name_identifier,
        file_options_dict,
        file_to_be_deleted_regex_lst,
    )
    return pp


# --------------- actual tests -------------------------


def test_init(mocker):
    """Test the init method."""
    experimental_ref_data = "dummy_data"
    displacement_fields = ["first_disp", "second_disp"]
    time_tol = 1e-03
    visualization_bool = False
    files_to_be_deleted_regex_lst = []
    problem_dim = "5d"

    file_name_identifier = "dummy_prefix*dummyfix"
    file_options_dict = {
        "path_to_ref_data": "dummy_path",
        "time_tol": time_tol,
        "visualization": visualization_bool,
        "displacement_fields": displacement_fields,
        "problem_dimension": problem_dim,
    }

    mocker.patch(
        "queens.data_processors.ensight_interface.EnsightInterfaceDiscrepancy.read_monitorfile",
        return_value="dummy_data",
    )
    my_data_processor = EnsightInterfaceDiscrepancy(
        file_name_identifier,
        file_options_dict,
        files_to_be_deleted_regex_lst,
    )

    assert my_data_processor.time_tol == time_tol
    assert my_data_processor.visualization_bool is visualization_bool
    assert my_data_processor.displacement_fields == displacement_fields
    assert my_data_processor.problem_dimension == problem_dim
    assert my_data_processor.experimental_ref_data_lst == experimental_ref_data

    assert my_data_processor.files_to_be_deleted_regex_lst == files_to_be_deleted_regex_lst
    assert my_data_processor.file_options_dict == file_options_dict
    assert my_data_processor.file_name_identifier == file_name_identifier


def test_from_config_create_data_processor(mocker):
    """Test the config method."""
    experimental_ref_data = np.array([[1, 2], [3, 4]])
    mp = mocker.patch(
        "queens.data_processors.ensight_interface.EnsightInterfaceDiscrepancy.__init__",
        return_value=None,
    )

    mocker.patch(
        "queens.data_processors.ensight_interface.EnsightInterfaceDiscrepancy.read_monitorfile",
        return_value=experimental_ref_data,
    )
    file_name_identifier = "dummyprefix*dummy.case"
    time_tol = 1e-03
    visualization_bool = False
    displacement_fields = ["first_disp", "second_disp"]
    delete_field_data = False
    problem_dimension = "5d"
    path_to_ref_data = "some_path"
    files_to_be_deleted_regex_lst = []

    file_options_dict = {
        "time_tol": time_tol,
        "visualization_bool": visualization_bool,
        "displacement_fields": displacement_fields,
        "delete_field_data": delete_field_data,
        "problem_dimension": problem_dimension,
        "path_to_ref_data": path_to_ref_data,
    }

    EnsightInterfaceDiscrepancy(
        file_name_identifier=file_name_identifier,
        file_options_dict=file_options_dict,
        files_to_be_deleted_regex_lst=files_to_be_deleted_regex_lst,
    )
    mp.assert_called_once_with(
        file_name_identifier=file_name_identifier,
        file_options_dict=file_options_dict,
        files_to_be_deleted_regex_lst=files_to_be_deleted_regex_lst,
    )


def test_read_monitorfile(mocker):
    """Test reading of monitor file."""
    # monitor_string will be used to mock the content of a monitor file that is linked at
    # path_to_ref_data whereas the indentation is compulsory
    monitor_string = """#somecomment
steps 2 npoints 4
2 0 1
2 0 2
2 1 2
3 0 1 2
#comments here and in following lines
#lines above: #number of dimensions for point pairs #ID of coordinate directions
# following lines in scheme seperated by arbitrary number of spaces
# (first time point) x1 y1 x1' y1' x2 y2 x2' y2' x3 y3 x3' y3' x4 y4 x4' y4' x5 y5 x5' y5'
# (x y) is a location of the interface (x' y') is a point that is associated with
# the direction in which the distance to the interface is measured
# the vectors (x y)->(x' y') should point towards the interface
4.0e+00 1.0 1.0 1.0 1.0  2.0 2.0 2.0 2.0  3.0 3.0 3.0 3.0  1.0 1.0 1.0 1.0 1.0 1.0
8.0e+00 5.0 5.0 5.0 5.0  6.0 6.0 6.0 6.0  7.0 7.0 7.0 7.0  5.0 5.0 5.0 5.0 5.0 5.0"""

    mp = mocker.patch("builtins.open", mocker.mock_open(read_data=monitor_string))
    data = EnsightInterfaceDiscrepancy.read_monitorfile("dummy_path")
    mp.assert_called_once()

    assert data == [
        [
            4.0,
            [
                [[1.0, 1.0, 0], [1.0, 1.0, 0]],
                [[2.0, 0, 2.0], [2.0, 0, 2.0]],
                [[0, 3.0, 3.0], [0, 3.0, 3.0]],
                [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
            ],
        ],
        [
            8.0,
            [
                [[5.0, 5.0, 0], [5.0, 5.0, 0]],
                [[6.0, 0, 6.0], [6.0, 0, 6.0]],
                [[0, 7.0, 7.0], [0, 7.0, 7.0]],
                [[5.0, 5.0, 5.0], [5.0, 5.0, 5.0]],
            ],
        ],
    ]

    monitor_string = """something wrong"""
    mocker.patch("builtins.open", mocker.mock_open(read_data=monitor_string))
    with pytest.raises(ValueError):
        EnsightInterfaceDiscrepancy.read_monitorfile("some_path")


def test_stretch_vector(default_data_processor):
    """Test for stretch vector helpre method."""
    assert default_data_processor.stretch_vector([1, 2, 3], [2, 4, 6], 2) == [
        [-1, -2, -3],
        [4, 8, 12],
    ]


def test_compute_distance(default_data_processor):
    """Test for distance computation."""
    assert default_data_processor.compute_distance(
        [[2, 4, 6], [1, 2, 3], [3, 6, 9]], [[0, 0, 0], [0.1, 0.2, 0.3]]
    ) == pytest.approx(np.sqrt(14), abs=10e-12)
