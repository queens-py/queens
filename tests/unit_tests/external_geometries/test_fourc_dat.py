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
"""Unit tests for FourcDat."""

import numpy as np
import pytest

from queens.external_geometries.fourc_dat import FourcDat


# general input fixtures
@pytest.fixture(name="default_geo_obj")
def fixture_default_geo_obj(tmp_path):
    """Create a default FourcDat object for testing."""
    path_to_dat_file = tmp_path / "myfile.dat"
    list_geometric_sets = ["DSURFACE 9"]
    list_associated_material_numbers = [[10, 11]]

    path_to_preprocessed_dat_file = tmp_path / "preprocessed"
    random_fields = (
        [{"name": "mat_param", "type": "material", "external_instance": "DSURFACE 1"}],
    )

    geo_obj = FourcDat(
        input_template=path_to_dat_file,
        input_template_preprocessed=path_to_preprocessed_dat_file,
        list_geometric_sets=list_geometric_sets,
        associated_material_numbers_geometric_set=list_associated_material_numbers,
        random_fields=random_fields,
    )
    return geo_obj


@pytest.fixture(name="dat_dummy_comment")
def fixture_dat_dummy_comment():
    """Provide dummy comment data for .dat files."""
    data = [
        "// this is a comment\n",
        " // this is another comment --------------------------------------"
        "-----------------NODE COORDS",
    ]
    data = "".join(data)
    return data


@pytest.fixture(name="dat_dummy_get_fun")
def fixture_dat_dummy_get_fun():
    """Provide dummy data for get function tests."""
    data = [
        "NODE    3419 DSURFACE 10\n",
        "NODE    3421 DSURFACE 10\n",
        "NODE    3423 DSURFACE 10\n",
        "NODE    3425 DSURFACE 10",
    ]
    data = "".join(data)
    return data


@pytest.fixture(
    name="dat_section_true",
    params=[
        "------------------------------------------------DESIGN DESCRIPTION    ",
        "------------------------------------------------DNODE-NODE TOPOLOGY   ",
        "------------------------------------------------DLINE-NODE TOPOLOGY   ",
        "------------------------------------------------DSURF-NODE TOPOLOGY   ",
        "------------------------------------------------DVOL-NODE TOPOLOGY    ",
        "------------------------------------------------NODE COORDS//         ",
    ],
)
def fixture_dat_section_true(request):
    """Provide valid .dat file section names."""
    return request.param


@pytest.fixture(
    name="dat_section_false",
    params=[
        "//------------------------------------------------DESIGN DESCRIPTION    ",
        " // ------------------------------------------------DNODE-NODE TOPOLOGY   ",
        "------- //-----------------------------------------DLINE-NODE TOPOLOGY   ",
        "------------------------------------------------DSRF-NDE TOOGY   ",
        "------------------------------------------------VOL-NODE TOPOLOGY    ",
        "------------------------------------------------NODECOORDS//           ",
    ],
)
def fixture_dat_section_false(request):
    """Provide invalid .dat file section names."""
    return request.param


@pytest.fixture(
    name="current_dat_sections",
    params=[
        "DNODE-NODE TOPOLOGY",
        "DLINE-NODE TOPOLOGY",
        "DSURF-NODE TOPOLOGY",
        "DVOL-NODE TOPOLOGY",
    ],
)
def fixture_current_dat_sections(request):
    """Provide current .dat file section names."""
    return request.param


@pytest.fixture(name="desired_sections")
def fixture_desired_sections():
    """Provide desired .dat file sections."""
    sections = {
        "DLINE-NODE TOPOLOGY": ["DLINE 1"],
        "DSURF-NODE TOPOLOGY": ["DSURFACE 2", "DSURFACE 1"],
        "DNODE-NODE TOPOLOGY": ["DNODE 1"],
        "DVOL-NODE TOPOLOGY": ["DVOL 1"],
    }
    return sections


@pytest.fixture(name="default_coords")
def fixture_default_coords():
    """Provide default node coordinates."""
    coords = [
        "NODE 1 COORD -1.0000000000000000e+00 -2.5000000000000000e-01 0.0000000000000000e+00",
        "NODE 2 COORD -1.0000000000000000e+00 -2.5000000000000000e-01 -3.7500000000000006e-02",
        "NODE 3 COORD -1.0000000000000000e+00 -2.1666666666666665e-01 -3.7500000000000006e-02",
        "NODE 4 COORD -1.0000000000000000e+00 -2.1666666666666662e-01 0.0000000000000000e+00",
    ]
    return coords


@pytest.fixture(name="default_topology_node")
def fixture_default_topology_node():
    """Provide default node topology."""
    data = [
        "NODE    1 DNODE 1",
        "NODE    2 DNODE 1",
        "NODE    4 DNODE 1",
        "NODE    9 DNODE 1",
        "NODE    13 DNODE 1",
    ]
    return data


@pytest.fixture(name="default_topology_line")
def fixture_default_topology_line():
    """Provide default line topology."""
    data = [
        "NODE    1 DLINE 1",
        "NODE    2 DLINE 1",
        "NODE    4 DLINE 1",
        "NODE    9 DLINE 1",
        "NODE    13 DLINE 1",
    ]
    return data


@pytest.fixture(name="default_topology_surf")
def fixture_default_topology_surf():
    """Provide default surface topology."""
    data = [
        "NODE    1 DSURFACE 1",
        "NODE    2 DSURFACE 1",
        "NODE    4 DSURFACE 1",
        "NODE    9 DSURFACE 1",
        "NODE    13 DSURFACE 1",
    ]
    return data


@pytest.fixture(name="default_topology_vol")
def fixture_default_topology_vol():
    """Provide default volume topology."""
    data = [
        "NODE    1 DVOL 1",
        "NODE    2 DVOL 1",
        "NODE    4 DVOL 1",
        "NODE    9 DVOL 1",
        "NODE    13 DVOL 1",
    ]
    return data


# ----------------- actual unit_tests -------------------------------------------------------------
def test_init(mocker, tmp_path):
    """Test the initialization of FourcDat."""
    path_to_dat_file = "dummy_path"
    list_geometric_sets = ["DSURFACE 9"]
    list_associated_material_numbers = [[10, 11]]
    element_topology = [{"element_number": [], "nodes": [], "material": []}]
    node_topology = [{"node_mesh": [], "node_topology": [], "topology_name": ""}]
    line_topology = [{"node_mesh": [], "line_topology": [], "topology_name": ""}]
    surface_topology = [{"node_mesh": [], "surface_topology": [], "topology_name": ""}]
    volume_topology = [{"node_mesh": [], "volume_topology": [], "topology_name": ""}]
    node_coordinates = {"node_mesh": [], "coordinates": []}
    mp = mocker.patch("queens.external_geometries.external_geometry.ExternalGeometry.__init__")

    path_to_preprocessed_dat_file = tmp_path / "preprocessed"
    random_fields = (
        [{"name": "mat_param", "type": "material", "external_instance": "DSURFACE 1"}],
    )

    geo_obj = FourcDat(
        input_template=path_to_dat_file,
        input_template_preprocessed=path_to_preprocessed_dat_file,
        list_geometric_sets=list_geometric_sets,
        associated_material_numbers_geometric_set=list_associated_material_numbers,
        random_fields=random_fields,
    )
    mp.assert_called_once()
    assert geo_obj.path_to_dat_file == path_to_dat_file
    assert geo_obj.list_geometric_sets == list_geometric_sets
    assert geo_obj.current_dat_section is None
    assert geo_obj.current_dat_section is None
    assert geo_obj.node_topology == node_topology
    assert geo_obj.surface_topology == surface_topology
    assert geo_obj.volume_topology == volume_topology
    assert geo_obj.desired_dat_sections == {"DNODE-NODE TOPOLOGY": []}
    assert geo_obj.nodes_of_interest is None
    assert geo_obj.node_coordinates == node_coordinates
    assert geo_obj.path_to_preprocessed_dat_file == path_to_preprocessed_dat_file
    assert geo_obj.line_topology == line_topology
    assert geo_obj.element_topology == element_topology
    assert geo_obj.random_fields == random_fields


def test_read_external_data_comment(mocker, tmp_path, dat_dummy_comment, default_geo_obj):
    """Test reading data with comments from dat file."""
    filepath = tmp_path / "myfile.dat"
    filepath.write_text(dat_dummy_comment, encoding="utf-8")

    mocker.patch(
        "queens.external_geometries.fourc_dat.FourcDat.get_current_dat_section",
        return_value=False,
    )

    mocker.patch(
        "queens.external_geometries.fourc_dat.FourcDat.get_only_desired_topology",
        return_value=1,
    )

    mocker.patch(
        "queens.external_geometries.fourc_dat.FourcDat.get_only_desired_coordinates",
        return_value=1,
    )

    mocker.patch(
        "queens.external_geometries.fourc_dat.FourcDat.get_materials",
        return_value=1,
    )

    mocker.patch(
        "queens.external_geometries.fourc_dat.FourcDat"
        ".get_elements_belonging_to_desired_material",
        return_value=1,
    )

    default_geo_obj.read_geometry_from_dat_file()

    assert default_geo_obj.get_current_dat_section.call_count == 2
    assert default_geo_obj.get_only_desired_topology.call_count == 0
    assert default_geo_obj.get_only_desired_coordinates.call_count == 0
    assert default_geo_obj.get_materials.call_count == 0
    assert default_geo_obj.get_elements_belonging_to_desired_material.call_count == 0


def test_read_external_data_get_functions(mocker, tmp_path, dat_dummy_get_fun, default_geo_obj):
    """Test reading data with get functions from dat file."""
    filepath = tmp_path / "myfile.dat"
    filepath.write_text(dat_dummy_get_fun)

    default_geo_obj.current_dat_section = "dummy"

    mocker.patch(
        "queens.external_geometries.fourc_dat.FourcDat.get_current_dat_section",
        return_value=False,
    )

    mocker.patch(
        "queens.external_geometries.fourc_dat.FourcDat.get_only_desired_topology",
        return_value=1,
    )

    mocker.patch(
        "queens.external_geometries.fourc_dat.FourcDat.get_only_desired_coordinates",
        return_value=1,
    )

    mocker.patch(
        "queens.external_geometries.fourc_dat.FourcDat.get_materials",
        return_value=1,
    )

    mocker.patch(
        "queens.external_geometries.fourc_dat.FourcDat"
        ".get_elements_belonging_to_desired_material",
        return_value=1,
    )

    default_geo_obj.read_geometry_from_dat_file()

    assert default_geo_obj.get_current_dat_section.call_count == 4
    assert default_geo_obj.get_only_desired_topology.call_count == 4
    assert default_geo_obj.get_only_desired_coordinates.call_count == 4
    assert default_geo_obj.get_materials.call_count == 4
    assert default_geo_obj.get_elements_belonging_to_desired_material.call_count == 4


def test_organize_sections(default_geo_obj):
    """Test organization of desired sections."""
    desired_geo_sets = ["DSURFACE 9", "DVOL 2", "DLINE 1", "DSURFACE 8"]
    expected_dat_section = {
        "DLINE-NODE TOPOLOGY": ["DLINE 1"],
        "DSURF-NODE TOPOLOGY": ["DSURFACE 9", "DSURFACE 8"],
        "DVOL-NODE TOPOLOGY": ["DVOL 2"],
        "DNODE-NODE TOPOLOGY": [],
    }
    default_geo_obj.list_geometric_sets = desired_geo_sets
    default_geo_obj.get_desired_dat_sections()

    assert default_geo_obj.desired_dat_sections == expected_dat_section


def test_get_current_dat_section_true(default_geo_obj, dat_section_true):
    """Test getting current dat section with a valid section."""
    default_geo_obj.get_current_dat_section(dat_section_true)
    clean_section_name = dat_section_true.strip()
    clean_section_name = clean_section_name.strip("-")
    assert default_geo_obj.current_dat_section == clean_section_name


def test_get_current_dat_section_false(default_geo_obj, dat_section_false):
    """Test getting current dat section with an invalid section."""
    default_geo_obj.get_current_dat_section(dat_section_false)
    assert default_geo_obj.current_dat_section is None


def test_check_if_in_desired_dat_section(default_geo_obj):
    """Test checking if in desired dat section."""
    default_geo_obj.desired_dat_sections = {
        "DLINE-NODE TOPOLOGY": ["DLINE 1"],
        "DSURF-NODE TOPOLOGY": ["DSURFACE 9", "DSURFACE 8"],
    }

    # return true
    default_geo_obj.current_dat_section = "DSURF-NODE TOPOLOGY"
    return_value = default_geo_obj.check_if_in_desired_dat_section()
    assert return_value

    # return false
    default_geo_obj.current_dat_section = "DVOL-NODE TOPOLOGY"
    return_value = default_geo_obj.check_if_in_desired_dat_section()
    assert not return_value


def test_get_topology(
    default_geo_obj,
    current_dat_sections,
    desired_sections,
    default_topology_node,
    default_topology_line,
    default_topology_surf,
    default_topology_vol,
):
    """Test getting topology for different sections."""
    default_geo_obj.current_dat_section = current_dat_sections
    default_geo_obj.desired_dat_sections = desired_sections

    if current_dat_sections == "DNODE-NODE TOPOLOGY":
        for line in default_topology_node:
            default_geo_obj.get_topology(line)
        assert default_geo_obj.node_topology[0]["node_mesh"] == [1, 2, 4, 9, 13]
        assert default_geo_obj.node_topology[0]["node_topology"] == [1, 1, 1, 1, 1]

    elif current_dat_sections == "DLINE-NODE TOPOLOGY":
        for line in default_topology_line:
            default_geo_obj.get_topology(line)
        assert default_geo_obj.line_topology[0]["node_mesh"] == [1, 2, 4, 9, 13]
        assert default_geo_obj.line_topology[0]["line_topology"] == [1, 1, 1, 1, 1]

    elif current_dat_sections == "DSURF-NODE TOPOLOGY":
        for line in default_topology_surf:
            default_geo_obj.get_topology(line)
        assert default_geo_obj.surface_topology[0]["node_mesh"] == [1, 2, 4, 9, 13]
        assert default_geo_obj.surface_topology[0]["surface_topology"] == [1, 1, 1, 1, 1]

    elif current_dat_sections == "DVOL-NODE TOPOLOGY":
        for line in default_topology_vol:
            default_geo_obj.get_topology(line)
        assert default_geo_obj.volume_topology[0]["node_mesh"] == [1, 2, 4, 9, 13]
        assert default_geo_obj.volume_topology[0]["volume_topology"] == [1, 1, 1, 1, 1]


def test_get_only_desired_topology(default_geo_obj, mocker):
    """Test getting only desired topology."""
    line = "dummy"
    mocker.patch(
        "queens.external_geometries.fourc_dat.FourcDat.check_if_in_desired_dat_section",
        return_value=True,
    )
    mp1 = mocker.patch(
        "queens.external_geometries.fourc_dat.FourcDat.get_topology",
        return_value=None,
    )
    default_geo_obj.get_only_desired_topology(line)
    assert mp1.call_count == 1

    mocker.patch(
        "queens.external_geometries.fourc_dat.FourcDat.check_if_in_desired_dat_section",
        return_value=False,
    )
    # counter stays 1 as called before
    default_geo_obj.get_only_desired_topology(line)
    assert mp1.call_count == 1


def test_get_only_desired_coordinates(default_geo_obj, mocker):
    """Test getting only desired coordinates."""
    line = "dummy 2"
    mp1 = mocker.patch(
        "queens.external_geometries.fourc_dat.FourcDat.get_nodes_of_interest",
        return_value=[1, 2, 3, 4, 5],
    )
    mp2 = mocker.patch(
        "queens.external_geometries.fourc_dat.FourcDat"
        ".get_coordinates_of_desired_geometric_sets",
        return_value=None,
    )
    default_geo_obj.current_dat_section = "NODE COORDS"
    default_geo_obj.get_only_desired_coordinates(line)
    mp1.assert_called_once()

    nodes_of_interest = [1, 2, 3, 4, 5]
    default_geo_obj.nodes_of_interest = nodes_of_interest
    default_geo_obj.get_only_desired_coordinates(line)
    mp1.assert_called_once()
    mp2.assert_called_once()


def test_get_coordinates_of_desired_geometric_sets(default_geo_obj, default_coords):
    """Test getting coordinates of desired geometric sets."""
    for line in default_coords:
        node_list = line.split()
        default_geo_obj.get_coordinates_of_desired_geometric_sets(node_list)
    assert default_geo_obj.node_coordinates["node_mesh"] == [1, 2, 3, 4]
    np.testing.assert_array_equal(
        default_geo_obj.node_coordinates["coordinates"],
        np.array(
            [
                [-1.0000000000000000e00, -2.5000000000000000e-01, 0.0000000000000000e00],
                [-1.0000000000000000e00, -2.5000000000000000e-01, -3.7500000000000006e-02],
                [-1.0000000000000000e00, -2.1666666666666665e-01, -3.7500000000000006e-02],
                [-1.0000000000000000e00, -2.1666666666666662e-01, 0.0000000000000000e00],
            ]
        ),
    )


def test_get_nodes_of_interest(default_geo_obj):
    """Test getting nodes of interest."""
    default_geo_obj.node_topology[-1]["node_mesh"] = [1, 2, 4, 9, 13]
    default_geo_obj.node_topology[-1]["node_topology"] = [1, 1, 1, 1, 1]
    default_geo_obj.line_topology[-1]["node_mesh"] = [1, 2, 4, 9, 13]
    default_geo_obj.line_topology[-1]["line_topology"] = [1, 1, 1, 1, 1]
    default_geo_obj.surface_topology[-1]["node_mesh"] = [1, 2, 4, 9, 13]
    default_geo_obj.surface_topology[-1]["surface_topology"] = [1, 1, 1, 1, 1]
    default_geo_obj.volume_topology[-1]["node_mesh"] = [1, 2, 4, 9, 13]
    default_geo_obj.volume_topology[-1]["volume_topology"] = [1, 1, 1, 1, 1]

    default_geo_obj.get_nodes_of_interest()
    assert default_geo_obj.nodes_of_interest == [1, 2, 4, 9, 13]
