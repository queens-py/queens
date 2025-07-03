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
"""4C random material fields preprocessor."""
from pathlib import Path

import numpy as np
from fourcipp.fourc_input import FourCInput


def get_node_coordinates(fourc_input, node_ids):
    """Get node coordinates by id.

    Args:
        fourc_input (FourCInput): 4C input data
        node_ids (list): List of node ids

    Returns:
        np.ndarray: Nodes coordintes
    """
    n_ids = len(node_ids)
    coordinates = {i: None for i in node_ids}
    counter = 0
    for node in fourc_input["NODE COORDS"]:
        if node["id"] in node_ids:
            coordinates[node["id"]] = node["COORD"]
            counter += 1
        if counter == n_ids:
            break

    return np.array(list(coordinates.values()))


def extract_elements(fourc_input, elements_section, extracting_condition):
    """Extract desired elements.

    Currenlty the mean of the element nodes is used as representative location.

    Args:
        fourc_input (FourCInput): 4C input data
        elements_section (str): Section of where to look for the elments
        extracting_condition (callable): Function to select the desired element

    Returns:
        tuple: element ids and representative element location
    """
    element_ids = []
    representative_locations = []
    for element in fourc_input[elements_section]:
        if extracting_condition(element):
            nodes_coordinates = get_node_coordinates(fourc_input, element["cell"]["connectivity"])
            element_ids.append(element["id"])
            representative_locations.append(np.mean(nodes_coordinates, axis=0))

    representative_locations = np.array(representative_locations)
    return element_ids, representative_locations


def create_jinja_json_template(parameter_name, element_ids, keys, template_path):
    """Create jinja2 template for the material data.

    Args:
        parameter_name (str): Name of the parameter
        element_ids (list): List of element ids
        keys (list): List of parameter key
        template_path (pathlib.Path): Path to create the template
    """
    template_data = "{" + f'"{parameter_name}"' + ":{"
    for element_id, key in zip(element_ids, keys):
        template_data += f'\n"{element_id}": {{{{ {key} }}}},'
    template_data = template_data[:-1]
    template_data += "}}"
    Path(template_path).write_text(template_data)


def extract_by_material_id(material_id):
    """Factory for material id comparison.

    Args:
        material_id (int): Desired material id

    Returns:
        callable: function to check if an element has the correct id
    """

    def check_material_id(element):
        return element["data"]["MAT"] == material_id

    return check_material_id


def create_random_elemenentwise_material_field(
    path_to_template, elements_section, extracting_condition, parameter_name, material_template_path
):
    """Create random elementwise material field.

    Args:
        path_to_template (str or Pathlib.path): Path to 4C yaml template
        elements_section (str): Section of where to look for the elments
        extracting_condition (callable): Function to select the desired element
        parameter_name (str): Name of the parameter
        material_template_path (str or Pathlib.path): Path to generate the material template

    Returns:
        dict: with parameter names and element locations
    """
    fourc_input = FourCInput.from_4C_yaml(path_to_template)
    extract_elements(fourc_input, elements_section, extracting_condition)

    element_ids, element_locations = extract_elements(
        fourc_input, elements_section, extracting_condition
    )
    keys = [f"{parameter_name}_{i}" for i in element_ids]

    create_jinja_json_template(parameter_name, element_ids, keys, material_template_path)

    return {"keys": keys, "coords": element_locations}
