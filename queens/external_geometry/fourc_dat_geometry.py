#
# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (c) 2025, QUEENS contributors.
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
"""4C geometry handling."""

# pylint: disable=too-many-lines
import copy
import fileinput
import os
import re
import shutil

import numpy as np

from queens.external_geometry.external_geometry import ExternalGeometry
from queens.utils.logger_settings import log_init_args


class FourcDatExternalGeometry(ExternalGeometry):
    """Class to read in external geometries based on 4C dat files.

    Attributes:
        path_to_dat_file (str): Path to dat file from which the *external_geometry_obj* should be
                                    extracted.
        path_to_preprocessed_dat_file (str): Path to preprocessed dat file with added
                                             placeholders.
        coords_dict (str): Dictionary containing coordinates of the discretized random
                                      fields and corresponding placeholder names.
        list_geometric_sets (lst): List of geometric sets that should be extracted.
        current_dat_section (str): String that encodes the current section in the dat file as
                                   file is read line-wise.
        desired_dat_sections (dict): Dictionary that holds only desired dat-sections and
                                     geometric sets within these sections, so that we can skip
                                     undesired parts of the dat-file.
        nodes_of_interest (lst): List that contains all (mesh) nodes that are part of a desired
                                 geometric component.
        new_nodes_lst (lst): List of new nodes that should be written in dnode topology.
        node_topology (lst): List with topology dicts of edges/nodes (here: mesh nodes not FEM
                             nodes) for each geometric set of this category.
        line_topology (lst): List with topology dicts of line components for each geometric set
                             of this category.
        surface_topology (lst): List of topology dicts of surfaces for each geometric set of this
                                category.
        volume_topology (lst): List of topology of volumes for each geometric set of this category.
        node_coordinates (dict): Dictionary that holds the desired nodes as well as their
                                 corresponding geometric coordinates.
        element_centers (np.array): Array with center coordinates of elements.
        element_topology (lst): List of dictionaries containing element topology, including node
                                mapping, materials, and element numbers.
        original_materials_in_dat (lst): List of original material numbers in dat template file.
        list_associated_material_numbers (lst): List of associated material numbers w.r.t. the
                                                geometric sets of interest.
        new_material_numbers (lst): List of new material numbers to be used.
        random_dirich_flag (bool): Flag to check if a random Dirichlet BC exists.
        random_transport_dirich_flag (bool): Flag to check if a random transport Dirichlet
                                             BC exists.
        random_neumann_flag (bool): Flag to check if a random Neumann BC exists.
        nodes_written (bool): Flag to check whether nodes have already been written.
        random_fields (lst): List of random field descriptions.

    Returns:
        geometry_obj (obj): Instance of FourcDatExternalGeometry class
    """

    dat_sections = [
        "DESIGN DESCRIPTION",
        "DESIGN POINT DIRICH CONDITIONS",
        "DESIGN POINT TRANSPORT DIRICH CONDITIONS",
        "DNODE-NODE TOPOLOGY",
        "DLINE-NODE TOPOLOGY",
        "DSURF-NODE TOPOLOGY",
        "DVOL-NODE TOPOLOGY",
        "NODE COORDS",
        "STRUCTURE ELEMENTS",
        "ALE ELEMENTS",
        "FLUID ELEMENTS",
        "LUBRIFICATION ELEMENTS",
        "TRANSPORT ELEMENTS",
        "TRANSPORT2 ELEMENTS",
        "THERMO ELEMENTS",
        "CELL ELEMENTS",
        "CELLSCATRA ELEMENTS",
        "ELECTROMAGNETIC ELEMENTS",
        "ARTERY ELEMENTS",
        "MATERIALS",
    ]
    section_match_dict = {
        "DNODE": "DNODE-NODE TOPOLOGY",
        "DLINE": "DLINE-NODE TOPOLOGY",
        "DSURFACE": "DSURF-NODE TOPOLOGY",
        "DVOL": "DVOL-NODE TOPOLOGY",
    }

    @log_init_args
    def __init__(
        self,
        input_template,
        input_template_preprocessed=None,
        list_geometric_sets=None,
        associated_material_numbers_geometric_set=None,
        random_fields=None,
    ):
        """Initialize 4C external geometry.

        Args:
            input_template (str): Path to dat file from which the external_geometry_obj should be
                                  extracted
            input_template_preprocessed (str): Path to preprocessed dat file with added
                                               placeholders
            list_geometric_sets (list): List of geometric sets that should be extracted
            associated_material_numbers_geometric_set (lst): List of associated material numbers wrt
                                                             to the geometric sets of interest
            random_fields (lst): List of random field descriptions
        """
        super().__init__()
        # settings / inputs
        self.path_to_dat_file = input_template
        self.path_to_preprocessed_dat_file = input_template_preprocessed
        self.coords_dict = {}
        self.list_geometric_sets = list_geometric_sets
        self.current_dat_section = None
        self.desired_dat_sections = {"DNODE-NODE TOPOLOGY": []}
        self.nodes_of_interest = None
        self.new_nodes_lst = None

        # topology
        self.node_topology = [{"node_mesh": [], "node_topology": [], "topology_name": ""}]
        self.line_topology = [{"node_mesh": [], "line_topology": [], "topology_name": ""}]
        self.surface_topology = [{"node_mesh": [], "surface_topology": [], "topology_name": ""}]
        self.volume_topology = [{"node_mesh": [], "volume_topology": [], "topology_name": ""}]
        self.node_coordinates = {"node_mesh": [], "coordinates": []}
        self.nodeset_names = set()

        # material specific attributes
        self.element_centers = []
        self.element_topology = [{"element_number": [], "nodes": [], "material": []}]
        self.original_materials_in_dat = []
        self.list_associated_material_numbers = associated_material_numbers_geometric_set
        self.new_material_numbers = []

        # some flags to memorize which sections have been written / manipulated
        self.random_dirich_flag = False
        self.random_transport_dirich_flag = False
        self.random_neumann_flag = False
        self.nodes_written = False
        self.random_fields = random_fields

    # --------------- child methods that must be implemented --------------------------------------
    def read_external_data(self):
        """Read the external input file with geometric data."""
        self.read_geometry_from_dat_file()

    def organize_sections(self):
        """Organize the sections of the external *external_geometry_obj*."""
        self.get_desired_dat_sections()

    def finish_and_clean(self):
        """Finish the analysis for the *external_geometry_obj* extraction."""
        self._sort_node_coordinates()
        self._get_element_centers()

    # -------------- helper methods ---------------------------------------------------------------
    def read_geometry_from_dat_file(self):
        """Read the dat-file line by line to be memory efficient."""
        with open(self.path_to_dat_file, encoding="utf-8") as my_dat:
            # read dat file line-wise
            for line in my_dat:
                line = line.strip()

                match_bool = self.get_current_dat_section(line)
                # skip comments outside of section definition
                if (
                    line[0:2] == "//"
                    or match_bool
                    or line == ""
                    or line.isspace()
                    or self.current_dat_section is None
                ):
                    pass
                else:
                    self.get_only_desired_topology(line)
                    self.get_only_desired_coordinates(line)
                    self.get_materials(line)
                    self.get_elements_belonging_to_desired_material(line)

    def _get_element_centers(self):
        """Calculate the geometric center of each finite element."""
        element_centers = []
        # TODO atm we only take first element of the list and dont loop over several random fields # pylint: disable=fixme
        for element_node_lst in self.element_topology[0]["nodes"]:
            element_center = np.array([0.0, 0.0, 0.0])
            for node in element_node_lst:
                element_center += np.array(
                    self.node_coordinates["coordinates"][
                        self.node_coordinates["node_mesh"].index(int(node))
                    ]
                )
            element_centers.append(element_center / float(len(element_node_lst)))
        self.element_centers = np.array(element_centers)

    def get_current_dat_section(self, line):
        """Check if the current line starts a new section in the dat-file.

        Update self.current_dat_section if new section was found. If the
        current line is the actual section identifier, return a True boolean.

        Args:
            line (str): Current line of the dat-file

        Returns:
            bool (boolean): True or False depending if current line is the section match
        """
        # regex for the sections
        section_name_re = re.compile("^-+([^-].+)$")
        match = section_name_re.match(line)
        # get the current section of the dat file
        if match:
            # remove whitespaces and horizontal line
            section_string = line.strip("-")
            section_string = section_string.strip()
            # check for comments
            if line[:2] == "//":
                return True
            # ignore comment pattern after actual string
            if section_string.strip("//") in self.dat_sections:
                self.current_dat_section = section_string
                return True
            self.current_dat_section = None
            return True
        return False

    def check_if_in_desired_dat_section(self):
        """Check if the dat-section contains the desired geometric set.

        Returns:
            bool: True if the case is found, False otherwise.
        """
        return self.current_dat_section in self.desired_dat_sections

    def get_desired_dat_sections(self):
        """Get the dat-sections that contain the desired geometric sets."""
        # initialize keys with empty lists
        for geo_set in self.list_geometric_sets:
            # TODO for now we read in all element_topology; this should be changed such that we only # pylint: disable=fixme
            #  store element_topology that belong to the desired geometric set (difficult though as
            # we would need to look-up the attached nodes and check their set assignment
            if geo_set.split()[1] == "ELEMENTS":
                self.desired_dat_sections[geo_set] = []
            else:
                self.desired_dat_sections[self.section_match_dict[geo_set.split()[0]]] = []

        # write desired geometric set to corresponding section key
        for geo_set in self.list_geometric_sets:
            if geo_set.split()[1] == "ELEMENTS":
                self.desired_dat_sections[geo_set].append(geo_set)
            else:
                self.desired_dat_sections[self.section_match_dict[geo_set.split()[0]]].append(
                    geo_set
                )

    def get_elements_belonging_to_desired_material(self, line):
        """Get finite element_topology that belongs to the material definition.

        Note that we assume that the geometric set of interest also has its own material name. This
        speeds-up the element-topology mapping significantly as we do not have to find which
        element belongs to which geometric set by performing a large node membership search per
        element and for all nodes of the element. On the other hand this assumption requires that
        the analyst provides a dat-file in the correct format.

        Args:
            line (str): Current line in the dat-file
        """
        # Note that the latter applies to all element sections!
        if "ELEMENTS" in self.current_dat_section and self.list_associated_material_numbers:
            material_number = int(line.split("MAT")[1].split()[0])
            # TODO atm we only can handle one material--> this should be changed to a list of # pylint: disable=fixme
            #  original_materials_in_dat
            if material_number == self.list_associated_material_numbers[0][0]:
                # get the element number
                self.element_topology[0]["element_number"].append(int(line.split()[0]))

                # get the nodes per element
                helper_list = line.split("MAT")
                nodes_list_str = helper_list[0].split()[3:]
                nodes = [int(node) for node in nodes_list_str]

                # below appends a list in a list of lists such that we keep the nodes per element
                self.element_topology[0]["nodes"].append(nodes)
                # the first number in the list is the main material the subsequent number the
                # nested material
                self.element_topology[0]["material"].append(material_number)

    def get_materials(self, line):
        """Get the different material definitions from the dat file.

        Args:
            line (str): Current line of the dat file
        """
        if self.current_dat_section == "MATERIALS":
            self.original_materials_in_dat.append(int(line.split()[1]))
        # pylint: disable=too-many-branches

    def get_topology(self, line):
        """Get the geometric topology by extracting and grouping (mesh) nodes.

        Only nodes that belong to the desired geometric sets and save them to their
        topology class.

        Args:
            line (str): Current line of the dat-file
        """
        topology_name = " ".join(line.split()[2:4])
        node_list = line.split()
        # get edges

        if self.current_dat_section == "DNODE-NODE TOPOLOGY":
            self.nodeset_names.add(int(line.split("DNODE ")[-1]))
            if topology_name in self.desired_dat_sections["DNODE-NODE TOPOLOGY"]:
                if (self.node_topology[-1]["topology_name"] == "") or (
                    self.node_topology[-1]["topology_name"] == topology_name
                ):
                    self.node_topology[-1]["node_mesh"].append(int(node_list[1]))
                    self.node_topology[-1]["node_topology"].append(int(node_list[3]))
                    self.node_topology[-1]["topology_name"] = topology_name
                else:
                    new_node_topology_dict = {
                        "node_mesh": int(node_list[1]),
                        "node_topology": int(node_list[3]),
                        "topology_name": topology_name,
                    }
                    self.node_topology.extend(new_node_topology_dict)

        # get lines
        elif self.current_dat_section == "DLINE-NODE TOPOLOGY":
            if topology_name in self.desired_dat_sections["DLINE-NODE TOPOLOGY"]:
                if (self.line_topology[-1]["topology_name"] == "") or (
                    self.line_topology[-1]["topology_name"] == topology_name
                ):
                    self.line_topology[-1]["node_mesh"].append(int(node_list[1]))
                    self.line_topology[-1]["line_topology"].append(int(node_list[3]))
                    self.line_topology[-1]["topology_name"] = topology_name
                else:
                    new_line_topology_dict = {
                        "node_mesh": int(node_list[1]),
                        "line_topology": int(node_list[3]),
                        "topology_name": topology_name,
                    }
                    self.line_topology.extend(new_line_topology_dict)

        # get surfaces
        elif self.current_dat_section == "DSURF-NODE TOPOLOGY":
            if topology_name in self.desired_dat_sections["DSURF-NODE TOPOLOGY"]:
                # append points to last list entry if geometric set is the name
                if (self.surface_topology[-1]["topology_name"] == "") or (
                    self.surface_topology[-1]["topology_name"] == topology_name
                ):
                    self.surface_topology[-1]["node_mesh"].append(int(node_list[1]))
                    self.surface_topology[-1]["surface_topology"].append(int(node_list[3]))
                    self.surface_topology[-1]["topology_name"] = topology_name

                # extend list with new geometric set
                else:
                    new_surf_topology_dict = {
                        "node_mesh": int(node_list[1]),
                        "surface_topology": int(node_list[3]),
                        "topology_name": topology_name,
                    }
                    self.surface_topology.extend(new_surf_topology_dict)

        # get volumes
        elif self.current_dat_section == "DVOL-NODE TOPOLOGY":
            if topology_name in self.desired_dat_sections["DVOL-NODE TOPOLOGY"]:
                # append points to last list entry if geometric set is the name
                if (self.volume_topology[-1]["topology_name"] == "") or (
                    self.volume_topology[-1]["topology_name"] == topology_name
                ):
                    self.volume_topology[-1]["node_mesh"].append(int(node_list[1]))
                    self.volume_topology[-1]["volume_topology"].append(int(node_list[3]))
                    self.volume_topology[-1]["topology_name"] = topology_name
                else:
                    new_volume_topology_dict = {
                        "node_mesh": int(node_list[1]),
                        "surface_topology": int(node_list[3]),
                        "topology_name": topology_name,
                    }
                    self.volume_topology.extend(new_volume_topology_dict)

    def get_only_desired_topology(self, line):
        """Check if the current line contains desired geometric sets.

        Skip the topology extraction if the current section does not contain a desired geometric
        set, anyways.

        Args:
            line (str): Current line of the dat-file
        """
        # skip lines that are not part of a desired section
        desired_section_boolean = self.check_if_in_desired_dat_section()

        if not desired_section_boolean:
            pass
        else:
            # get topology groups
            self.get_topology(line)

    def get_only_desired_coordinates(self, line):
        """Get coordinates of nodes that belong to a desired geometric set.

        Args:
            line (str): Current line of the dat-file
        """
        if self.current_dat_section == "NODE COORDS":
            node_list = line.split()
            if self.nodes_of_interest is None:
                self.get_nodes_of_interest()

            if self.nodes_of_interest is not None:
                if int(node_list[1]) in self.nodes_of_interest:
                    self.get_coordinates_of_desired_geometric_sets(node_list)

    def get_coordinates_of_desired_geometric_sets(self, node_list):
        """Extract node and coordinate information of the current line.

        Args:
            node_list (list): Current line of the dat-file
        """
        self.node_coordinates["node_mesh"].append(int(node_list[1]))
        nodes_as_float_list = [float(value) for value in node_list[3:6]]
        self.node_coordinates["coordinates"].append(nodes_as_float_list)

    def get_nodes_of_interest(self):
        """From the extracted topology, get a unique list of all nodes."""
        node_mesh_nodes = []
        for node_topo in self.node_topology:
            node_mesh_nodes.extend(node_topo["node_mesh"])

        line_mesh_nodes = []
        for line_topo in self.line_topology:
            line_mesh_nodes.extend(line_topo["node_mesh"])

        surf_mesh_nodes = []
        for surf_topo in self.surface_topology:
            surf_mesh_nodes.extend(surf_topo["node_mesh"])

        vol_mesh_nodes = []
        for vol_topo in self.volume_topology:
            vol_mesh_nodes.extend(vol_topo["node_mesh"])

        element_material_nodes = []
        for element_topo in self.element_topology:
            element_material_nodes.extend(element_topo["nodes"])

        nodes_of_interest = (
            node_mesh_nodes
            + line_mesh_nodes
            + surf_mesh_nodes
            + vol_mesh_nodes
            + element_material_nodes
        )

        # make node_list unique
        self.nodes_of_interest = list(set(nodes_of_interest))

    def _sort_node_coordinates(self):
        """Sort node coordinates based on the node mesh."""
        self.node_coordinates["coordinates"] = [
            coord
            for _, coord in sorted(
                zip(self.node_coordinates["node_mesh"], self.node_coordinates["coordinates"]),
                key=lambda pair: pair[0],
            )
        ]
        self.node_coordinates["node_mesh"] = sorted(self.node_coordinates["node_mesh"])

    # -------------- write random fields to dat file ----------------------------------------------
    # pylint: disable=too-many-branches
    def write_random_fields_to_dat(self):
        """Write placeholders for random fields to the dat file."""
        # copy the dat file and rename it for the current simulation
        shutil.copy2(self.path_to_dat_file, self.path_to_preprocessed_dat_file)

        # this has to be done outside of the file read as order is not known a priori
        self._create_new_node_sets(self.random_fields)

        # potentially organize new material definitions
        self._organize_new_material_definitions()

        # save original file ownership details
        stat = os.stat(self.path_to_dat_file)
        uid, gid = stat[4], stat[5]

        # random_fields_lst is here a list containing a dict description per random field
        with fileinput.input(
            self.path_to_preprocessed_dat_file, inplace=True, backup=".bak"
        ) as my_dat:
            # read dat file line-wise
            for line in my_dat:
                old_line = line
                line = line.strip()

                match_bool = self.get_current_dat_section(line)
                # skip comments outside of section definition
                if line[0:2] == "//" or match_bool or line.isspace() or line == "":
                    print(old_line, end="")
                else:
                    # check if in sec. DNODE-NODE topology and if so adjust this section in case
                    # of random BCs; write this only once
                    if self.current_dat_section == "DNODE-NODE TOPOLOGY" and not self.nodes_written:
                        self._write_new_node_sets()
                        self.nodes_written = True
                        # print the current line that was overwritten
                        print(old_line, end="")

                    # check if in sec. for random dirichlet cond. and if point dirich exists extend
                    elif (
                        self.current_dat_section == "DESIGN POINT DIRICH CONDITIONS"
                        and not self.random_dirich_flag
                    ):
                        self._write_design_point_dirichlet_conditions(self.random_fields, line)
                        self.random_dirich_flag = True

                    elif (
                        self.current_dat_section == "DESIGN POINT TRANSPORT DIRICH CONDITIONS"
                        and not self.random_transport_dirich_flag
                    ):
                        self._write_design_point_dirichlet_transport_conditions()
                        self.random_transport_dirich_flag = True

                    elif (
                        self.current_dat_section == "DESIGN POINT NEUM"
                        and not self.random_neumann_flag
                    ):
                        self._write_design_point_neumann_conditions()
                        self.random_neumann_flag = True

                    # materials and elements / constitutive random fields -----------------------
                    elif self.current_dat_section == "STRUCTURE ELEMENTS":
                        self._assign_elementwise_material_to_structure_ele(line)

                    elif self.current_dat_section == "FLUID ELEMENTS":
                        raise NotImplementedError()

                    elif self.current_dat_section == "ALE ELEMENTS":
                        raise NotImplementedError()

                    elif self.current_dat_section == "TRANSPORT ELEMENTS":
                        raise NotImplementedError()

                    elif (
                        self.current_dat_section == "MATERIALS"
                        and self.list_associated_material_numbers
                    ):
                        self._write_elementwise_materials(line, self.random_fields)

                    # If end of dat file is reached but certain sections did not exist so far,
                    # write them now
                    elif self.current_dat_section == "END":
                        bcs_list = [random_field["type"] for random_field in self.random_fields]
                        if ("dirichlet" in bcs_list) and (self.random_dirich_flag is False):
                            print(
                                "----------------------------------------------DESIGN POINT "
                                "DIRICH CONDITIONS\n"
                            )
                            self._write_design_point_dirichlet_conditions(self.random_fields, line)

                        elif ("transport_dirichlet" in bcs_list) and (
                            self.random_transport_dirich_flag is False
                        ):
                            print(
                                "----------------------------------------------DESIGN POINT "
                                "TRANSPORT DIRICH CONDITIONS\n"
                            )
                            self._write_design_point_dirichlet_transport_conditions()

                        elif ("neumann" in bcs_list) and (self.random_neumann_flag is False):
                            print(
                                "---------------------------------------------DESIGN POINT "
                                "NEUMANN CONDITIONS\n"
                            )
                            self._write_design_point_neumann_conditions()
                        print(
                            "---------------------------------------------------------------------"
                            "----END\n"
                        )
                    else:
                        print(old_line, end="")

        os.chown(self.path_to_preprocessed_dat_file, uid, gid)
        # pylint: enable=too-many-branches

    # ------ write random material fields -----------------------------------------------
    def _organize_new_material_definitions(self):
        """Organize  and generate new material definitions.

        The new material naming continues from the maximum material
        number that was found in the dat file.
        """
        # Material definitions -> these are the mat numbers
        # note that this is a list of lists
        materials_copy = copy.copy(self.original_materials_in_dat)

        # remove materials of interest and nested dependencies from material copy
        # note we remove here the entire sublist that belongs to a geometric set of interest
        if self.list_associated_material_numbers:
            for associated_materials in self.list_associated_material_numbers:
                for material in associated_materials:
                    materials_copy.remove(material)

            # set number of new materials equal to length of element_topology of interest
            # TODO atm we just take the first list entry here and dont loop over several fields # pylint: disable=fixme
            n_mats = len(self.element_topology[0]["element_number"])

            # Check the largest material definition to avoid overwriting existing original_materials
            material_numbers_flat = [item for sublist in materials_copy for item in sublist]
            if material_numbers_flat:
                max_material_number = max(material_numbers_flat)
            else:
                max_material_number = int(0)

            # TODO at the moment we only take the first given materials (list in list of lists) # pylint: disable=fixme
            #  and cannot handle several material definitions # pylint: disable=fixme
            if len(self.list_associated_material_numbers[0]) == 1:
                # below is also a list in a list such that we cover the general case of base and
                # nested material even if in this if branch only the base material is of interest
                self.new_material_numbers = [
                    list((max_material_number + 1 + np.arange(0, n_mats, 1)).astype(int))
                ]
            elif len(self.list_associated_material_numbers[0]) == 2:
                self.new_material_numbers = [
                    list((max_material_number + 1 + np.arange(0, n_mats, 1)).astype(int))
                ]

                list_nested = list(
                    (self.new_material_numbers[0][-1] + 1 + np.arange(0, n_mats, 1)).astype(int)
                )

                # note this is a list of two lists containing the numbers of the base material
                # and the nested material separately
                self.new_material_numbers.append(list_nested)

            else:
                raise RuntimeError(
                    "At the moment we can only handle one nested material but you "
                    "provided {len(materials_copy[0])} material nestings. Abort..."
                )

    def _assign_elementwise_material_to_structure_ele(self, line):
        """Assign and write the new material definitions.

        One per element in desired geometric set to the associated
        structure element under the dat section `STRUCTURE ELEMENTS`
        """
        # TODO below we take the first list entry and dont loop for now over lists of list. # pylint: disable=fixme
        # The first material in the sublist is assumed to be the base material, the following
        # numbers are associated to nested materials. Hence we take the first number to write the
        # element
        if self.list_associated_material_numbers:
            material_expression = "MAT " + str(self.list_associated_material_numbers[0][0])
            if material_expression in line:
                # Current element number
                current_element_number = int(line.split()[0])

                # TODO atm we just take first list entry below and dont loop over all element # pylint: disable=fixme
                # topologies get index of current element within element_topology
                element_idx = self.element_topology[0]["element_number"].index(
                    current_element_number
                )

                # TODO note that new materials is atm only one list not a list of lists as iteration # pylint: disable=fixme
                #  over several topologies is not implemented so far # pylint: disable=fixme
                # we use first list here as element depend normally on base material
                new_material_number = self.new_material_numbers[0][element_idx]

                # exchange material number for new number in structure element
                new_line = line.replace(material_expression, "MAT " + str(new_material_number))
                print(new_line)
        else:
            print(line)

    def _write_elementwise_materials(self, line, random_field_lst):
        """Write the new material definitions under `MATERIALS`.

        We have one new material per element that is part of the geometric set where the random
        field is defined on.

        Args:
            line (str): Current line in the dat-file
            random_field_lst (lst): List containing dictionaries with random field description
                                    and realizations per associated geometric set
        """
        current_material_number = int(line.split()[1])

        # pylint: disable=fixme
        # TODO see how to use the random field lst here
        # TODO but also only address first random_field for now
        # TODO maybe directly separate the random_field types as different attributes
        # pylint: enable=fixme
        # get random fields of type material
        material_fields = [field for field in random_field_lst if field["type"] == "material"]

        material_field_placeholders = [
            material_fields[0]["name"] + "_" + str(i) for i in range(len(self.element_centers))
        ]
        self._write_coords_to_dict(
            material_fields[0]["name"], material_field_placeholders, np.array(self.element_centers)
        )

        # check if the current material number is equal to base material and rewrite the base
        # materials as well as the potentially associated nested materials here
        # TODO atm we only focus on first sublist and do not iterate over several materials # pylint: disable=fixme
        if current_material_number == self.list_associated_material_numbers[0][0]:
            self._write_base_materials(current_material_number, line, material_fields)

        # in case the current material number is a nested material overwrite now the nested
        # material block. Now we are in the next line and the base material was already written
        # TODO note that we again assume only one topology (first sublist) # pylint: disable=fixme
        elif (current_material_number in self.list_associated_material_numbers[0]) and (
            current_material_number != self.list_associated_material_numbers[0][0]
        ):
            self._write_nested_materials(line, material_fields)

        else:
            print(line)

    def _write_base_materials(self, current_material_number, line, material_fields):
        """Write the base materials field elementwise.

        Also materials without nested dependencies.

        Args:
            current_material_number (int): Old/current material number
            line (str): Current line in the dat-file
            material_fields (lst): List of dictionaries containing descriptions of material fields
        """
        # Add new material definitions
        # TODO Note: new_material numbers is atm just a list for the first topology # pylint: disable=fixme
        for idx, material_num in enumerate(self.new_material_numbers[0]):
            old_material_expression = "MAT " + str(current_material_number)
            new_material_expression = "MAT " + str(material_num)

            # potentially replace material parameter
            #  TODO idx seems to be wrong here # pylint: disable=fixme
            line_new = FourcDatExternalGeometry._parse_material_value_dependent_on_element_center(
                line, idx, material_fields
            )

            # TODO check if associated material intends nested material # pylint: disable=fixme
            if len(self.list_associated_material_numbers[0]) == 1:
                # Update the old material line/definition
                new_line = line_new.replace(old_material_expression, new_material_expression)

                # print the new material definition
                print(new_line)

            elif len(self.list_associated_material_numbers[0]) == 2:
                old_base_mat_str = "MATIDS " + line_new.split("MATIDS")[1].split()[0]
                new_base_mat_str = "MATIDS " + str(self.new_material_numbers[1][idx])

                # Update the old material line/definition
                new_line = line_new.replace(old_material_expression, new_material_expression)
                new_line = new_line.replace(old_base_mat_str, new_base_mat_str)

                # print the new material definition
                print(new_line)

            else:
                raise RuntimeError(
                    "At the moment we can only handle one nested material but "
                    "you provided "
                    f"{len(self.list_associated_material_numbers[0])} nested"
                    " materials. Abort...."
                )

    def _write_nested_materials(self, line, material_fields):
        """Write the nested materials field elementwise.

        Args:
            line (str): Current line in the dat-file
            material_fields (lst): List of dictionaries containing descriptions of material fields
        """
        # below we loop over numbers of nested materials
        for idx, material_num in enumerate(self.new_material_numbers[1]):
            # potentially replace material parameter
            line_new = FourcDatExternalGeometry._parse_material_value_dependent_on_element_center(
                line, idx, material_fields
            )

            # correct and print the nested material
            # note we use every second element here as the first ones are the base materials
            new_line = line_new.replace(
                "MAT " + str(self.list_associated_material_numbers[0][1]),
                "MAT " + str(material_num),
            )

            print(new_line)

    @staticmethod
    def _parse_material_value_dependent_on_element_center(line, realization_index, material_fields):
        """Parse the material value.

        Only in case the line contains the material parameter name.

        Args:
            line (str): Current line of the dat-input file
            realization_index (int): Index of field value that corresponds to current element /
                                     line which should be changed
            material_fields (lst): List containing the random field descriptions and realizations

        Returns:
            line_new (str): New updated line that contains material value of field realization
        """
        # TODO atm we only assume one random material field (see [0]). This should be generalized! # pylint: disable=fixme
        mat_param_name = material_fields[0][
            "name"
        ]  # TODO see if name is correct here # pylint: disable=fixme
        # potentially replace material parameter
        line_new = line
        if mat_param_name in line:
            string_to_replace = "{ " + mat_param_name + " }"
            line_new = line.replace(
                string_to_replace, f"{{ {mat_param_name}_{realization_index} }}"
            )
            # TODO key field realization prob wrong # pylint: disable=fixme

        return line_new

    def _write_design_point_dirichlet_transport_conditions(self):
        pass

    def _write_design_point_neumann_conditions(self):
        pass

    def _write_design_point_dirichlet_conditions(self, random_field_lst, line):
        """Write Dirichlet design conditions.

        Convert the random fields, defined on the geometric set of interest, into design point
        Dirichlet BCs such that each dpoint contains a placeholder value of the random field.

        Args:
            random_field_lst (lst): List containing the design description of the
                                              involved random fields
            line (str): String for the current line in the dat-file that is read in
        """
        # random fields for these sets if Dirichlet
        for geometric_set in self.list_geometric_sets:
            fields_dirich_on_geo_set = [
                field
                for field in random_field_lst
                if (field["type"] == "dirichlet") and (field["external_instance"] == geometric_set)
            ]
            if fields_dirich_on_geo_set:
                old_num = FourcDatExternalGeometry._get_old_num_design_point_dirichlet_conditions(
                    line
                )
                self._overwrite_num_design_point_dirichlet_conditions(random_field_lst, old_num)
                # select correct node set
                node_set = [
                    node_set for node_set in self.new_nodes_lst if node_set["name"] == geometric_set
                ][0]

                # assign random dirichlet fields
                (
                    realized_random_field_1,
                    realized_random_field_2,
                    realized_random_field_3,
                    fun_1,
                    fun_2,
                    fun_3,
                ) = self._assign_random_dirichlet_fields_per_geo_set(fields_dirich_on_geo_set)

                # take care of remaining deterministic dofs on the geometric set
                # we take the first field to get deterministic dofs
                (
                    realized_random_field_1,
                    realized_random_field_2,
                    realized_random_field_3,
                    fun_1,
                    fun_2,
                    fun_3,
                ) = self._assign_deterministic_dirichlet_fields_per_geo_set(
                    fields_dirich_on_geo_set,
                    realized_random_field_1,
                    realized_random_field_2,
                    realized_random_field_3,
                    fun_1,
                    fun_2,
                    fun_3,
                )

                # write the new fields to the dat file --------------------------------------------
                for topo_node, random_field_1, random_field_2, random_field_3, f1, f2, f3 in zip(
                    node_set["topo_dnodes"],
                    realized_random_field_1,
                    realized_random_field_2,
                    realized_random_field_3,
                    fun_1,
                    fun_2,
                    fun_3,
                ):
                    print(
                        f"E {topo_node} - NUMDOF 3 ONOFF 1 1 1 VAL {random_field_1} "
                        f"{random_field_2} {random_field_3} FUNCT {int(f1)} {int(f2)} {int(f3)}"
                    )

            else:
                print(line)

    @staticmethod
    def _get_old_num_design_point_dirichlet_conditions(line):
        """Return the former number of dirichlet point conditions.

        Args:
            line (str): String of current dat-file line

        Returns:
            old_num (int): old number of dirichlet point conditions
        """
        old_num = int(line.split()[1])

        return old_num

    def _overwrite_num_design_point_dirichlet_conditions(self, random_field_lst, old_num):
        """Write the new number of design point dirichlet conditions.

        Write them to the design description.

        Args:
            random_field_lst (lst): List containing vectors with the values of the
                                              realized random fields
            old_num (int): Former number of design point Dirichlet conditions
        """
        # loop over all dirichlet nodes
        field_values = []
        for geometric_set in self.list_geometric_sets:
            field_values.extend(
                [
                    [
                        field["name"] + "_" + str(i)
                        for i in range(len(self.node_coordinates["node_mesh"]))
                    ]
                    for field in random_field_lst
                    if (field["type"] == "dirichlet")
                    and (field["external_instance"] == geometric_set)
                ]
            )

        if field_values:
            num_new_dpoints = len(field_values[0])
            num_existing_dpoints = old_num
            total_num_dpoints = num_new_dpoints + num_existing_dpoints
            print(f"DPOINT                          {total_num_dpoints}")

    def _assign_random_dirichlet_fields_per_geo_set(self, fields_dirich_on_geo_set):
        """Assign random Dirichlet fields.

        Args:
            fields_dirich_on_geo_set (lst): List containing the descriptions and a
                                            realization of the Dirichlet BCs random fields.

        Returns:
            realized_random_field_1 (np.array): Array containing values of the random field (each
                                                row is associated to a corresponding row in the
                                                coordinate matrix) for the depicted dimension of the
                                                Dirichlet DOF.
            realized_random_field_2 (np.array): Array containing values of the random field (each
                                                row is associated to a corresponding row in the
                                                coordinate matrix) for the depicted dimension of the
                                                Dirichlet DOF.
            realized_random_field_3 (np.array): Array containing values of the random field (each
                                                row is associated to a corresponding row in the
                                                coordinate matrix) for the depicted dimension of the
                                                Dirichlet DOF.
            fun_1 (np.array): Array containing integer identifiers for functions that are applied to
                              associated dimension of the random field. This is a 4C specific
                              function that might, e.g., vary in time.
            fun_2 (np.array): Array containing integer identifiers for functions that are applied to
                              associated dimension of the random field. This is a 4C specific
                              function that might, e.g., vary in time.
            fun_3 (np.array): Array containing integer identifiers for functions that are applied to
                              associated dimension of the random field. This is a 4C specific
                              function that might, e.g., vary in time.
        """
        # take care of random dirichlet fields
        realized_random_field_1 = realized_random_field_2 = realized_random_field_3 = None
        fun_1 = fun_2 = fun_3 = None
        for dirich_field in fields_dirich_on_geo_set:
            set_shape = len(self.node_coordinates["node_mesh"])
            placeholders = [dirich_field["name"] + "_" + str(i) for i in range(set_shape)]
            if dirich_field["dof_for_field"] == 1:
                realized_random_field_1 = [
                    "{{ " + placeholder + " }}" for placeholder in placeholders
                ]
                fun_1 = dirich_field["funct_for_field"] * np.ones(set_shape)

            elif dirich_field["dof_for_field"] == 2:
                realized_random_field_2 = [
                    "{{ " + placeholder + " }}" for placeholder in placeholders
                ]
                fun_2 = dirich_field["funct_for_field"] * np.ones(set_shape)

            elif dirich_field["dof_for_field"] == 3:
                realized_random_field_3 = [
                    "{{ " + placeholder + " }}" for placeholder in placeholders
                ]
                fun_3 = dirich_field["funct_for_field"] * np.ones(set_shape)

            self._write_coords_to_dict(
                dirich_field["name"], placeholders, np.array(self.node_coordinates["coordinates"])
            )

        return (
            realized_random_field_1,
            realized_random_field_2,
            realized_random_field_3,
            fun_1,
            fun_2,
            fun_3,
        )

    def _assign_deterministic_dirichlet_fields_per_geo_set(
        self,
        fields_dirich_on_geo_set,
        realized_random_field_1,
        realized_random_field_2,
        realized_random_field_3,
        fun_1,
        fun_2,
        fun_3,
    ):
        """Adopt and write the DOFs of the Dirichlet point conditions.

        These conditions did not exist before but only one DOF at each discrete point might be rawn
         form a random field; the other DOFs might be a constant value, e.g., 0.

        Args:
            fields_dirich_on_geo_set (lst): List containing the descriptions and a
                                            realization of the Dirichlet BCs random fields.
            realized_random_field_1 (np.array): Array containing values of the random field (each
                                                row is associated to a corresponding row in the
                                                coordinate matrix) for the depicted dimension of the
                                                Dirichlet DOF.
            realized_random_field_2 (np.array): Array containing values of the random field (each
                                                row is associated to a corresponding row in the
                                                coordinate matrix) for the depicted dimension of the
                                                Dirichlet DOF.
            realized_random_field_3 (np.array): Array containing values of the random field (each
                                                row is associated to a corresponding row in the
                                                coordinate matrix) for the depicted dimension of the
                                                Dirichlet DOF.
            fun_1 (np.array): Array containing integer identifiers for functions that are applied to
                              associated dimension of the random field. This is a fourc specific
                              function that might, e.g., vary in time.
            fun_2 (np.array): Array containing integer identifiers for functions that are applied to
                              associated dimension of the random field. This is a fourc specific
                              function that might, e.g., vary in time.
            fun_3 (np.array): Array containing integer identifiers for functions that are applied to
                              associated dimension of the random field. This is a fourc specific
                              function that might, e.g., vary in time.

        Returns:
            realized_random_field_1 (np.array): Array containing values of the random field (each
                                                row is associated to a corresponding row in the
                                                coordinate matrix) for the depicted dimension of the
                                                Dirichlet DOF.
            realized_random_field_2 (np.array): Array containing values of the random field (each
                                                row is associated to a corresponding row in the
                                                coordinate matrix) for the depicted dimension of the
                                                Dirichlet DOF.
            realized_random_field_3 (np.array): Array containing values of the random field (each
                                                row is associated to a corresponding row in the
                                                coordinate matrix) for the depicted dimension of the
                                                Dirichlet DOF.
            fun_1 (np.array): Array containing integer identifiers for functions that are applied to
                              associated dimension of the random field. This is a fourc specific
                              function that might, e.g., vary in time.
            fun_2 (np.array): Array containing integer identifiers for functions that are applied to
                              associated dimension of the random field. This is a fourc specific
                              function that might, e.g., vary in time.
            fun_3 (np.array): Array containing integer identifiers for functions that are applied to
                              associated dimension of the random field. This is a fourc specific
                              function that might, e.g., vary in time.
        """
        # TODO see how this behaves for several fields # pylint: disable=fixme
        set_shape = len(self.node_coordinates["node_mesh"])
        for deter_dof, value_deter_dof, funct_deter in zip(
            fields_dirich_on_geo_set[0]["dofs_deterministic"],
            fields_dirich_on_geo_set[0]["value_dofs_deterministic"],
            fields_dirich_on_geo_set[0]["funct_for_deterministic_dofs"],
        ):
            if deter_dof == 1:
                if realized_random_field_1 is None:
                    realized_random_field_1 = value_deter_dof * np.ones(set_shape)
                    fun_1 = funct_deter * np.ones(set_shape)
                else:
                    raise ValueError(
                        "Dof 1 of geometric set is already defined and cannot be set to a "
                        "deterministic value now. Abort..."
                    )
            elif deter_dof == 2:
                if realized_random_field_2 is None:
                    realized_random_field_2 = value_deter_dof * np.ones(set_shape)
                    fun_2 = funct_deter * np.ones(set_shape)
                else:
                    raise ValueError(
                        "Dof 2 of geometric set is already defined and cannot be set to a "
                        "deterministic value now. Abort..."
                    )
            elif deter_dof == 3:
                if realized_random_field_2 is None:
                    realized_random_field_2 = value_deter_dof * np.ones(set_shape)
                    fun_2 = funct_deter * np.ones(set_shape)
                else:
                    raise ValueError(
                        "Dof 3 of geometric set is already defined and cannot be set to a "
                        "deterministic value now. Abort..."
                    )

        # catch fields that were not set
        if fun_1 is None or fun_2 is None or fun_3 is None:
            raise ValueError("One fourc function of a Dirichlet DOF was not defined! Abort...")
        if (
            realized_random_field_1 is None
            or realized_random_field_2 is None
            or realized_random_field_3 is None
        ):
            raise ValueError(
                "One random fields realization for a Dirichlet BC was not defined. Abort..."
            )

        return (
            realized_random_field_1,
            realized_random_field_2,
            realized_random_field_3,
            fun_1,
            fun_2,
            fun_3,
        )

    def _get_my_topology(self, geo_set_name_type):
        """Get the topology of a geometric set.

        I.e.e its node mappings, based on the type of the geometric set.

        Args:
            geo_set_name_type (str): Name of the geometric set type.

        Returns:
            my_topology (lst): List with desired geometric topology
        """
        # TODO this is problematic as topology has not been read it for the new object # pylint: disable=fixme
        if geo_set_name_type == "DNODE":
            my_topology = self.node_topology

        elif geo_set_name_type == "DLINE":
            my_topology = self.line_topology

        elif geo_set_name_type == "DSURFACE":
            my_topology = self.surface_topology

        elif geo_set_name_type == "DVOL":
            my_topology = self.volume_topology
        else:
            my_topology = None
        return my_topology

    def _create_new_node_sets(self, random_fields_lst):
        """Create nodeset form geometric associated nodes.

        From a given geometric set of interest: Identify associated nodes
        and add them as a new node-set to the geometric set-description of the
        external external_geometry_obj.

        Args:
            random_fields_lst (lst): List containing descriptions of involved random fields
        """
        # iterate through all random fields that encode BCs
        boundary_conditions_random_fields = (
            random_field
            for random_field in random_fields_lst
            if (
                (random_field["type"] == "dirichlet")
                or (random_field["type"] == "neumann")
                or (random_field["type"] == "transport_dirichlet")
            )
        )
        nodes_mesh_lst = []  # note, this is a list of dicts
        for random_field in boundary_conditions_random_fields:
            # get associated geometric set
            topology_name = random_field["external_instance"]
            topology_type = topology_name.split()[0]
            # check if line, surf or vol
            my_topology_lst = self._get_my_topology(topology_type)
            nodes_mesh_lst.extend(
                [
                    {"node_mesh": topo["node_mesh"], "topo_dnodes": [], "name": topology_name}
                    for topo in my_topology_lst
                    if topo["topology_name"] == topology_name
                ]
            )

        ndnode_min = len(self.nodeset_names)

        for num, _ in enumerate(nodes_mesh_lst):
            if num == 0:
                nodes_mesh_lst[num]["topo_dnodes"] = np.arange(
                    ndnode_min, ndnode_min + len(nodes_mesh_lst[num]["node_mesh"])
                )
            else:
                last_node = nodes_mesh_lst[num - 1]["topo_dnodes"][-1]
                nodes_mesh_lst[num]["topo_dnodes"] = np.arange(
                    last_node, last_node + len(nodes_mesh_lst[num]["node_mesh"])
                )

        self.new_nodes_lst = nodes_mesh_lst

    def _write_new_node_sets(self):
        """Write the new node sets to the dat-file as individual dnodes."""
        for node_set in self.new_nodes_lst:
            for mesh_node, topo_node in zip(node_set["node_mesh"], node_set["topo_dnodes"]):
                print(f"NODE {mesh_node} DNODE {topo_node}")

    def _write_coords_to_dict(self, field_name, field_keys, field_coords):
        """Write random field coordinates to dict.

        Args:
            field_name (str): Name of the random field
            field_keys (str): Placeholders for the discretized random field
            field_coords (np.ndarray): Coordinates of the discretized random field
        """
        self.coords_dict[field_name] = {"keys": field_keys, "coords": field_coords}
