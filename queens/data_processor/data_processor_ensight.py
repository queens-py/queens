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
"""Data processor class for reading vtk-ensight data."""

import logging

import numpy as np
import vtk
from vtkmodules.util.numpy_support import numpy_to_vtk, vtk_to_numpy

from queens.data_processor.data_processor import DataProcessor
from queens.utils.logger_settings import log_init_args

_logger = logging.getLogger(__name__)


class DataProcessorEnsight(DataProcessor):
    """Class for data-processing ensight output.

    Attributes:
        experimental_data (dict): dict with experimental data.
        coordinates_label_experimental (lst): List of (spatial) coordinate labels
                                                of the experimental data set.
        time_label_experimental (str): Time label of the experimental data set.
        external_geometry (obj): QUEENS external geometry object.
        target_time_lst (lst): Target time list for the ensight data, meaning time for which the
                                simulation state should be extracted.
        time_tol (float): Tolerance for the target time extraction.
        vtk_field_label (str): Label defining which physical field should be extraced from
                               the vtk data.
        vtk_field_components (lst): List with vector components that should be extracted
                                    from the field.
        vtk_array_type (str): Type of vtk array (e.g. *point_array*).
        geometric_target (lst): List with information about specific geometric target in vtk data
                                (This might be dependent on the simulation software that generated
                                the vtk file).
        geometric_set_data (dict): Dictionary describing the topology of the geometry
    """

    @log_init_args
    def __init__(
        self,
        file_name_identifier=None,
        file_options_dict=None,
        files_to_be_deleted_regex_lst=None,
        external_geometry=None,
        experimental_data_reader=None,
    ):
        """Init method for ensight data processor object.

        Args:
            file_name_identifier (str): Identifier of file name.
                                             The file prefix can contain regex expression
                                             and subdirectories.
            file_options_dict (dict): Dictionary with read-in options for the file:
                - target_time_lst (lst): Target time list for the ensight data, meaning time for
                                         which the simulation state should be extracted
                - time_tol (float): Tolerance for the target time extraction
                - geometric_target (lst): List with information about specific geometric target in
                                          vtk data (This might be dependent on the simulation
                                          software that generated the vtk file.)
                - physical_field_dict (dict): Dictionary with options for physical field:
                    -- vtk_field_label (str): Label defining which physical field should be
                                              extracted from the vtk data
                    -- field_components (lst): List with vector components that should be extracted
                                        from the field
                    -- vtk_array_type (str): Type of vtk array (e.g., point_array)


            files_to_be_deleted_regex_lst (lst): List with paths to files that should be deleted.
                                                 The paths can contain regex expressions.
            external_geometry (obj): QUEENS external geometry object
            experimental_data_reader (obj): Experimental data reader object

        Returns:
            Instance of DataProcessorEnsight class (obj)
        """
        super().__init__(
            file_name_identifier=file_name_identifier,
            file_options_dict=file_options_dict,
            files_to_be_deleted_regex_lst=files_to_be_deleted_regex_lst,
        )

        self._check_field_specification_dict(file_options_dict)
        # load the necessary options from the file options dictionary
        target_time_lst = file_options_dict.get("target_time_lst")
        if not isinstance(target_time_lst, list):
            raise TypeError(
                "The option 'target_time_lst' in the data_processor settings must be of type 'list'"
                f" but you provided type {type(target_time_lst)}. Abort..."
            )
        time_tol = file_options_dict.get("time_tol")
        if time_tol:
            if not isinstance(time_tol, float):
                raise TypeError(
                    "The option 'time_tol' in the data_processor settings must be of type 'float' "
                    f"but you provided type {type(time_tol)}. Abort..."
                )

        vtk_field_label = file_options_dict["physical_field_dict"]["vtk_field_label"]
        if not isinstance(vtk_field_label, str):
            raise TypeError(
                "The option 'vtk_field_label' in the data_processor settings must be of type 'str' "
                f"but you provided type {type(vtk_field_label)}. Abort..."
            )
        vtk_field_components = file_options_dict["physical_field_dict"]["field_components"]
        if not isinstance(vtk_field_components, list):
            raise TypeError(
                "The option 'field_components' in the data_processor settings must be of type "
                f"'list' but you provided type {type(vtk_field_components)}. Abort..."
            )

        vtk_array_type = file_options_dict["physical_field_dict"]["vtk_array_type"]
        if not isinstance(vtk_array_type, str):
            raise TypeError(
                "The option 'vtk_array_type' in the data_processor settings must be of type 'str' "
                f"but you provided type {type(vtk_array_type)}. Abort..."
            )

        # geometric_target (lst): List specifying where (for which geometric target) vtk data
        #                         should be read-in. The first list element can have the entries:
        #                         - "geometric_set": Data is read-in from a specific geometric set
        #                         - "experimental_data": Data is read-in at experimental data
        #                                  coordinates (which must be part of the domain)
        geometric_target = file_options_dict["geometric_target"]
        if not isinstance(geometric_target, list):
            raise TypeError(
                "The option 'geometric_target' in the data_processor settings must be of type "
                f"'list' but you provided type {type(geometric_target)}. Abort..."
            )

        self.geometric_set_data = None

        if geometric_target[0] == "geometric_set":
            self.geometric_set_data = self.read_geometry_coordinates(external_geometry)

        if experimental_data_reader is not None:
            (
                _,
                _,
                _,
                self.experimental_data,
                self.time_label_experimental,
                self.coordinates_label_experimental,
            ) = experimental_data_reader.get_experimental_data()
        else:
            self.experimental_data = None
            self.time_label_experimental = None
            self.coordinates_label_experimental = None

        self.external_geometry = external_geometry
        self.target_time_lst = target_time_lst
        self.time_tol = time_tol
        self.vtk_field_label = vtk_field_label
        self.vtk_field_components = vtk_field_components
        self.vtk_array_type = vtk_array_type
        self.geometric_target = geometric_target

    @staticmethod
    def _check_field_specification_dict(file_options_dict):
        """Check the file_options_dict for valid inputs.

        Args:
            file_options_dict (dict): Dictionary containing the field description for the
                                      physical fields of interest that should be read-in.
        """
        required_keys_lst = ["target_time_lst", "physical_field_dict", "geometric_target"]
        if not set(required_keys_lst).issubset(set(file_options_dict.keys())):
            raise KeyError(
                "The option 'file_options_dict' within the data_processor section must at least "
                f"contain the following keys: {required_keys_lst}. "
                f"You only provided: {file_options_dict.keys()}. Abort..."
            )

        required_field_keys = [
            "vtk_array_type",
            "vtk_field_label",
            "vtk_array_type",
            "field_components",
        ]

        if not set(required_field_keys).issubset(
            set(file_options_dict["physical_field_dict"].keys())
        ):
            raise KeyError(
                "The option 'physical_field_dict' within the data_processor section must at least "
                f"contain the following keys: {required_field_keys}. You only provided the keys: "
                f"{file_options_dict['physical_field_dict'].keys()}"
            )

    def get_raw_data_from_file(self, file_path):
        """Read-in EnSight files using the vtkEnsightGoldBinaryReader.

        Args:
            file_path (str): Actual path to the file of interest.

        Returns:
            raw_data (obj): Raw data from file.
        """
        # Set vtk reader object as raw file data
        raw_data = vtk.vtkEnSightGoldBinaryReader()
        raw_data.SetCaseFileName(file_path)
        raw_data.Update()
        return raw_data

    def filter_and_manipulate_raw_data(self, raw_data):
        """Filter the ensight raw data for desired time steps.

        Args:
            raw_data (obj): Raw data from file.

        Returns:
            processed_data (np.array): Cleaned, filtered or manipulated *data_processor* data.
        """
        # determine the correct time vector depending on input specification
        if self.target_time_lst[0] == "from_experimental_data":
            # get unique time list
            time_lst = list(set(self.experimental_data[self.time_label_experimental]))
        else:
            time_lst = self.target_time_lst

        # loop over different time-steps here.
        time_lst = sorted(time_lst)
        processed_data = []
        for time_value in time_lst:
            vtk_data_per_time_step = self._vtk_from_ensight(raw_data, time_value)

            # check if data was found
            if not vtk_data_per_time_step:
                break

            # get ensight field from specified coordinates
            processed_data_per_time_step_interpolated = self._get_field_values_by_coordinates(
                vtk_data_per_time_step,
                time_value,
            )
            processed_data.append(processed_data_per_time_step_interpolated)

        return np.hstack(processed_data)

    def _get_field_values_by_coordinates(
        self,
        vtk_data_obj,
        time_value,
    ):
        """Interpolate the vtk field to the coordinates from experimental data.

        Args:
            vtk_data_obj (obj): VTK ensight object that contains that solution fields of
                                     interest
            time_value (float): Current time value at which simulation should be evaluated

        Returns:
            interpolated_data (np.array): Array of field values at specified coordinates/geometric
                                          sets
        """
        if self.geometric_target[0] == "experimental_data":
            response_data = self._get_data_from_experimental_coordinates(
                vtk_data_obj,
                time_value,
            )
        elif self.geometric_target[0] == "geometric_set":
            response_data = self._get_data_from_geometric_set(
                vtk_data_obj,
            )
        else:
            raise ValueError(
                "Geometric target for ensight vtk must be either 'geometric_set' or"
                f"'experimental_data'. You provided '{self.geometric_target[0]}'. Abort..."
            )

        return response_data

    def _get_data_from_experimental_coordinates(self, vtk_data_obj, time_value):
        """Interpolate the ensight/vtk field to experimental coordinates.

        Args:
            vtk_data_obj (obj): VTK ensight object that contains that solution fields of
                                     interest
            time_value (float): Current time value at which the simulation shall be evaluated

        Returns:
            interpolated_data (np.array): Array of field values interpolated to coordinates
                                          of experimental data
        """
        experimental_coordinates_for_snapshot = []
        if self.time_label_experimental:
            for _, row in self.experimental_data.iterrows():
                if time_value == row[self.time_label_experimental]:
                    experimental_coordinates_for_snapshot.append(
                        row[self.coordinates_label_experimental].to_list()
                    )
        else:
            for coord in self.coordinates_label_experimental:
                experimental_coordinates_for_snapshot.append(
                    np.array(self.experimental_data[coord]).reshape(-1, 1)
                )
            if len(experimental_coordinates_for_snapshot) != 3:
                raise ValueError("Please provide 3d coordinates in the observation data")
            experimental_coordinates_for_snapshot = np.concatenate(
                experimental_coordinates_for_snapshot, axis=1
            )
        # interpolate vtk solution to experimental coordinates
        interpolated_data = DataProcessorEnsight._interpolate_vtk(
            experimental_coordinates_for_snapshot,
            vtk_data_obj,
            self.vtk_array_type,
            self.vtk_field_label,
            self.vtk_field_components,
        )

        return interpolated_data

    def _get_data_from_geometric_set(
        self,
        vtk_data_obj,
    ):
        """Get entire fields for desired components.

        Args:
            vtk_data_obj (obj): VTK ensight object that contains that solution fields of
                                     interest
        Returns:
            data (np.array): Array of field values for nodes of geometric set
        """
        geometric_set = self.geometric_target[1]

        # get node coordinates by geometric set, loop over all topologies
        nodes_of_interest = None
        for nodes in self.geometric_set_data["node_topology"]:
            if nodes["topology_name"] == geometric_set:
                nodes_of_interest = nodes["node_mesh"]

        for lines in self.geometric_set_data["line_topology"]:
            if lines["topology_name"] == geometric_set:
                nodes_of_interest = lines["node_mesh"]

        for surfs in self.geometric_set_data["surface_topology"]:
            if surfs["topology_name"] == geometric_set:
                nodes_of_interest = surfs["node_mesh"]

        for vols in self.geometric_set_data["volume_topology"]:
            if vols["topology_name"] == geometric_set:
                nodes_of_interest = vols["node_mesh"]
        if nodes_of_interest is None:
            raise ValueError("Nodes of interest are not in the geometric set.")

        both = set(nodes_of_interest).intersection(
            self.geometric_set_data["node_coordinates"]["node_mesh"]
        )
        indices = [self.geometric_set_data["node_coordinates"]["node_mesh"].index(x) for x in both]
        geometric_set_coordinates = [
            self.geometric_set_data["node_coordinates"]["coordinates"][index] for index in indices
        ]

        # interpolate vtk solution to experimental coordinates
        interpolated_data = DataProcessorEnsight._interpolate_vtk(
            geometric_set_coordinates,
            vtk_data_obj,
            self.vtk_array_type,
            self.vtk_field_label,
            self.vtk_field_components,
        )

        return interpolated_data

    @staticmethod
    def _interpolate_vtk(
        coordinates, vtk_data_obj, vtk_array_type, vtk_field_label, vtk_field_components
    ):
        """Interpolate the vtk solution field to given coordinates.

        Args:
            coordinates (np.array): Coordinates at which the vtk solution field should be
                                    interpolated at
            vtk_data_obj (obj): VTK ensight object that contains that solution fields of
                                     interest
            vtk_array_type (str): Type of vtk array (cell array or point array)
            vtk_field_label (str): Field label that should be extracted
            vtk_field_components (lst): List of components of the respective fields that should
                                        be extracted

        Returns:
            interpolated_data (np.array): Solution data interpolated to respective coordinates.
        """
        # -- create vtk point object from experimental coordinates --
        vtk_points_obj = vtk.vtkPoints()
        vtk_points_obj.SetData(numpy_to_vtk(coordinates))
        # from vtk point obj create polydata obj
        polydata_obj = vtk.vtkPolyData()
        polydata_obj.SetPoints(vtk_points_obj)

        # Generate a vtk filter with desired vtk data as source
        probe_filter_obj = vtk.vtkProbeFilter()
        probe_filter_obj.SetSourceData(vtk_data_obj)

        # Evaluate/interpolate this filter at experimental coordinates
        probe_filter_obj.SetInputData(polydata_obj)
        probe_filter_obj.Update()

        # Return the fields as numpy array
        if vtk_array_type == "point_array":
            field = vtk_to_numpy(
                probe_filter_obj.GetOutput().GetPointData().GetArray(vtk_field_label)
            )

        elif vtk_array_type == "cell_array":
            field = vtk_to_numpy(
                probe_filter_obj.GetOutput().GetCellData().GetArray(vtk_field_label)
            )

        else:
            raise ValueError(
                "VTK array type must be either 'point_array' or 'cell_array', but you provided"
                f"{vtk_array_type}! Abort..."
            )

        data = []

        for i in vtk_field_components:
            if field.ndim == 1:
                data.append(field.reshape(-1, 1))
            else:
                data.append(field[:, i].reshape(-1, 1))
        data = np.concatenate(data, axis=1)

        # QUEENS expects a float64 numpy object as result
        interpolated_data = data.astype("float64")

        return interpolated_data

    def _vtk_from_ensight(self, raw_data, target_time):
        """Load a vtk-object from the ensight file.

        Args:
            raw_data (obj): Raw data from file
            target_time (float): Time the field should be evaluated on

        Returns:
            vtk_solution_field (obj)
        """
        # Ensight contains different "timesets" which are containers for the actual data
        number_of_timesets = raw_data.GetTimeSets().GetNumberOfItems()

        vtk_solution_field = None
        for num in range(number_of_timesets):
            time_set = vtk_to_numpy(raw_data.GetTimeSets().GetItem(num))
            # Find the timeset that has more than one entry as sometimes there is an empty dummy
            # timeset in the ensight file (seems to be an artifact)
            if time_set.size > 1:
                # if the keyword `last` was provided, get the last timestep
                if target_time == "last":
                    raw_data.SetTimeValue(time_set[-1])
                else:
                    timestep = time_set.flat[np.abs(time_set - target_time).argmin()]

                    if np.abs(timestep - target_time) > self.time_tol:
                        raise RuntimeError(
                            "Time not within tolerance"
                            f"Target time: {target_time}, selected time: {timestep},"
                            f"tolerance {self.time_tol}"
                        )
                    raw_data.SetTimeValue(timestep)

                raw_data.Update()
                input_vtk = raw_data.GetOutput()

                # the data is in the first block of the vtk object
                vtk_solution_field = input_vtk.GetBlock(0)

        return vtk_solution_field

    @staticmethod
    def read_geometry_coordinates(external_geometry):
        """Read geometry of interest.

        This method uses the QUEENS external geometry module.

        Args:
            external_geometry (queens.fourc_dat_geometry): QUEENS external geometry object

        Returns:
            dict: set with 4C topology
        """
        # read in the external geometry
        external_geometry.main_run()

        geometric_set_data = {
            "node_topology": external_geometry.node_topology,
            "line_topology": external_geometry.line_topology,
            "surface_topology": external_geometry.surface_topology,
            "volume_topology": external_geometry.volume_topology,
            "node_coordinates": external_geometry.node_coordinates,
            "element_centers": external_geometry.element_centers,
            "element_topology": external_geometry.element_topology,
        }

        return geometric_set_data
