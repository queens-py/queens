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
"""Data processor module for vtk ensight boundary data."""

import re
from pathlib import Path

import numpy as np
import vtk
from vtkmodules.util.numpy_support import vtk_to_numpy

from queens.data_processor.data_processor import DataProcessor
from queens.utils.logger_settings import log_init_args


class DataProcessorEnsightInterfaceDiscrepancy(DataProcessor):
    """Discrepancy measure for boundaries and shapes.

    *data_processor* class uses full ensight result in vtk to measure distance of
    surface from simulations to experiment.

    Attributes:
        time_tol (float): Time tolerance for given reference time points.
        visualization_bool (bool): Boolean for vtk visualization control.
        displacement_fields (str): String with exact field names for displacement to apply.
        problem_dimension (string): String to determine problems in spatial dimension.
        experimental_ref_data_lst (list): Experimental reference data to which the
                                          discrepancy measure is computed.
    """

    @log_init_args
    def __init__(
        self,
        file_name_identifier=None,
        file_options_dict=None,
        files_to_be_deleted_regex_lst=None,
    ):
        """Initialize data_processor_ensight_interface class.

        Args:
            data_processor_name (str): Name of the data processor.
            file_name_identifier (str): Identifier of file name. The file prefix can contain regex
                                        expression and subdirectories.
            file_options_dict (dict): Dictionary with read-in options for the file:
                - path_to_ref_data (str): Path to experimental reference data to which the
                                          discrepancy measure is computed.
                - time_tol (float): time tolerance for given reference time points
                - visualization (bool): boolean for vtk visualization control
                - displacement_fields (str): String with exact field names for displacement to apply
                - problem_dimension (string): string to determine problems spatial dimension
            files_to_be_deleted_regex_lst (lst): List with paths to files that should be deleted.
                                                 The paths can contain regex expressions.
        """
        super().__init__(
            file_name_identifier=file_name_identifier,
            file_options_dict=file_options_dict,
            files_to_be_deleted_regex_lst=files_to_be_deleted_regex_lst,
        )

        path_ref_data_str = file_options_dict.get("path_to_ref_data")
        if not path_ref_data_str:
            raise ValueError(
                "You must provide the option 'path_to_ref_data' within the 'file_options_dict' "
                f"in '{self.__class__.__name__}'. Abort ..."
            )
        path_ref_data = Path(path_ref_data_str)
        experimental_reference_data = self.read_monitorfile(path_ref_data)

        time_tol = file_options_dict.get("time_tol")
        if not time_tol:
            raise ValueError(
                "You must provide the option 'time_tol' within the 'file_options_dict' "
                f"in '{self.__class__.__name__}'. Abort ..."
            )

        visualization_bool = file_options_dict.get("visualization", False)
        if not isinstance(visualization_bool, bool):
            raise TypeError(
                "The option 'visualization' must be of type 'bool' "
                f"but you provided type {type(visualization_bool)}. Abort..."
            )

        displacement_fields = file_options_dict.get("displacement_fields", ["displacement"])
        if not isinstance(displacement_fields, list):
            raise TypeError(
                "The option 'displacement_fields' must be of type 'list' "
                f"but you provided type {type(displacement_fields)}. Abort..."
            )

        problem_dimension = file_options_dict.get("problem_dimension", "2d")
        if not isinstance(problem_dimension, str):
            raise TypeError(
                "The option 'problem_dimension' must be of type 'str' "
                f"but you provided type {type(problem_dimension)}. Abort..."
            )

        self.time_tol = time_tol
        self.visualization_bool = visualization_bool
        self.displacement_fields = displacement_fields
        self.problem_dimension = problem_dimension
        self.experimental_ref_data_lst = experimental_reference_data

    @staticmethod
    def read_monitorfile(path_to_experimental_reference_data):
        """Read Monitor file.

        The Monitor File contains measurements from the
        experiments.

        Args:
            path_to_experimental_reference_data (path obj):
                Path to experimental reference data

        Returns:
            monfile_data (list): Data from monitor file in numbers
        """
        with open(path_to_experimental_reference_data, encoding="utf-8") as my_file:
            lines = my_file.readlines()
            i = 0
            npoints = 0
            steps = 0
            # lines specifying number of spatial dimensions and dimension ids
            npoint_lines = []
            # measurements for all points in different time steps
            steps_lines = []
            # sort lines into npoint_lines and steps_lines
            for line in lines:
                if line.startswith("#"):
                    continue
                line = line.strip()
                if line.startswith("steps"):
                    firstline = line
                    steps = re.findall("^(?:steps )(.+)(?= npoints)", firstline, re.M)
                    steps = int(steps[0])
                    npoints = re.findall("^(?:steps )(?:.+)?(?: npoints )(.+)", firstline, re.M)
                    npoints = int(npoints[0])
                    continue
                if i < npoints:
                    npoint_lines.append(line.split())
                    i += 1
                    continue
                if i - npoints - 1 < steps:
                    steps_lines.append(line.split())

            if npoints == 0 or steps == 0:
                raise ValueError(
                    "read_monitorfile did not find useful content. Monitor format is probably wrong"
                )

        # read numeric content from file data
        npoint_lines = [[int(ii) for ii in i] for i in npoint_lines]
        steps_lines = [[float(ii) for ii in i] for i in steps_lines]

        # prefill monfile_data of adequate size with zeros
        # monfile_data has dimensions
        # [number of timesteps][2][number of points][2][3dim]
        # it contains pairs of points on the interface and in the domain (for distance
        # in prescribed direction) measured in experiment
        monfile_data = []
        for i in steps_lines:
            monfile_data.append(
                [[0.0e0], [[[0, 0, 0] for j in range(0, 2)] for k in range(0, npoints)]]
            )

        # for all npoint_lines read according data from steps_lines to monfile_data
        # loop over time steps

        for i, steps_line in enumerate(steps_lines):
            k = 1
            # save time value for time step
            monfile_data[i][0] = steps_line[0]
            # loop over points
            for ii, npoint_line in enumerate(npoint_lines):
                for x in range(0, 2):
                    for iii in range(0, npoint_line[0]):
                        monfile_data[i][1][ii][x][npoint_line[iii + 1]] = steps_line[k]
                        k += 1
        return monfile_data

    def get_raw_data_from_file(self, file_path):
        """Read-in EnSight file using vtkGenericEnSightReader.

        Args:
            file_path (str): Actual path to the file of interest.

        Returns:
            raw_data (obj): Raw data from file.
        """
        raw_data = vtk.vtkGenericEnSightReader()
        raw_data.SetCaseFileName(file_path)
        raw_data.ReadAllVariablesOn()
        raw_data.Update()
        return raw_data

    def filter_and_manipulate_raw_data(self, raw_data):
        """Get deformed boundary from vtk.

        Create vtk representation of deformed external_geometry_obj and
        evaluate surface distance measurement for every given time step from the
        experiment.

        Args:
            raw_data (obj): Raw data from file.

        Returns:
            residual (list): Full residual from this data_processor class
        """
        residual_distance_lst = []
        points = vtk.vtkPoints()
        vertices = vtk.vtkCellArray()

        for current_step_experimental_data in self.experimental_ref_data_lst:
            grid = self.deformed_grid(raw_data, current_step_experimental_data[0])
            geo = vtk.vtkGeometryFilter()
            geo.SetInputData(grid)
            geo.Update()
            geometry_output = geo.GetOutput()
            outline_out, outline_data = self._get_dim_dependent_vtk_output(geometry_output)

            for measured_point_pair in current_step_experimental_data[1]:
                point_vector = self.stretch_vector(
                    measured_point_pair[0], measured_point_pair[1], 10
                )
                intersection_points = self._get_intersection_points(
                    outline_data, outline_out, point_vector
                )
                distance = self.compute_distance(intersection_points, measured_point_pair)
                residual_distance_lst.append(distance)

                self._visualize_intermediate_discrepancy_measure(
                    points, vertices, intersection_points, point_vector
                )

            self._visualize_final_discrepancy_measure(outline_out, points, vertices)

        return residual_distance_lst

    def _get_intersection_points(self, outline_data, outline_out, point_vector):
        """Get intersection points."""
        counter = 0
        intersection_points_lst = []

        while counter < len(outline_data):
            numpoints = outline_data.item(counter)
            locpointids = outline_data[counter + 1 : counter + 1 + numpoints]

            locations = []
            for idx in np.nditer(locpointids):
                x = [0, 0, 0]
                outline_out.GetPoint(idx, x)
                locations.append(x)

            local_points = vtk.vtkPoints()
            for location in locations:
                local_points.InsertNextPoint(location)

            local_element = self._get_local_element(locations)
            local_element.Initialize(len(locations), local_points)

            intersection_point = [0, 0, 0]
            pcoords = [0, 0, 0]

            # key line of the algorithm: line intersection
            intersectionfound = local_element.IntersectWithLine(
                point_vector[0],
                point_vector[1],
                1e-12,
                vtk.reference(0),
                intersection_point,
                pcoords,
                vtk.reference(0),
            )

            if intersectionfound:
                intersection_points_lst.append(intersection_point)

            counter += numpoints
            counter += 1

        return intersection_points_lst

    def _get_local_element(self, locations):
        """Get the local element based on input dimension."""
        if len(locations) == 2:
            local_element = vtk.vtkLine()
        elif len(locations) == 3:
            local_element = vtk.vtkTriangle()
        elif len(locations) == 4:
            local_element = vtk.vtkQuad()
        else:
            raise ValueError("Unknown local_element type for structure surface discretization.")
        return local_element

    def _get_dim_dependent_vtk_output(self, geoout):
        """Return the vtk output dependent of problem dimension."""
        if self.problem_dimension == "2d":
            outline = vtk.vtkFeatureEdges()
            outline.SetInputData(geoout)
            outline.Update()
            outlineout = outline.GetOutput()

            outlines = outlineout.GetLines()
            outline_data_vtk = outlines.GetData()

        elif self.problem_dimension == "3d":
            outlineout = geoout
            outlines = outlineout.GetPolys()
            outline_data_vtk = outlines.GetData()

        else:
            raise KeyError(
                "The problem dimension must be either '2d' or '3d' "
                f"but you provided {self.problem_dimension}! Abort..."
            )

        outline_data = vtk_to_numpy(outline_data_vtk)
        return outlineout, outline_data

    def _visualize_intermediate_discrepancy_measure(
        self, points, vertices, intersectionpoints, point_vector
    ):
        """Visualize intermediate discrepancy measure in vtk."""
        if self.visualization_bool:
            for j in point_vector:
                x = points.InsertNextPoint(j)
                vertices.InsertNextCell(1)
                vertices.InsertCellPoint(x)

            for idx, j in enumerate(intersectionpoints):
                x = points.InsertNextPoint(j)
                vertices.InsertNextCell(idx + 1)
                vertices.InsertCellPoint(x)

    def _visualize_final_discrepancy_measure(self, outlineout, points, vertices):
        """Visualize the final discrepancy measure in vtk."""
        if self.visualization_bool:
            colors = vtk.vtkNamedColors()
            renderer = vtk.vtkRenderer()
            ren_win = vtk.vtkRenderWindow()
            ren_win.AddRenderer(renderer)
            iren = vtk.vtkRenderWindowInteractor()
            iren.SetRenderWindow(ren_win)

            pointdata = vtk.vtkPolyData()

            pointdata.SetPoints(points)
            pointdata.SetVerts(vertices)

            ugrid_mapper = vtk.vtkDataSetMapper()
            ugrid_mapper.SetInputData(outlineout)

            point_mapper = vtk.vtkPolyDataMapper()

            point_mapper.SetInputData(pointdata)
            point_actor = vtk.vtkActor()
            point_actor.SetMapper(point_mapper)
            point_actor.GetProperty().SetColor([0.0, 0.0, 1.0])
            point_actor.GetProperty().SetPointSize(10)
            point_actor.GetProperty().SetRenderPointsAsSpheres(True)

            ugrid_actor = vtk.vtkActor()
            ugrid_actor.SetMapper(ugrid_mapper)
            ugrid_actor.GetProperty().SetColor(colors.GetColor3d("Peacock"))
            ugrid_actor.GetProperty().EdgeVisibilityOn()

            renderer.AddActor(ugrid_actor)
            renderer.AddActor(point_actor)
            renderer.SetBackground(colors.GetColor3d("Beige"))

            renderer.ResetCamera()
            renderer.GetActiveCamera().Elevation(1.0)
            renderer.GetActiveCamera().Azimuth(0.01)
            renderer.GetActiveCamera().Dolly(0)

            ren_win.SetSize(640, 480)

            # Generate viewer
            ren_win.Render()
            iren.Start()

    def stretch_vector(self, vec1, vec2, scalar):
        """Extend a vector by scalar factor on both ends.

        Args:
            vec1 (list): root point coordinates
            vec2 (list): directional point coordinates
            scalar (float): scalar multiplier

        Returns:
            vec (list): vector from modified root to modified direction point
        """
        vec = [[], []]

        for vec1_ele, vec2_ele in zip(vec1, vec2):
            vec[0].append(vec1_ele - scalar * (vec2_ele - vec1_ele))
            vec[1].append(vec2_ele + scalar * (vec2_ele - vec1_ele))

        return vec

    def compute_distance(self, intersection_points, measured_points):
        """Find the furthest point for a set of intersection points.

        Args:
            intersection_points (list): intersection point coordinates
            measured_points (list): pair of points from monitor file

        Returns:
            distance (float): signed distance between root point and furthest outward
                            intersection point; positive if in positive direction from root
        """
        distance = np.inf

        np1m = np.array(measured_points[0])
        np2m = np.array(measured_points[1])

        for p in intersection_points:
            npp = np.array(p)
            dist = np.linalg.norm(p - np1m, ord=2)
            if np.dot(np2m - np1m, npp - np1m) < 0:
                dist *= -1
            distance = min(distance, dist)

        return distance

    def deformed_grid(self, raw_data, time):
        """Read deformed grid from Ensight file at specified time.

        Initially, the undeformed grid is read from the Ensight file
        Afterward, *warpbyvector* applies the displacement of *structure* field at time *time*
        such that the final result is the deformed grid at the specified time.

        Args:
            raw_data (obj): Raw data from file
            time (float): Time value for data processing

        Returns:
            deformed_grid (vtkUnstructuredGrid): Deformed grid for given time
        """
        time_steps_in_ensight = raw_data.GetTimeSets()

        times_iter = time_steps_in_ensight.NewIterator()
        times_iter.GoToFirstItem()

        steps = np.array([])

        while not times_iter.IsDoneWithTraversal():
            curr = times_iter.GetCurrentObject()
            steps = np.append(steps, vtk_to_numpy(curr))
            times_iter.GoToNextItem()

        steps = np.unique(steps)
        idx = np.where(abs(steps - time) < self.time_tol)
        ensight_time = steps[idx]

        if len(ensight_time) > 1:
            raise ValueError(
                "point in time from *.monitor file used with time_tol is not unique in results"
            )
        if len(ensight_time) == 0:
            raise ValueError(
                "point in time from *.monitor file used with time_tol not existing in results"
            )

        raw_data.SetTimeValue(ensight_time)
        raw_data.Update()

        output = raw_data.GetOutput()
        number_of_blocks = output.GetNumberOfBlocks()
        if number_of_blocks != 1:
            raise ValueError(
                "ensight reader output has more or less than one block. This is not expected."
                "Investigate your data!"
            )
        block = output.GetBlock(0)

        block.GetPointData().SetActiveVectors(self.displacement_fields[0])

        vtk_warp_vector = vtk.vtkWarpVector()
        vtk_warp_vector.SetScaleFactor(1.0)
        vtk_warp_vector.SetInputData(block)
        vtk_warp_vector.Update()
        if len(self.displacement_fields) > 1:
            for i, field in enumerate(self.displacement_fields):
                if i > 0:
                    second_block = vtk_warp_vector.GetOutput()
                    second_block.GetPointData().SetActiveVectors(field)
                    wvb = vtk.vtkWarpVector()
                    wvb.SetScaleFactor(1.0)
                    wvb.SetInputData(second_block)
                    wvb.Update()
                    vtk_warp_vector = wvb

        deformed_grid = vtk_warp_vector.GetUnstructuredGridOutput()

        return deformed_grid
