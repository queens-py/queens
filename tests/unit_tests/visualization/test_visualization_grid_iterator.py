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
"""Unit tests for grid iterator visualization."""

import re

import numpy as np
import pytest

from queens.visualization.grid_iterator_visualization import GridIteratorVisualization


class TestVisualizationGridIterator:
    """Collection of tests for visualization features for the grid iterator."""

    @pytest.fixture(name="grid_visualization")
    def fixture_grid_visualization(self, tmp_path):
        """Grid visualization object."""
        paths = [tmp_path / "myplot.png"]
        save_bools = [True]
        plot_booleans = [False]
        scale_types_list = ["lin", "lin"]
        var_names_list = ["x1", "x2"]
        grid_vis = GridIteratorVisualization(
            paths, save_bools, plot_booleans, scale_types_list, var_names_list
        )
        return grid_vis

    # ------------------------------- actual unit_tests --------------------------------------------
    def test_init(self, tmp_path):
        """Test initialization of GridIteratorVisualization attributes."""
        # expected attributes
        paths = [tmp_path / "myplot.png"]
        save_bools = [True]
        plot_booleans = [False]
        scale_types_list = ["lin", "lin"]
        var_names_list = ["x1", "x2"]

        # instantiated class
        grid_vis = GridIteratorVisualization(
            paths, save_bools, plot_booleans, scale_types_list, var_names_list
        )

        # tests / asserts
        assert grid_vis.saving_paths_list == paths
        assert grid_vis.save_bools == save_bools
        assert grid_vis.plot_booleans == plot_booleans
        assert grid_vis.scale_types_list == scale_types_list
        assert grid_vis.var_names_list == var_names_list

    def test_plot_qoi_grid(self, tmp_path, grid_visualization):
        """Test plotting of grid."""
        # set arguments
        output = {"result": np.array([1.0, 2.0, 3.0, 4.0])}
        samples = np.array([[1.0, 1.0], [2.0, 2.0], [1.0, 2.0], [2.0, 1.0]])
        num_params = 2
        n_grid_p = [2, 2]
        grid_visualization.plot_qoi_grid(output, samples, num_params, n_grid_p)
        filepath = tmp_path / "myplot.png"
        assert filepath.is_file()

    def test_get_plotter_one(self, grid_visualization):
        """Test retrieval of plotter function for one-dimensional data."""
        num_params = 1
        plotter = grid_visualization.get_plotter(num_params)
        expected_str = re.split("[\\s\\.]", str(plotter))[3]
        assert "plot_one_d" == expected_str

    def test_get_plotter_two(self, grid_visualization):
        """Test retrieval of plotter function for two-dimensional data."""
        num_params = 2
        plotter = grid_visualization.get_plotter(num_params)
        expected_str = re.split("[\\s\\.]", str(plotter))[3]
        assert "plot_two_d" == expected_str

    def test_higher_d(self, grid_visualization):
        """Test for plotter function with more than two parameters."""
        num_params = 3
        with pytest.raises(NotImplementedError) as not_implemented_error:
            grid_visualization.get_plotter(num_params)
        assert str(not_implemented_error.value) == "Grid plot only possible up to 2 parameters"

    def test_plot_one_d(self, grid_visualization):
        """Test one-dimensional plot."""
        output = {"result": np.array([0.0, 1.0])}
        samples = np.array([0.0, 1.0])
        dummy = 1.0
        grid_visualization.plot_one_d(output, samples, dummy)

    def test_plot_two_d(self, grid_visualization):
        """Test two-dimensional plot."""
        samples = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
        output = {"result": np.array([0.0, 1.0, 0.0, 1.0])}
        n_grid_p = [2, 2]
        grid_visualization.plot_two_d(output, samples, n_grid_p)
