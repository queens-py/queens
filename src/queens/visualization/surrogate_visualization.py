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
"""Visualization of surrogate models.

A module that provides utilities and a class for visualization of
surrogate models.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable


def convert_to_dict(values):
    """Convert values to dictionary with plot keys.

    Args:
        values (list): Values for 1D and 2D plot

    Returns:
        plot_dict (dict): Data as dictionary with plot keys
    """
    plot_keys = ["1d", "2d"]
    plot_dict = dict(zip(plot_keys, values))

    return plot_dict


class SurrogateVisualization:
    """Visualize a surrogate model.

    Visualization class for surrogate models that contains several plotting,
    storing and visualization methods that can be used anywhere in QUEENS.

    Attributes:
       saving_paths (dict): Dict of paths where to save the plots.
       should_be_saved (dict): Dict of booleans to save plots or not.
       should_be_displayed (dict): Dict of booleans for determining whether individual plots
                                   should be displayed or not.
       parameter_names (list): List of parameter names as strings.
       figures (dict): Dict of visualization figures.

    Returns:
        SAVisualization (obj): Instance of the SurrogateVisualization Class
    """

    def __init__(self, saving_paths, save_plot, display_plot):
        """Initialize the SurrogateVisualization object.

        Args:
            saving_paths: paths where to save the plots
            save_plot (dict): true if plots should be saved
            display_plot (dict): true if plots should be displayed
        """
        self.saving_paths = saving_paths
        self.should_be_saved = save_plot
        self.should_be_displayed = display_plot
        self.parameter_names = None
        self.figures = {}

    @classmethod
    def from_config_create(cls, plotting_options):
        """Create the SurrogateVisualization object.

        Args:
            plotting_options (dict): Dictionary containing the plotting_options

        Returns:
            Instance of SurrogateVisualization (obj)
        """
        if plotting_options:
            paths = [
                Path(plotting_options.get("plotting_dir", None), name)
                for name in plotting_options["plot_names"]
            ]
            saving_paths = convert_to_dict(paths)

            save_booleans = plotting_options.get("save_bool", False)
            save_plot = convert_to_dict(save_booleans)

            plot_booleans = plotting_options.get("plot_booleans", False)
            display_plot = convert_to_dict(plot_booleans)
        else:
            saving_paths = convert_to_dict([None, None])
            save_plot = convert_to_dict([False, False])
            display_plot = convert_to_dict([False, False])

        return cls(saving_paths, save_plot, display_plot)

    def plot(self, parameter_names, surrogate_model):
        """Call plotting methods for surrogate model.

        Args:
            parameter_names (lst): Parameter names
            surrogate_model (Model): Surrogate Model

        Returns:
            Plots of sensitivity indices
        """
        self.parameter_names = parameter_names

        if self.should_be_saved["1d"] or self.should_be_displayed["1d"]:
            self.plot_1d(surrogate_model)

        if self.should_be_saved["2d"] or self.should_be_displayed["2d"]:
            self.plot_2d(surrogate_model)

        # show all result plots in the end
        if any(self.should_be_displayed.values()):
            self._display_plots()

    def _display_plots(self):
        """Show plots according to input plot_booleans.

        Return:
            Displays plots.
        """
        for plot_key, current_figure in self.figures.items():
            if self.should_be_displayed[plot_key] is not True:
                plt.close(current_figure)

        plt.show()

    def plot_1d(self, gp_approximation):
        """Plot 1D projection of Gaussian process.

        Args:
            gp_approximation (RegressionApproximation object): Surrogate that holds GP model and
                                                               training data
        """
        for free_idx, parameter_name in enumerate(self.parameter_names):
            fig = self.plot_gp_from_gpflow(gp_approximation, free_idx)
            if self.should_be_saved["1d"]:
                fig.savefig(self.saving_paths["1d"] + "_" + parameter_name + ".png", dpi=200)

    def plot_gp_from_gpflow(self, gp_approximation, free_idx):
        """Plot 1D Gaussian process from GPFlow.

        Args:
            gp_approximation (RegressionApproximation object): Surrogate that holds GP model and
                                                               training data
            free_idx (int): Free index for plot

        Returns:
            fig: Figure of the Gaussian process
        """
        # training data
        x_train = gp_approximation.x_train
        y_train = gp_approximation.y_train

        # set up testing data for plot
        x_grid = np.linspace(x_train[:, free_idx].min(), x_train[:, free_idx].max(), 40)
        # all fixed parameters set to zero (standardized parameters!)
        x_test = np.zeros((40, x_train.shape[1]))
        x_test[:, free_idx] = x_grid

        # predict
        mean, variance = gp_approximation.model.predict_f(x_test)
        x_grid = x_grid.ravel()
        mean = mean.numpy().ravel()
        uncertainty = 1.96 * np.sqrt(variance.numpy().ravel())

        # plot
        fig, ax = plt.subplots()
        ax.fill_between(x_grid, mean + uncertainty, mean - uncertainty, alpha=0.1)
        ax.plot(x_grid, mean, label="Mean")
        ax.set_xlim((x_grid.min(), x_grid.max()))
        ax.set_xlabel(self.parameter_names[free_idx])

        if x_train:
            ax.scatter(x_train[:, free_idx].ravel(), y_train)

        return fig

    def plot_2d(self, gp_approximation):
        """Plot Gaussian process in 2D with training data points.

        Args:
            gp_approximation (RegressionApproximation object): Surrogate that holds GP model and
                                                               training data
        """
        if gp_approximation.x_train.shape[1] == 2:
            # training data
            x_train = gp_approximation.x_train
            y_train = gp_approximation.y_train

            # set up grid for plotting
            x1_grid = np.linspace(x_train[:, 0].min(), x_train[:, 0].max(), 40)
            x2_grid = np.linspace(x_train[:, 1].min(), x_train[:, 1].max(), 40)
            xx, yy = np.meshgrid(x1_grid, x2_grid)
            x_plot = np.vstack((xx.flatten(), yy.flatten())).T

            # predict
            mean, _ = gp_approximation.model.predict_f(x_plot)
            mean = mean.numpy()
            v_min, v_max = (fun(np.concatenate([y_train, mean])) for fun in (np.min, np.max))

            self._plot_2d_view(mean, x_train, y_train, xx, yy, v_min, v_max)
            self._plot_3d_view(mean, x_train, y_train, xx, yy, v_min, v_max)

    def _plot_3d_view(self, mean, x_train, y_train, xx, yy, v_min, v_max):
        """Plot 3D view of Gaussian process.

        Args:
            mean (ndarray): mean of GP
            x_train (ndarray): input values of training data
            y_train (ndarray): output values of training data
            xx (ndarray): grid for plotting (first axis)
            yy (ndarray): grid for plotting (second axis)
            v_min (float): minimum of colorbar
            v_max (float): maximum of colorbar
        """
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        ax.plot_surface(
            xx, yy, mean.reshape(*xx.shape), cmap="RdBu_r", linewidth=0, antialiased=False
        )
        ax.scatter(
            x_train[:, 0],
            x_train[:, 1],
            y_train,
            c=y_train,
            cmap="RdBu_r",
            vmin=v_min,
            vmax=v_max,
        )
        ax.set_xlabel(self.parameter_names[0])
        ax.set_ylabel(self.parameter_names[1])
        fig.savefig(self.saving_paths["2d"] + "_3d_view.png", dpi=200)

    def _plot_2d_view(self, mean, x_train, y_train, xx, yy, v_min, v_max):
        """Plot 2D view of Gaussian process.

        Args:
            mean (ndarray): mean of GP
            x_train (ndarray): input values of training data
            y_train (ndarray): output values of training data
            xx (ndarray): grid for plotting (first axis)
            yy (ndarray): grid for plotting (second axis)
            v_min (float): minimum of colorbar
            v_max (float): maximum of colorbar
        """
        fig, ax = plt.subplots()
        ax.scatter(
            x_train[:, 0],
            x_train[:, 1],
            c=y_train,
            cmap="RdBu_r",
            vmin=v_min,
            vmax=v_max,
            edgecolors="k",
        )
        ax.contour(
            xx,
            yy,
            mean.reshape(*xx.shape),
            levels=14,
            linewidths=0.5,
            colors="k",
            zorder=-100,
            vmin=v_min,
            vmax=v_max,
        )
        ax.contourf(
            xx,
            yy,
            mean.reshape(*xx.shape),
            levels=14,
            cmap="RdBu_r",
            zorder=-100,
            vmin=v_min,
            vmax=v_max,
        )
        # set colorbar
        for pcm in ax.get_children():
            if isinstance(pcm, ScalarMappable):
                fig.colorbar(pcm, ax=ax)
                break
        ax.set_xlabel(self.parameter_names[0])
        ax.set_ylabel(self.parameter_names[1])

        fig.savefig(self.saving_paths["2d"] + "_2d_view.png", dpi=200)
