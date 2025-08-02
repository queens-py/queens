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
"""Module providing visualization utilities for sensitivity analysis."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def convert_to_dict(values):
    """Convert values to dictionary with plot keys.

    Args:
        values (list): List of values to be converted.

    Returns:
        plot_dict (dict): Data as dictionary with plot keys
    """
    plot_keys = ["bar", "scatter"]
    plot_dict = dict(zip(plot_keys, values))

    return plot_dict


def convert_to_pandas(results):
    """Convert results to pandas DataFrame.

    Args:
        results (dict): Data as dictionary

    Returns:
        output (DataFrame): Data as pandas DataFrame with parameter names as index
    """
    output = pd.DataFrame.from_dict(results["sensitivity_indices"])
    output = output.set_index("names")
    return output


def annotate_points(data):
    """Annotate points in scatter plot with parameter names.

    Args:
         data (DataFrame): Data to be annotated
    """
    for parameter in data.index.values:
        plt.annotate(
            parameter,
            (data.loc[parameter, "mu_star"], data.loc[parameter, "sigma"]),
            textcoords="offset points",
            xytext=(0, 5),
            ha="center",
            fontsize=8,
        )


class SAVisualization:
    """Visualization class for sensitivity analysis.

    Visualization class for sensitivity analysis that contains several
    plotting, storing and visualization methods that can be used anywhere in
    QUEENS.

    Attributes:
       saving_paths (dict): Dict of paths where to save the plots.
       should_be_saved (dict): Dict of booleans to save plots or not.
       should_be_displayed (dict): Dict of booleans for determining whether individual plots
                                   should be displayed or not.
       figures (dict): Dictionary to hold figures for displaying later.

    Returns:
        SAVisualization (obj): Instance of the SAVisualization Class
    """

    def __init__(self, saving_paths, save_plot, display_plot):
        """Initialize the SAVisualization.

        Args:
            saving_paths (dict): Dictionary of paths where plots will be saved
            save_plot (dict): Dictionary of booleans indicating whether plots should be saved
            display_plot (dict): Dictionary of booleans indicating whether plots should be displayed
        """
        self.saving_paths = saving_paths
        self.should_be_saved = save_plot
        self.should_be_displayed = display_plot
        self.figures = {}

    @classmethod
    def from_config_create(cls, plotting_options):
        """Create the SAVisualization object from the problem description.

        Args:
            plotting_options (dict): Dictionary containing the plotting options

        Returns:
            Instance of SAVisualization (obj)
        """
        paths = [
            Path(plotting_options.get("plotting_dir"), name)
            for name in plotting_options["plot_names"]
        ]
        saving_paths = convert_to_dict(paths)

        save_booleans = plotting_options.get("save_bool")
        save_plot = convert_to_dict(save_booleans)

        plot_booleans = plotting_options.get("plot_booleans")
        display_plot = convert_to_dict(plot_booleans)

        return cls(saving_paths, save_plot, display_plot)

    def plot(self, results):
        """Call plotting methods for sensitivity analysis.

        Args:
            results (dict): Dictionary containing results to plot

        Returns:
            Plots of sensitivity indices
        """
        self.plot_si_bar(results)
        self.plot_si_scatter(results)

        # show all result plots in the end
        if any(self.should_be_displayed.values()):
            self._display_plots()

    def plot_si_bar(self, results):
        """Plot the sensitivity indices as bar plot with error bars.

        Args:
            results (dict): Dictionary containing results to plot

        Returns:
            Plot of sensitivity indices as bar plot
        """
        if self.should_be_saved["bar"] or self.should_be_displayed["bar"]:
            sensitivity_indices = convert_to_pandas(results)

            ax = sensitivity_indices.plot.bar(y="mu_star", yerr="sigma")

            ax.set_xlabel("Factors")
            ax.set_ylabel(r"$\mu^*_i$ and $\sigma_i$ (confidence intervals)")
            ax.set_title("Elementary Effects Analysis")

            ax.yaxis.grid(True)
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()

            self._save_plot("bar")
            self.figures["bar"] = plt.gcf()

    def plot_si_scatter(self, results):
        """Plot sensitivity indices as a scatter plot.

        Plot the sensitivity indices as scatter plot of *sigma* over
        *mu_star*.

        Args:
            results (dict): Dictionary containing results to plot

        Returns:
            Plot of sensitivity indices as scatter plot
        """
        if self.should_be_saved["scatter"] or self.should_be_displayed["scatter"]:
            sensitivity_indices = convert_to_pandas(results)

            ax = sensitivity_indices.plot.scatter(x="mu_star", y="sigma")
            annotate_points(sensitivity_indices)

            ax.set_xlabel(r"$\mu^*_i$")
            ax.set_ylabel(r"$\sigma_i$")
            ax.set_title("Elementary Effects Analysis")

            ax.yaxis.grid(True)
            plt.tight_layout()

            self._save_plot("scatter")
            self.figures["scatter"] = plt.gcf()

    def _display_plots(self):
        """Show plots according to input plot_booleans.

        Return:
            Displays plots.
        """
        for plot_key, current_figure in self.figures.items():
            if not self.should_be_displayed[plot_key]:
                plt.close(current_figure)

        plt.show()

    def _save_plot(self, key):
        """Save the plot to specified path.

        Args:
            key (str): key of current plot

        Returns:
            Saved plot.
        """
        if self.should_be_saved[key]:
            plt.savefig(self.saving_paths[key], dpi=300)
