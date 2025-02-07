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
"""Provide utilities and a class for visualization in BMFIA analysis."""

# pylint: disable=invalid-name
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import seaborn as sns
from plotly.subplots import make_subplots


class BMFIAVisualization:
    """Visualization class for BMFIA with plotting and saving capabilities.

    Visualization class for BMFIA that contains several plotting, storing
    and visualization methods that can be used anywhere in QUEENS.

    Attributes:
       paths (list): List with paths to save the plots.
       save_bools (list): List with booleans to save plots.
       plot_booleans (list): List of booleans for determining whether individual plots should be
                             plotted or not.

    Returns:
        BMFIAVisualization (obj): Instance of the BMFIAVisualization Class
    """

    def __init__(self, paths, save_bools, plot_booleans):
        """Initialize the BMFIAVisualization.

        Args:
            paths (list): Paths to save the plots.
            save_bools (list): Booleans indicating whether to save plots.
            plot_booleans (list): Booleans indicating whether to plot data.
        """
        self.paths = paths
        self.save_bools = save_bools
        self.plot_booleans = plot_booleans

    @classmethod
    def from_config_create(cls, plotting_options):
        """Create the BMFIA visualization object from the problem description.

        Args:
            plotting_options (dict): Dictionary with plotting options

        Returns:
            Instance of BMFIA visualization (obj)
        """
        paths = [
            Path(plotting_options.get("plotting_dir")) / name
            for name in plotting_options["plot_names"]
        ]
        save_bools = plotting_options.get("save_bool")
        plot_booleans = plotting_options.get("plot_booleans")
        return cls(paths, save_bools, plot_booleans)

    def plot(self, z_train, Y_HF_train, regression_obj_lst):
        """Plot probabilistic manifold and informative features.

        Plot the probabilistic manifold of high-fidelity, low-fidelity
        outputs and informative features of the input space, depending on the
        description in the input file. Also plot the probabilistic mapping
        along with its training points. Potentially animate and save these
        plots.

        Args:
            z_train (np.array): Low-fidelity feature vector
            Y_HF_train (np.array): High-fidelity output training points
            regression_obj_lst (lst): List of involved regression objects

        Returns:
            Plots of the probabilistic manifold
        """
        if self.plot_booleans[0]:
            plot_model_dependency(z_train, Y_HF_train, regression_obj_lst)
            if self.save_bools[0] is not None:
                _save_plot(self.save_bools[0], self.paths[0])

    def plot_posterior_from_samples(self, samples, weights, dim_labels_lst):
        """Visualize the posterior distribution or marginals for posteriors.

        Visualize the multi-fidelity posterior distribution (up to 2D) or
        its marginals for higher dimensional posteriors.

        Args:
            samples (np.array): Samples of the posterior. Each row is a different sample-vector.
                                Different columns represent the different dimensions of the
                                posterior.
            weights (np.array): Weights of the posterior samples. One weight for each sample row.
            dim_labels_lst (lst): List with labels/naming of the involved dimensions.
                                Order of the list corresponds to order of columns in sample matrix.
        """
        if self.plot_booleans[1]:
            if samples.shape[1] > 2:
                raise RuntimeError(
                    f"At the moment we only support posterior plots up to two dimensions. "
                    f"Your posterior has {samples.shape[1]}-dimensions. Abort ...."
                )

            sns.set_theme(style="whitegrid")
            _, ax = plt.subplots(figsize=(6, 6))
            sns.scatterplot(x=samples[:, 0], y=samples[:, 1], s=5)
            sns.kdeplot(x=samples[:, 0], y=samples[:, 1], weights=weights)

            ax.set_title(r"Posterior distribution $p(x,y|D)$")
            ax.set_xlabel(rf"${dim_labels_lst[0]}$")
            ax.set_ylabel(rf"${dim_labels_lst[1]}$")
            ax.set_xlim(-0.2, 1.2)
            ax.set_ylim(-0.2, 1.2)

            plt.show()

            if self.save_bools[1] is not None:
                _save_plot(self.save_bools[1], self.paths[1])


def plot_model_dependency(z_train, Y_HF_train, regression_obj_lst):
    r"""Plot multi-fidelity dependencies with optional informative features.

    Plot the multi-fidelity dependency in :math:`\Omega_{y_{lf}\times
    y_{hf}}` or in :math:`\Omega_{y_{lf}\times y_{hf}\times \gamma_1}`

    Args:
        z_train (np.array): Training data for the low-fidelity vector that contains the
                            output of the low-fidelity model and potentially informative
                            low-fidelity features
        Y_HF_train (np.array): Training vector of the high-fidelity model outputs
        regression_obj_lst (list): List containing (probabilistic)
    """
    z_train = z_train.squeeze()
    Y_HF_train = Y_HF_train.squeeze()

    if z_train.ndim == 2:  # one array is already 2 dim due potentially several GPs
        _plot_2d_dependency(z_train, Y_HF_train, regression_obj_lst)
    elif z_train.ndim == 3:  # one array is already 3 dim due potentially several GPs
        _plot_3d_dependency(z_train, Y_HF_train, regression_obj_lst)
    else:
        raise RuntimeError("Dimension of intended surrogate is too high to be plotted! Abort")


def _plot_3d_dependency(z_train, y_hf_train, regression_obj_lst):
    """Plot 3D dependencies of low-fidelity and high-fidelity outputs.

    Plot the 3D-dependency meaning the LF-HF dependency with one more
    informative feature.

    Args:
        z_train (np.array): Array of low-fidelity model output and informative features.
                            One sample per row
        y_hf_train (np.array): Array of high-fidelity model outputs. One sample per row.
        regression_obj_lst (np.array): List with regression models
    """
    num_test_points = 50

    num_rows = int(np.ceil(np.sqrt(z_train.shape[2])))
    row_list = [{"type": "surface"}] * num_rows

    specs_list = [row_list for x in range(num_rows)]
    fig = make_subplots(rows=num_rows, cols=num_rows, specs=specs_list)

    for i in np.arange(num_rows):
        for j in np.arange(num_rows):
            num_coord = j + i * num_rows

            y_lf_test = np.linspace(
                np.min(z_train[0, :, num_coord]), np.max(z_train[0, :, num_coord]), num_test_points
            )
            gamma_lf_test = np.linspace(
                np.min(z_train[1, :, num_coord]), np.max(z_train[1, :, num_coord]), num_test_points
            )
            y_test, gamma_test = np.meshgrid(y_lf_test, gamma_lf_test)
            z_test = np.hstack((y_test.reshape(-1, 1), gamma_test.reshape(-1, 1)))

            reg_obj = regression_obj_lst[num_coord]
            output = reg_obj.predict(z_test, support="y")
            mu = output["mean"]
            var = output["variance"]
            row = j + 1
            col = i + 1

            fig.add_trace(
                go.Scatter3d(
                    x=z_train[0, :, num_coord].flatten(),
                    y=z_train[1, :, num_coord].flatten(),
                    z=y_hf_train[:, num_coord].flatten(),
                    mode="markers",
                    marker={"size": 3},
                ),
                row=row,
                col=col,
            )

            fig.add_trace(
                go.Surface(
                    x=y_test,
                    y=gamma_test,
                    z=mu.reshape(y_test.shape),
                    colorscale="Viridis",
                    showscale=False,
                ),
                row=row,
                col=col,
            )

            fig.add_trace(
                go.Surface(
                    x=y_test,
                    y=gamma_test,
                    z=(mu - np.sqrt(var)).reshape(y_test.shape),
                    colorscale="Viridis",
                    showscale=False,
                    opacity=0.5,
                ),
                row=row,
                col=col,
            )
            fig.add_trace(
                go.Surface(
                    x=y_test,
                    y=gamma_test,
                    z=(mu + np.sqrt(var)).reshape(y_test.shape),
                    colorscale="Viridis",
                    showscale=False,
                    opacity=0.5,
                ),
                row=row,
                col=col,
            )

            fig.update_layout(
                scene={
                    "xaxis": {"title": r"$y_{LF}$"},
                    "yaxis": {"title": r"$\gamma$"},
                    "zaxis": {"title": r"$y_{HF}$"},
                },
                xaxis_range=[np.min(z_train[0, :, num_coord]), np.max(z_train[0, :, num_coord])],
                yaxis_range=[np.min(z_train[1, :, num_coord]), np.max(z_train[1, :, num_coord])],
                scene_aspectmode="cube",
            )

    fig.show()


def _plot_2d_dependency(z_train, Y_HF_train, regression_obj_lst):
    """Plot the 2D-dependency meaning the LF-HF dependency.

    Args:
        z_train (np.array): Array of low-fidelity model output and informative features.
                            One sample per row
        Y_HF_train (np.array): Array of high-fidelity model outputs. One sample per row.
        regression_obj_lst (np.array): List with regression models
    """
    sub_plot_square = int(np.ceil(np.sqrt(len(regression_obj_lst))))

    fig1, axs = plt.subplots(sub_plot_square, sub_plot_square)
    axs = axs.reshape(-1, 1)

    # --------------------------------------------------------------------------------

    # posterior mean
    for axx, regression_obj, z, yhf in zip(axs, regression_obj_lst, z_train.T, Y_HF_train.T):
        # generate support for plotting
        y_lf_test = np.linspace(np.min(yhf), np.max(yhf), 100)

        ax = axx[0]
        output_dict = regression_obj.predict(y_lf_test.T, "y")
        mean_vec = np.atleast_2d(output_dict["mean"]).T
        var_vec = output_dict["variance"]
        std_vec = np.atleast_2d(np.sqrt(var_vec)).T

        # identity
        ax.plot(
            Y_HF_train.reshape(-1, 1).squeeze(),
            Y_HF_train.reshape(-1, 1).squeeze(),
            linestyle="-",
            marker="",
            color="g",
            alpha=1,
            linewidth=2,
            label=r"$y_{\mathrm{HF}}=y_{\mathrm{LF}}$, (Identity)",
        )

        ax.scatter(
            z,
            yhf,
            marker="x",
            s=70,
            color="r",
            label=r"$\mathcal{D}_{y}=\{Y_{\mathrm{LF}},Y_{\mathrm{HF}}\}$, (Training)",
        )

        ax.plot(
            y_lf_test,
            mean_vec,
            color="darkblue",
            linewidth=1,
            label=r"$\mathrm{m}_{\mathcal{D}_y}(y_{\mathrm{LF}})$, (Posterior mean)",
        )

        # posterior confidence
        ax.plot(
            y_lf_test,
            np.add(mean_vec, 2 * std_vec),
            color="darkblue",
            linewidth=1,
            linestyle="--",
            alpha=0.5,
            label=r"$\mathrm{m}_{\mathcal{D}_y}(y_{\mathrm{LF}})\pm 2\cdot\sqrt{\mathrm{v}_"
            r"{\mathcal{D}_y}(y_{\mathrm{LF}})}$, (Confidence)",
        )

        ax.plot(
            y_lf_test,
            np.add(mean_vec, -2 * std_vec),
            color="darkblue",
            alpha=0.5,
            linewidth=1,
            linestyle="--",
        )

        ax.set_xlabel(r"$y_{\mathrm{LF}}$")
        ax.set_ylabel(r"$y_{\mathrm{HF}}$")
        ax.grid(which="major", linestyle="-")
        ax.grid(which="minor", linestyle="--", alpha=0.5)
        ax.set_xlim([min(y_lf_test), max(y_lf_test)])
        ax.set_ylim([min(y_lf_test), max(y_lf_test)])
        ax.minorticks_on()
        ax.legend()

    fig1.set_size_inches(25, 25)


def _save_plot(save_bool, path):
    """Save the plot to specified path.

    Args:
        save_bool (bool): Flag to decide whether saving option is triggered.
        path (str): Path where to save the plot.

    Returns:
        Saved plot.
    """
    if save_bool:
        plt.savefig(path, dpi=300)
