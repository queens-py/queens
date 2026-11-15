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
"""Visualization utilities for adaptive sampling results."""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from matplotlib import colormaps
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from queens.distributions._distribution import Continuous
from queens.distributions.gaussian_kde import GaussianKDE
from queens.parameters.parameters import Parameters

if TYPE_CHECKING:
    from queens.global_settings import GlobalSettings

_logger = logging.getLogger(__name__)

AdaptiveSamplingResults = Mapping[str, Sequence[Any]]


class AdaptiveSamplingVisualization:
    """Plot the posterior of an adaptive sampling iteration.

    Layouts:

    - ``"pair_grid"``: univariate posterior densities on the diagonal, bivariate posterior
      densities in the lower triangle and training samples in the upper triangle.
    - ``"1d_marginals"``: univariate posterior densities only.

    The most likely evaluated training sample is marked in the univariate posterior densities; its
    values are listed in the legend.

    Attributes:
        layout: Layout of the figure, either "pair_grid" or "1d_marginals".
        figure_size: Figure size (width, height) in inches.
        kde_grid_size: Number of grid points per axis for bivariate densities.
        kde_num_points: Number of grid points for univariate densities.
        contour_levels: Number of contour levels for bivariate densities.
        colormap: Name of the matplotlib colormap for bivariate densities.
        output_format: File format of the saved plots, either "svg" or "png".
    """

    def __init__(
        self,
        layout: Literal["pair_grid", "1d_marginals"] = "pair_grid",
        figure_size: tuple[float, float] | None = None,
        kde_grid_size: int = 50,
        kde_num_points: int = 500,
        contour_levels: int = 10,
        colormap: str = "Blues",
        output_format: Literal["svg", "png"] = "svg",
    ) -> None:
        """Initialize adaptive sampling visualization.

        Args:
            layout: Layout of the figure, either "pair_grid" or "1d_marginals".
            figure_size: Figure size (width, height) in inches. Grows with the number of
                parameters by default.
            kde_grid_size: Number of grid points per axis for bivariate densities.
            kde_num_points: Number of grid points for univariate densities.
            contour_levels: Number of contour levels for bivariate densities.
            colormap: Name of the matplotlib colormap for bivariate densities.
            output_format: File format of the saved plots, either "svg" or "png".
        """
        self.layout = layout
        self.figure_size = figure_size
        self.kde_grid_size = kde_grid_size
        self.kde_num_points = kde_num_points
        self.contour_levels = contour_levels
        self.colormap = colormap
        self.output_format = output_format

    def plot(
        self,
        results: AdaptiveSamplingResults,
        iteration: int,
        parameters: Parameters,
        global_settings: GlobalSettings,
    ) -> None:
        """Save the posterior plot of one iteration.

        The plot is saved as ``<experiment_name>_posterior_iteration_<iteration>`` in the output
        directory. A failing plot is logged as a warning, so that it does not stop the iterator.

        Args:
            results: Adaptive sampling results, as stored by the iterator.
            iteration: Index of the iteration in the results.
            parameters: Parameters of the iterator.
            global_settings: Global settings of the experiment.
        """
        kde_logger = logging.getLogger(GaussianKDE.__module__)
        kde_level = kde_logger.level
        # GaussianKDE logs its arguments on every initialization.
        kde_logger.setLevel(logging.WARNING)
        try:
            figure = self.posterior_figure(results, iteration, parameters)
            figure.savefig(
                global_settings.result_file(
                    suffix=f"_posterior_iteration_{iteration}", extension=self.output_format
                ),
                dpi=300,
                bbox_inches="tight",
            )
        # plotting must not stop the iterator
        except Exception:  # pylint: disable=broad-exception-caught
            _logger.warning(
                "Plotting iteration %i failed. Continuing without the plot.",
                iteration,
                exc_info=True,
            )
        finally:
            # restore the logging level
            kde_logger.setLevel(kde_level)

    def posterior_figure(
        self,
        results: AdaptiveSamplingResults,
        iteration: int,
        parameters: Parameters,
    ) -> Figure:
        """Create the posterior figure of one iteration.

        Args:
            results: Adaptive sampling results.
            iteration: Index of the iteration in the results.
            parameters: Parameters of the inverse problem.

        Returns:
            Posterior figure.
        """
        parameter_keys = parameters.parameters_keys
        num_parameters = parameters.num_parameters

        particles = results["particles"][iteration]
        weights = results["weights"][iteration]
        x_train = results["x_train"][iteration]
        x_train_new = results["x_train_new"][iteration]
        x_train_failed = results["x_train_failed"][iteration]
        log_likelihood = results["y_train"][iteration].reshape(-1)

        # max likelihood sample
        ml_index = np.argmax(log_likelihood)
        ml_sample = x_train[ml_index]
        # Values of the most likely sample, listed below its legend entry.
        ml_values = [
            f"log-likelihood = {log_likelihood[ml_index]:.4g}",
            *(f"{name} = {value:.4g}" for name, value in zip(parameter_keys, ml_sample)),
        ]

        # determine plot bounds containing all samples and particles
        bounds = _get_bounds(samples=np.vstack([particles, x_train, x_train_new, x_train_failed]))

        # Set up the figure layout
        if self.layout == "1d_marginals":
            # arrange in a square approximately
            columns = math.ceil(math.sqrt(num_parameters))
            rows = math.ceil(num_parameters / columns)
        else:
            rows = columns = num_parameters

        # Compute the figure size
        grid_width = 2.5 * columns
        grid_height = 2.5 * rows
        legend_width = 3.2
        title_height = 0.8
        figure_size = self.figure_size or (grid_width + legend_width, grid_height + title_height)

        # Create the figure and axes
        figure = Figure(figsize=figure_size, layout="compressed")
        axes = figure.subplots(
            rows, columns, squeeze=False, sharex="col" if self.layout == "pair_grid" else False
        )
        for panel in axes.flat:
            panel.set_box_aspect(1)

        # Univariate densities: on the diagonal of the pair grid, otherwise in reading order.
        if self.layout == "pair_grid":
            univariate_panels = list(axes.diagonal())
        else:
            univariate_panels = list(axes.ravel()[:num_parameters])
            # remove axes that are not needed
            for panel in axes.ravel()[num_parameters:]:
                panel.remove()

        # Plot the univariate densities onto the univariate panels
        for index, (panel, parameter_key) in enumerate(zip(univariate_panels, parameter_keys)):
            self._plot_univariate_posterior(
                axes=panel,
                values=particles[:, index],
                weights=weights,
                prior=parameters.dict.get(parameter_key),
                name=parameter_key,
                bounds=bounds[index],
                ml_sample_value=ml_sample[index],
                label=_posterior_label(dimension=1, num_parameters=num_parameters),
            )

        if self.layout == "pair_grid":
            # only in off-diagonal panels
            for row in range(num_parameters):
                for col in range(row):
                    self._plot_bivariate_posterior(
                        lower_axes=axes[row, col],
                        upper_axes=axes[col, row],
                        values=particles[:, [col, row]],
                        weights=weights,
                    )
                    # plot the training samples on top in the upper triangle
                    _plot_training_samples(
                        axes=axes[col, row],
                        x_train=x_train,
                        x_train_new=x_train_new,
                        x_train_failed=x_train_failed,
                        x_index=row,
                        y_index=col,
                    )
            _format_pairs_axes(axes=axes, parameter_keys=parameter_keys, bounds=bounds)

        figure.suptitle(
            f"Adaptive sampling posterior, iteration {iteration}",
            fontsize="x-large",
            fontweight="bold",
        )
        self._add_legend(figure=figure, ml_values=ml_values, num_parameters=num_parameters)
        return figure

    def _plot_univariate_posterior(
        self,
        axes: Axes,
        values: np.ndarray,
        weights: np.ndarray,
        prior: Any,
        name: str,
        bounds: np.ndarray,
        ml_sample_value: float,
        label: str,
    ) -> None:
        """Plot a univariate posterior density."""
        kde = GaussianKDE(values, weights=weights)
        density_bounds = _density_bounds(kde)
        density_grid = np.linspace(density_bounds[0, 0], density_bounds[0, 1], self.kde_num_points)
        density = kde.pdf(density_grid)
        axes.plot(density_grid, density, color="tab:blue", label=label)
        axes.fill_between(density_grid, density, color="tab:blue", alpha=0.3)

        grid = np.linspace(bounds[0], bounds[1], self.kde_num_points)
        prior_density = _prior_density(prior=prior, grid=grid)
        if prior_density is not None:
            axes.plot(grid, prior_density, color="tab:orange", label="Prior density")

        axes.axvline(
            ml_sample_value,
            color="black",
            linestyle="--",
            linewidth=1.2,
            label="Most likely evaluated sample",
        )
        axes.set_xlim(bounds[0], bounds[1])
        axes.set_xlabel(name)
        # Only the shape of the density matters.
        axes.set_ylim(bottom=0)
        axes.set_yticks([])

    def _plot_bivariate_posterior(
        self,
        lower_axes: Axes,
        upper_axes: Axes,
        values: np.ndarray,
        weights: np.ndarray,
    ) -> None:
        """Plot a bivariate posterior density."""
        kde = GaussianKDE(values, weights=weights)
        density_bounds = _density_bounds(kde)
        grid_x, grid_y = np.meshgrid(
            np.linspace(density_bounds[0, 0], density_bounds[0, 1], self.kde_grid_size),
            np.linspace(density_bounds[1, 0], density_bounds[1, 1], self.kde_grid_size),
        )
        density = kde.pdf(np.column_stack([grid_x.ravel(), grid_y.ravel()])).reshape(grid_x.shape)
        contours = lower_axes.contourf(
            grid_x,
            grid_y,
            density,
            levels=np.linspace(0.0, np.max(density), self.contour_levels + 1),
            cmap=self.colormap,
        )
        # The density grid may not cover the panel, so fill the rest with the lowest band's color.
        lower_axes.set_facecolor(tuple(np.asarray(contours.get_facecolor())[0]))
        upper_axes.contour(
            grid_y,
            grid_x,
            density,
            levels=3,
            colors="lightgrey",
            linewidths=0.6,
            zorder=0,
        )

    def _add_legend(self, figure: Figure, ml_values: list[str], num_parameters: int) -> None:
        """Add the legend of all panels to the right of the grid."""
        entries: dict[str, Any] = {}
        if self.layout == "pair_grid" and num_parameters > 1:
            # Add a legend entry for the contourf plots
            label = _posterior_label(dimension=2, num_parameters=num_parameters)
            entries[label] = tuple(
                Patch(color=color) for color in colormaps[self.colormap]([0.2, 0.45, 0.7, 0.95])
            )
        # Legend entries of all panels without duplicates.
        for axes in figure.get_axes():
            for handle, label in zip(*axes.get_legend_handles_labels()):
                if label in entries:
                    continue
                entries[label] = handle
                if label == "Most likely evaluated sample":
                    # Add the ML sample values as fake legend entries
                    for value in ml_values:
                        entries[value] = Line2D([], [], linestyle="none")
        figure.legend(
            list(entries.values()),
            list(entries),
            loc="outside right center",
            handler_map={tuple: HandlerTuple(ndivide=None, pad=0)},
        )


def _posterior_label(dimension: int, num_parameters: int) -> str:
    """Label of a univariate or bivariate posterior density.

    The density is a marginal if it leaves out parameters.
    """
    name = "Univariate" if dimension == 1 else "Bivariate"
    marginal = " marginal" if dimension < num_parameters else ""
    return f"{name}{marginal} posterior density"


def _density_bounds(kde: GaussianKDE) -> np.ndarray:
    """Determine bounds outside which a KDE is negligible.

    The bounds span four standard deviations around the mean, so that
    the density grid resolves narrow posteriors independently of the
    plot bounds.
    """
    width = 4.0 * np.sqrt(np.diag(kde.covariance))
    return np.column_stack([kde.mean - width, kde.mean + width])


def _prior_density(prior: Any, grid: np.ndarray) -> np.ndarray | None:
    """Evaluate a scalar continuous prior density, if there is one."""
    if not isinstance(prior, Continuous) or prior.dimension != 1:
        return None
    return prior.pdf(grid.reshape(-1, 1))


def _plot_training_samples(
    axes: Axes,
    x_train: np.ndarray,
    x_train_new: np.ndarray,
    x_train_failed: np.ndarray,
    x_index: int,
    y_index: int,
) -> None:
    """Plot the evaluated, next and failed training samples."""
    axes.scatter(
        x_train[:, x_index],
        x_train[:, y_index],
        marker=".",
        color="gray",
        alpha=0.6,
        s=15,
        label=f"Training samples ({len(x_train)})",
    )
    axes.scatter(
        x_train_new[:, x_index],
        x_train_new[:, y_index],
        marker="+",
        color="green",
        linewidth=1.5,
        s=60,
        label=f"Next training samples ({len(x_train_new)})",
    )
    axes.scatter(
        x_train_failed[:, x_index],
        x_train_failed[:, y_index],
        marker="x",
        color="red",
        linewidth=1.5,
        s=30,
        label=f"Failed training samples ({len(x_train_failed)})",
    )


def _format_pairs_axes(axes: np.ndarray, parameter_keys: list[str], bounds: np.ndarray) -> None:
    """Set the parameter axis limits and labels of the pair grid."""
    for index, name in enumerate(parameter_keys):
        # In row `index`, the off-diagonal panels show parameter `index` on the y-axis. The
        # diagonal panel shows a density instead.
        for col, panel in enumerate(axes[index]):
            if col != index:
                panel.set_ylim(bounds[index])
        # Label the outer panels only. Row 0 has the density panel in the first column.
        axes[-1, index].set_xlabel(name)
        if index > 0:
            axes[index, 0].set_ylabel(name)
    # Hide the tick labels and axis labels of inner panels.
    for panel in axes.flat:
        panel.label_outer()


def _get_bounds(samples: np.ndarray) -> np.ndarray:
    """Determine plot bounds containing all samples."""
    lower = np.nanmin(samples, axis=0)
    upper = np.nanmax(samples, axis=0)
    margin = 0.05 * (upper - lower)
    return np.column_stack([lower - margin, upper + margin])
