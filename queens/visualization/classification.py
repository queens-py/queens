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
"""Classification visualization."""

import itertools
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import sklearn.inspection._plot.decision_boundary
from matplotlib import cm


class ClassificationVisualization:
    """Visualization class for ClassificationIterator.

    Attributes:
        saving_path (Path): Path to save the plots
        plot_basename (str): Common basename for all plots
        save_bool (bool): Boolean to save plots
        plot_bool (bool): Boolean for determining whether individual plots should be plotted

    Returns:
        Instance of the visualization class
    """

    # some overall class states
    plt.rcParams["mathtext.fontset"] = "cm"

    def __init__(self, plotting_dir, plot_name, save_bool, plot_bool=False):
        """Initialise iterator.

        Args:
            plotting_dir (Path,str): Path to save the plots.
            plot_name (str): Common basename for all plots
            save_bool (bool): Boolean to save plots.
            plot_bool (bool): Boolean for determining whether individual plots should be plotted
        """
        self.saving_path = Path(plotting_dir)
        self.plot_basename = plot_name
        self.save_bool = save_bool
        self.plot_bool = plot_bool

    @classmethod
    def from_config_create(cls, plotting_options):
        """Create the visualization object from the problem description.

        Args:
            plotting_options (dict): Dictionary containing the options for plotting

        Returns:
            Instance of ClassificationVisualization
        """
        path = Path(plotting_options.get("plotting_dir"))
        plot_basename = plotting_options["plot_name"]
        save_bool = plotting_options.get("save_bool")
        plot_bool = plotting_options.get("plot_bool", False)
        return cls(
            plotting_dir=path, plot_name=plot_basename, save_bool=save_bool, plot_bool=plot_bool
        )

    def plot_decision_boundary(
        self, output, samples, classifier, parameter_names, iteration_index="final"
    ):
        """Plot decision boundary of the trained classifier.

        If num_params>2, each combination of 2 parameters is plotted in a separate subplot.

        Args:
            output (np.array): Classification results obtained from simulation
            samples (np.array): Array with sample points, size: (num_sample_points, num_params)
            classifier (obj): Classifier object from Queens
            parameter_names (list): List of parameters names
            iteration_index (str): additional name for saving plots
        """
        num_params = len(parameter_names)
        all_ind = set(np.arange(num_params))
        if self.plot_bool or self.save_bool:
            num_combinations, idx = self._get_axes(num_params)
            fig, axes = plt.subplots(figsize=(10 * num_combinations, 10), ncols=num_combinations)
            for i, params in enumerate(idx):
                axis = axes[i] if num_params > 2 else axes
                not_params = list(all_ind - set(params))
                ConditionalDecisionBoundaryDisplay.from_estimator(
                    classifier.classifier_obj,
                    samples,
                    eps=0.1 * np.abs(np.min(samples[:, params])),
                    params=params,
                    conditial_values=[
                        (ii, 0) for ii in not_params
                    ],  # currently all conditional values are set to 0
                    axes=axis,
                    cmap=cm.coolwarm,
                    alpha=0.8,
                    xlabel=parameter_names[params[0]],
                    ylabel=parameter_names[params[1]],
                    grid_resolution=200,
                )
                axis.scatter(
                    samples[:, params[0]],
                    samples[:, params[1]],
                    c=output[:].astype(np.int64),
                    cmap=cm.coolwarm,
                    s=20,
                    edgecolors="k",
                )
                title = f"Classification with model calls {len(samples)}"
                # Plot the samples of the last batch
                if classifier.is_active:
                    axes.scatter(
                        samples[-classifier.batch_size :, params[0]],
                        samples[-classifier.batch_size :, params[1]],
                        c="g",
                        s=100,
                        marker="x",
                        label="Last batch",
                    )
                    title += f", active learning iteration {iteration_index}"
                    axes.legend()
            fig.suptitle(title)
            if self.save_bool:
                self._save_plot(plot_name=f"_{iteration_index}.jpg")
            plt.close(fig)
        if self.plot_bool:
            plt.show()

    @staticmethod
    def _get_axes(num_params):
        """Get all parameter combinations for conditional plots.

        Args:
            num_params (int): Number of parameters varied

        Returns:
            number of combinations, list of parameter indexes
        """
        num_combinations = math.comb(num_params, 2)
        idx = itertools.combinations(range(num_params), 2)
        return num_combinations, idx

    def _save_plot(self, plot_name):
        """Save the plot to specified path.

        Args:
            save_bool (bool): Flag to decide whether saving option is triggered.
            plot_name (str): name of the plot (specifying the basename)

        Returns:
            Saved plot.
        """
        path = self.saving_path / (self.plot_basename + plot_name)
        plt.savefig(path, dpi=300)


def conditional_prediction_decorator(prediction_method, conditial_values):
    """Decorator for the estimators.

    Currently, the DecisionBoundary only creates grids in a 2d setting. Hence, the conditional fixed
    values need to be added.

    Example:
        4d classifier where the second input parameter is the conditional one and fixed to the
        valued 5 and the last one to -1.

        conditonal_values=[(1,5),(3,-1)]

    Args:
        prediction_method (fun): method to be decorated
        conditial_values (list): list of tuple to add the conditional values.

    Returns:
        the wrapped method.
    """

    def predict(*args):
        args = list(args)
        # The last argument is X_grid
        x_grid = args[-1]
        n_fill = len(x_grid)
        for parameter in conditial_values:
            # Add column in parameter[0] with value parameter[1]
            x_grid = np.insert(x_grid, parameter[0], parameter[1] * np.ones(n_fill), axis=1)
        args[-1] = x_grid
        return prediction_method(*args)

    return predict


def _check_boundary_response_method(estimator, response_method, _class_of_interest):
    """Get the classifier response function.

    We exploit this function to plot the conditional predictions by passing the conditional_values
    in response_method.

    Args:
        estimator (obj): Classifier
        response_method (list): conditional values
        _class_of_interest (int, float, bool, str): The class considered when plotting the decision

    Returns:
        fun: prediction method
    """
    prediction_method = estimator.predict

    if response_method is None:
        return prediction_method

    return conditional_prediction_decorator(prediction_method, response_method)


# Currently sklearn only supports plotting 2 features. This overwrites a sanity check (hacky).
# pylint: disable=protected-access
sklearn.inspection._plot.decision_boundary._num_features = lambda X: 2
sklearn.inspection._plot.decision_boundary._check_boundary_response_method = (
    _check_boundary_response_method
)
sklearn.utils._response._check_response_method = lambda estimator, response_method: response_method
DecisionBoundaryDisplay = sklearn.inspection._plot.decision_boundary.DecisionBoundaryDisplay
# pylint: enable=protected-access


class ConditionalDecisionBoundaryDisplay(DecisionBoundaryDisplay):
    """Custom decision boundary display class."""

    @classmethod
    def from_estimator(
        cls,
        estimator,
        X,
        *,
        params=None,
        conditial_values=None,
        grid_resolution=100,
        eps=1.0,
        plot_method="contourf",
        xlabel=None,
        ylabel=None,
        axes=None,
        **kwargs,
    ):
        """Create boundary from classifier.

        Example:
            4d classifier where the second input parameter is the conditional one and fixed to the
            valued 5 and the last one to -1.

            conditonal_values=[(1,5),(3,-1)]
            params=(0,2)

        Args:
            estimator (obj): Trained classifier
            X (array): Input (n_samples, n_params)
            params (tuple, optional): List of paramter indices of the input dimensions to
                                      plot. Defaults to (0, 1).
            conditial_values (list, optional): List of tuple with the conditional values.
                                               Defaults to None.
            grid_resolution (int, optional): Number of grid points to use for plotting decision
                                             boundary. Higher values will make the plot look nicer
                                             but be slower to render. Defaults to 100.
            eps (float, optional): Extends the minimum and maximum values of X for evaluating the
                                   response function. Defaults to 1.0.
            plot_method (str, optional):  matplotlib plot methods. Defaults to "contourf".
            xlabel (str, optional): Label for the x-axis. Defaults to None.
            ylabel (str, optional): Label for the y-axis. Defaults to None.
            axes (Matplotlib axes, optional): Axes object to plot on. If `None`, a new figure and
                                            axes is created. Defaults to None.
            kwargs: Additional keyword arguments for DecisionBoundaryDisplay parent class

        Returns:
            sklearn.inspection.DecisionBoundaryDisplay object
        """
        n_params = X.shape[1]
        if n_params < 2:
            raise ValueError("X has to be at least 2d!")
        if n_params > 2:
            # If the conditional is to be plotted.
            # Check if everything is set.
            if len(params) + len(conditial_values) != n_params:
                raise ValueError(
                    "Either the number of params or conditional values is not defined correctly!"
                )

        # Return the original from_estimator with the modifications in the estimator
        return super().from_estimator(
            estimator,
            X,
            grid_resolution=grid_resolution,
            eps=eps,
            plot_method=plot_method,
            response_method=conditial_values,
            xlabel=xlabel,
            ylabel=ylabel,
            ax=axes,
            **kwargs,
        )
