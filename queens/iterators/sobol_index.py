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
"""Estimate Sobol indices."""

import logging

import numpy as np
import pandas as pd
import plotly.graph_objs as go
from SALib.analyze import sobol
from SALib.sample import saltelli

from queens.distributions import lognormal, normal, uniform
from queens.iterators.iterator import Iterator
from queens.utils.logger_settings import log_init_args
from queens.utils.process_outputs import write_results

_logger = logging.getLogger(__name__)


# TODO deal with non-uniform input distribution # pylint: disable=fixme
class SobolIndex(Iterator):
    """Sobol Index Iterator.

    This class essentially provides a wrapper around the SALib library.

    Attributes:
        seed (int): Seed for random number generator.
        num_samples (int): Number of samples.
        calc_second_order (bool): Whether to calculate second-order sensitivities.
        skip_values (int or None): Number of points in Sobol' sequence to skip, ideally a value of
                                   base 2 (default: 1024).
        num_bootstrap_samples (int): Number of bootstrap samples for confidence intervals.
        confidence_level (float): Confidence level for the intervals.
        result_description (dict): Description of the desired results.
        samples (np.array): Samples used for analysis.
        output (dict): Model outputs corresponding to samples.
        salib_problem (dict): Problem definition for SALib.
        num_params (int): Number of parameters.
        parameter_names (list): List of parameter names.
        sensitivity_indices (dict): Sensitivity indices from Sobol analysis.
    """

    @log_init_args
    def __init__(
        self,
        model,
        parameters,
        global_settings,
        seed,
        num_samples,
        calc_second_order,
        num_bootstrap_samples,
        confidence_level,
        result_description,
        skip_values=None,
    ):
        """Initialize Saltelli SALib iterator object.

        Args:
            model (Model): Model to be evaluated by iterator.
            parameters (Parameters): Parameters object.
            global_settings (GlobalSettings): Experiment settings with name and output directory.
            seed (int): Seed for random number generation.
            num_samples (int): Number of random samples.
            calc_second_order (bool): Whether to calculate second-order sensitivities.
            num_bootstrap_samples (int): Number of bootstrap samples.
            confidence_level (float): Confidence level for intervals.
            result_description (dict): Description of the desired results.
            skip_values (int or None): Number of points in Sobol' sequence to skip, ideally a value
                                       of base 2. None triggers the SALib default value:
                                       a power of 2 >= N, or 16; whichever is greater.
                                       (default: None).
        """
        super().__init__(model, parameters, global_settings)

        self.seed = seed
        self.num_samples = num_samples
        self.calc_second_order = calc_second_order
        self.skip_values = skip_values
        self.num_bootstrap_samples = num_bootstrap_samples
        self.confidence_level = confidence_level
        self.result_description = result_description

        self.samples = None
        self.output = None
        self.salib_problem = None
        self.num_params = self.parameters.num_parameters
        self.parameter_names = self.parameters.names
        self.sensitivity_indices = None

    def pre_run(self):
        """Generate samples for subsequent analysis and update model."""
        # setup SALib problem dict

        if self.parameters.random_field_flag:
            raise RuntimeError(
                "The SaltelliIterator does not work in conjunction with random fields."
            )
        distribution_types, distribution_parameters = extract_parameters_of_parameter_distributions(
            self.parameters
        )

        self.salib_problem = {
            "num_vars": self.parameters.num_parameters,
            "names": self.parameters.names,
            "bounds": distribution_parameters,
            "dists": distribution_types,
        }

        _logger.info("Draw %s samples...", self.num_samples)
        self.samples = saltelli.sample(
            self.salib_problem,
            self.num_samples,
            calc_second_order=self.calc_second_order,
            skip_values=self.skip_values,
        )
        _logger.debug(self.samples)

    def get_all_samples(self):
        """Return all samples."""
        return self.samples

    def core_run(self):
        """Run Analysis on model."""
        _logger.info("Evaluate model...")
        self.output = self.model.evaluate(self.samples)

        _logger.info("Calculate Sensitivity Indices...")
        self.sensitivity_indices = sobol.analyze(
            self.salib_problem,
            np.reshape(self.output["result"], (-1)),
            calc_second_order=self.calc_second_order,
            num_resamples=self.num_bootstrap_samples,
            conf_level=self.confidence_level,
            print_to_console=False,
            seed=self.seed,
        )

    def post_run(self):
        """Analyze the results."""
        results = self.process_results()
        if self.result_description is not None:
            if self.result_description["write_results"]:
                write_results(results, self.global_settings.result_file(".pickle"))
            self.print_results(results)
            if self.result_description["plot_results"]:
                self.plot_results(results)

    def print_results(self, results):
        """Print results.

        Args:
            results (dict): Dictionary with Sobol indices and confidence intervals
        """
        sensitivity_indices = results["sensitivity_indices"]
        parameter_names = results["parameter_names"]

        additivity = np.sum(sensitivity_indices["S1"])
        higher_interactions = 1 - additivity
        sensitivity_indices_df = pd.DataFrame(
            {
                key: value
                for (key, value) in sensitivity_indices.items()
                if key not in ["S2", "S2_conf"]
            },
            index=parameter_names,
        )
        _logger.info("Main and Total Effects:")
        _logger.info(sensitivity_indices_df)
        _logger.info("Additivity, sum of main effects (for independent variables):")
        _logger.info("S_i = %s", additivity)

        if self.calc_second_order:
            second_order_indices = sensitivity_indices["S2"]
            second_order_indices_conf = sensitivity_indices["S2_conf"]

            for j in range(self.parameters.num_parameters):
                second_order_indices[j, j] = sensitivity_indices["S1"][j]
                for k in range(j + 1, self.parameters.num_parameters):
                    second_order_indices[k, j] = second_order_indices[j, k]
                    second_order_indices_conf[k, j] = second_order_indices_conf[j, k]

            second_order_indices_df = pd.DataFrame(
                second_order_indices, columns=parameter_names, index=parameter_names
            )
            second_order_indices_conf_df = pd.DataFrame(
                second_order_indices_conf, columns=parameter_names, index=parameter_names
            )

            _logger.info("Second Order Indices (diagonal entries are main effects):")
            _logger.info(second_order_indices_df)
            _logger.info("Confidence Second Order Indices:")
            _logger.info(second_order_indices_conf_df)

            # we extract the upper triangular matrix which includes the diagonal entries
            # therefore we have to subtract the trace
            second_order_interactions = np.sum(np.triu(second_order_indices, k=1))
            higher_interactions = higher_interactions - second_order_interactions
            str_second_order_interactions = f"S_ij = {second_order_interactions}"

            _logger.info("Sum of second order interactions:")
            _logger.info(str_second_order_interactions)

            str_higher_order_interactions = f"1 - S_i - S_ij = {higher_interactions}"
        else:
            str_higher_order_interactions = f"1 - S_i = {higher_interactions}"

        _logger.info("Higher order interactions:")
        _logger.info(str_higher_order_interactions)

    def process_results(self):
        """Write all results to self contained dictionary.

        Returns:
            results (dict): Dictionary with Sobol indices and confidence intervals
        """
        results = {
            "parameter_names": self.parameters.names,
            "sensitivity_indices": self.sensitivity_indices,
            "second_order": self.calc_second_order,
            "samples": self.samples,
            "output": self.output,
        }

        return results

    def plot_results(self, results):
        """Create bar graph of first order sensitivity indices.

        Args:
            results (dict): Dictionary with Sobol indices and confidence intervals
        """
        # Plot first-order indices also called main effect
        chart_path = self.global_settings.result_file(suffix="_S1", extension=".html")
        bars = go.Bar(
            x=results["parameter_names"],
            y=results["sensitivity_indices"]["S1"],
            error_y={
                "type": "data",
                "array": results["sensitivity_indices"]["S1_conf"],
                "visible": True,
            },
        )
        data = [bars]

        layout = {
            "title": "First-Order Sensitivity Indices",
            "xaxis": {"title": "Parameter"},
            "yaxis": {"title": "Main Effect"},
        }

        fig = go.Figure(data=data, layout=layout)
        fig.write_html(chart_path)

        # Plot total indices also called total effect
        chart_path = self.global_settings.result_file(suffix="_ST", extension=".html")
        bars = go.Bar(
            x=results["parameter_names"],
            y=results["sensitivity_indices"]["ST"],
            error_y={
                "type": "data",
                "array": results["sensitivity_indices"]["ST_conf"],
                "visible": True,
            },
        )
        data = [bars]

        layout = {
            "title": "Total Sensitivity Indices",
            "xaxis": {"title": "Parameter"},
            "yaxis": {"title": "Total Effect"},
        }

        fig = go.Figure(data=data, layout=layout)
        fig.write_html(chart_path)

        # Plot second order indices (if applicable)
        if self.calc_second_order:
            second_order_indices = results["sensitivity_indices"]["S2"]
            second_order_indices = second_order_indices[np.triu_indices(self.num_params, k=1)]

            second_order_indices_conf = results["sensitivity_indices"]["S2_conf"]
            second_order_indices_conf = second_order_indices_conf[
                np.triu_indices(self.num_params, k=1)
            ]

            # build list of names of second-order indices
            names = []
            for i in range(1, self.num_params + 1):
                for j in range(i + 1, self.num_params + 1):
                    names.append(f"S{i}{j}")

            chart_path = self.global_settings.result_file(suffix="_S2", extension=".html")
            bars = go.Bar(
                x=names,
                y=second_order_indices,
                error_y={"type": "data", "array": second_order_indices_conf, "visible": True},
            )
            data = [bars]

            layout = {
                "title": "Second Order Sensitivity Indices",
                "xaxis": {"title": "Parameter"},
                "yaxis": {"title": "Second Order Effects"},
            }

            fig = go.Figure(data=data, layout=layout)
            fig.write_html(chart_path)


def extract_parameters_of_parameter_distributions(parameters):
    """Extract the parameters of the parameter distributions.

    Args:
        parameters (Parameters): QUEENS Parameters object containing the metadata
    Returns:
        distribution_types (list): list with distribution types of the parameter distributions
        distribution_parameters (list): list with parameters of the parameter distributions
    """
    distribution_types = []
    distribution_parameters = []
    for parameter in parameters.dict.values():
        if isinstance(parameter, uniform.Uniform):
            upper_bound = parameter.upper_bound
            lower_bound = parameter.lower_bound
            distribution_name = "unif"
        # in queens normal distributions are parameterized with mean and var
        # in salib normal distributions are parameterized via mean and std
        # -> we need to reparameterize normal distributions
        elif isinstance(parameter, normal.Normal):
            lower_bound = parameter.mean.squeeze()
            upper_bound = np.sqrt(parameter.covariance.squeeze())
            distribution_name = "norm"
        elif isinstance(parameter, lognormal.LogNormal):
            lower_bound = parameter.mu.squeeze()
            upper_bound = parameter.sigma.squeeze()
            distribution_name = "lognorm"
        else:
            raise ValueError("Valid distributions are normal, lognormal and uniform!")

        distribution_types.append(distribution_name)
        distribution_parameters.append([lower_bound, upper_bound])

    return distribution_types, distribution_parameters
