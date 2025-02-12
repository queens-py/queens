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
"""Helper classes for statistics of Sobol index estimates."""

import logging

import numpy as np
import pandas as pd
from scipy.stats import norm

from queens.utils.logger_settings import log_init_args

_logger = logging.getLogger(__name__)


class StatisticsSobolIndexEstimates:
    """Statistics class for Sobol index estimates.

    Calculate statistics for the first- or total order Sobol index estimates, including mean,
    variances and confidence intervals of the Sobol index estimates.

    Based on
    Le Gratiet, Loic, Claire Cannamela, and Bertrand Iooss. ‘A Bayesian Approach
    for Global Sensitivity Analysis of (Multifidelity) Computer Codes’. SIAM/ASA Journal on
    Uncertainty Quantification 2, no. 1 (1 January 2014): 336–63. https://doi.org/10.1137/130926869.

    Attributes:
        number_bootstrap_samples (int): number of bootstrap samples
        number_parameters (int): number of input parameter space dimensions
        number_gp_realizations (int): number of Gaussian process realizations
        parameter_names (list): list of parameter names
    """

    @log_init_args
    def __init__(self, parameter_names, number_gp_realizations, number_bootstrap_samples):
        """Initialize.

        Args:
            parameter_names (list): list of parameter names
            number_gp_realizations (int): number of metamodel realizations
            number_bootstrap_samples (int): number of bootstrap samples
        """
        self.parameter_names = parameter_names
        self.number_parameters = len(self.parameter_names)
        self.number_gp_realizations = number_gp_realizations
        self.number_bootstrap_samples = number_bootstrap_samples

    @classmethod
    def from_config_create(cls, method_options, parameter_names):
        """Create statistics from problem description.

        Args:
            method_options (dict): dictionary with method options
            parameter_names (list): list of parameter names

        Returns:
            statistics: Statistics object
        """
        number_gp_realizations = method_options["number_gp_realizations"]
        number_bootstrap_samples = method_options["number_bootstrap_samples"]
        return cls(
            parameter_names=parameter_names,
            number_gp_realizations=number_gp_realizations,
            number_bootstrap_samples=number_bootstrap_samples,
        )

    def evaluate(self, estimates, conf_level=0.95):
        """Evaluate statistics.

        Args:
            estimates (xr.DataArray): Sobol index estimates
            conf_level (float): confidence level (default: 0.95)
        """
        result = self._init_result()
        for current_parameter in self.parameter_names:
            current_data = estimates.loc[{"parameter": current_parameter}]

            self._overall_mean(result, current_data, current_parameter)
            self._total_variance(result, current_data, current_parameter)
            self._gp_variance(result, current_data, current_parameter)
            self._monte_carlo_variance(result, current_data, current_parameter)

        self._confidence_bounds(result, conf_level)
        _logger.info(str(result))

        return result

    @staticmethod
    def _overall_mean(result, current_data, current_parameter):
        """Calculate the overall mean of the Sobol index estimates.

        Args:
            result (DataFrame): Sobol index result
            current_data (xr.DataArray): Sobol index estimates for current_parameter
            current_parameter (str): name of current parameter
        """
        result.loc[current_parameter, "Sobol_index"] = current_data.mean()

    def _total_variance(self, result, current_data, current_parameter):
        """Calculate the total variance of the Sobol index estimates.

        The total variance of the Sobol index estimate includes the variance due to Monte-Carlo
        integration and the variance due to the use of Gaussian process as surrogate model.

        Args:
            result (DataFrame): Sobol index result
            current_data (xr.DataArray): Sobol index estimates for current_parameter
            current_parameter (str): name of current parameter
        """
        result.loc[current_parameter, "var_total"] = np.sum(
            (current_data - result.loc[current_parameter, "Sobol_index"]) ** 2
        ) / (self.number_gp_realizations * self.number_bootstrap_samples - 1)

    def _gp_variance(self, result, current_data, current_parameter):
        """Calculate the variance due to the use of a Gaussian process.

        Args:
            result (DataFrame): Sobol index result
            current_data (xr.DataArray): Sobol index estimates for current_parameter
            current_parameter (str): name of current parameter
        """
        si_mean_gp = current_data.mean(dim="gp_realization")
        result.loc[current_parameter, "var_gp"] = np.sum((current_data - si_mean_gp) ** 2) / (
            (self.number_gp_realizations - 1) * self.number_bootstrap_samples
        )

    def _monte_carlo_variance(self, result, current_data, current_parameter):
        """Calculate the variance due to Monte-Carlo integration.

        Args:
            result (DataFrame): Sobol index result
            current_data (xr.DataArray): Sobol index estimates for current_parameter
            current_parameter (str): name of current parameter
        """
        mean_monte_carlo = current_data.mean(dim="bootstrap")
        result.loc[current_parameter, "var_monte_carlo"] = np.sum(
            (current_data - mean_monte_carlo) ** 2
        ) / ((self.number_bootstrap_samples - 1) * self.number_gp_realizations)

    @staticmethod
    def _confidence_bounds(result, conf_level):
        """Calculate confidence bounds from variances.

        Args:
            result (DataFrame): Sobol index result
            conf_level (float): confidence level (default: 0.95)
        """
        result.loc[:, ["conf_total", "conf_gp", "conf_monte_carlo"]] = np.sqrt(
            result[["var_total", "var_gp", "var_monte_carlo"]].values
        ) * norm.ppf(0.5 + conf_level / 2)

    def _init_result(self):
        """Initialize the dataset.

        Returns:
            result (DataFrame): Sobol index result
        """
        columns = [
            "Sobol_index",
            "var_total",
            "var_gp",
            "var_monte_carlo",
            "conf_total",
            "conf_gp",
            "conf_monte_carlo",
        ]
        result = pd.DataFrame(
            data=np.empty((self.number_parameters, len(columns))),
            columns=columns,
            index=self.parameter_names,
        )

        return result


class StatisticsSecondOrderEstimates(StatisticsSobolIndexEstimates):
    """Statistics class for second-order Sobol index estimates.

    Calculate statistics for the second-order Sobol index estimates,
    including mean, variances and confidence intervals of the estimates.
    """

    def evaluate(self, estimates, conf_level=0.95):
        """Evaluate statistics.

        Args:
            estimates (xr.DataArray): first-order index estimates
            conf_level (float): confidence level (default: 0.95)
        """
        result = self._init_result()
        for current_parameter in self.parameter_names:
            for current_cross_parameter in self.parameter_names:
                current_data = estimates.loc[
                    {"parameter": current_parameter, "crossparameter": current_cross_parameter}
                ]
                if not np.any(np.isnan(current_data.values)):
                    parameter_pair = current_parameter + "#" + current_cross_parameter
                    self._overall_mean(result, current_data, parameter_pair)
                    self._total_variance(result, current_data, parameter_pair)
                    self._gp_variance(result, current_data, parameter_pair)
                    self._monte_carlo_variance(result, current_data, parameter_pair)

        self._confidence_bounds(result, conf_level)
        _logger.info(str(result))

        return result

    def _init_result(self):
        """Initialize the dataset.

        Set up unique parameter-pair labels for all interaction effects,
        e.g. for three parameters ['x1#x2', 'x1#x3', 'x2#x3']. Redundant
        labels are not included twice since S_ij == S_ji.
        """
        parameter_pairs = []
        for parameter in self.parameter_names:
            for crossparameter in self.parameter_names:
                if parameter != crossparameter:
                    if not crossparameter + "#" + parameter in parameter_pairs:
                        parameter_pairs.append(parameter + "#" + crossparameter)

        columns = [
            "Sobol_index",
            "var_total",
            "var_gp",
            "var_monte_carlo",
            "conf_total",
            "conf_gp",
            "conf_monte_carlo",
        ]
        result = pd.DataFrame(
            data=np.empty((len(parameter_pairs), len(columns))),
            columns=columns,
            index=parameter_pairs,
        )

        return result


class StatisticsThirdOrderSobolIndexEstimates(StatisticsSobolIndexEstimates):
    """Statistics class for third-order Sobol index estimates.

    Calculate statistics for the third-order Sobol index estimates, including mean,
    variances and confidence intervals of the estimates.

    Attributes:
        third_order_parameters (list): list of parameter combination for third-order index
    """

    def __init__(
        self,
        parameter_names,
        number_gp_realizations,
        number_bootstrap_samples,
        third_order_parameters,
    ):
        """Initialize.

        Args:
            parameter_names (list): list of parameter names
            number_gp_realizations (int): number of metamodel realizations
            number_bootstrap_samples (int): number of bootstrap samples
            third_order_parameters (list): list of parameter combination for third-order index
        """
        self.third_order_parameters = third_order_parameters
        super().__init__(
            parameter_names=parameter_names,
            number_gp_realizations=number_gp_realizations,
            number_bootstrap_samples=number_bootstrap_samples,
        )

    @classmethod
    def from_config_create(cls, method_options, parameter_names):
        """Create statistics from problem description.

        Args:
            method_options (dict): dictionary with method options
            parameter_names (list): list of parameter names

        Returns:
            statistics: StatisticsThirdOrder object
        """
        number_gp_realizations = method_options["number_gp_realizations"]
        number_bootstrap_samples = method_options["number_bootstrap_samples"]
        third_order_parameters = method_options.get("third_order_parameters", None)

        return cls(
            parameter_names=parameter_names,
            number_gp_realizations=number_gp_realizations,
            number_bootstrap_samples=number_bootstrap_samples,
            third_order_parameters=third_order_parameters,
        )

    def evaluate(self, estimates, conf_level=0.95):
        """Evaluate statistics.

        Args:
            estimates (xr.DataArray): Sobol index estimates
            conf_level (float): confidence level (default: 0.95)
        """
        result = self._init_result()
        self._overall_mean(result, estimates, self.third_order_parameters[0])
        self._total_variance(result, estimates, self.third_order_parameters[0])
        self._gp_variance(result, estimates, self.third_order_parameters[0])
        self._monte_carlo_variance(result, estimates, self.third_order_parameters[0])

        self._confidence_bounds(result, conf_level)
        _logger.info(str(result))

        return result

    def _init_result(self):
        """Initialize the dataset."""
        columns = [
            "Sobol_index",
            "var_total",
            "var_gp",
            "var_monte_carlo",
            "conf_total",
            "conf_gp",
            "conf_monte_carlo",
        ]
        result = pd.DataFrame(
            data=np.empty((1, len(columns))),
            columns=columns,
            index=[self.third_order_parameters[0]],
        )

        return result
