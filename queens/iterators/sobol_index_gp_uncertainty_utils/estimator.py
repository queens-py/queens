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
"""Helper classes for estimation of Sobol indices."""

import logging
import multiprocessing as mp
import time

import numpy as np
import xarray as xr

from queens.iterators.sobol_index_gp_uncertainty_utils.utils_estimate_indices import (
    calculate_indices_first_total_order,
    calculate_indices_second_order_gp_mean,
    calculate_indices_second_order_gp_realizations,
    calculate_indices_third_order,
)
from queens.utils.logger_settings import log_init_args

_logger = logging.getLogger(__name__)


class SobolIndexEstimator:
    """Sobol Index Estimator class.

    Attributes:
        number_bootstrap_samples (int): number of bootstrap samples
        number_parameters (int): number of input space dimensions
        number_monte_carlo_samples (int): number of Monte-Carlo samples
        number_gp_realizations (int): number of Gaussian process realizations
        calculate_second_order (bool): true if second-order indices are calculated
        estimates_first_order (xr.DataArray): first-order index estimates
        estimates_second_order (xr.DataArray): second-order index estimates
        estimates_total_order (xr.DataArray): total-order index estimates
        first_order_estimator (string): estimator for first-order Sobol indices
        parameter_names (list): list of parameter names
        seed_bootstrap_samples (int): seed for bootstrap samples
    """

    @log_init_args
    def __init__(
        self,
        parameter_names,
        calculate_second_order,
        number_monte_carlo_samples,
        number_gp_realizations,
        number_bootstrap_samples,
        seed_bootstrap_samples,
        first_order_estimator,
        estimates_first_order,
        estimates_second_order,
        estimates_total_order,
    ):
        """Initialize.

        Args:
            parameter_names (list): list of parameter names
            calculate_second_order (bool): true if second-order indices are calculated
            number_monte_carlo_samples (int): number of Monte-Carlo samples
            number_gp_realizations (int): number of Gaussian process realizations
            number_bootstrap_samples (int): number of bootstrap samples
            seed_bootstrap_samples (int): seed for bootstrap samples
            first_order_estimator (string): estimator for first-order Sobol indices
            estimates_first_order (xr.DataArray): first-order index estimates
            estimates_second_order (xr.DataArray): second-order index estimates
            estimates_total_order (xr.DataArray): total-order index estimates
        """
        self.parameter_names = parameter_names
        self.number_parameters = len(self.parameter_names)
        self.calculate_second_order = calculate_second_order
        self.number_monte_carlo_samples = number_monte_carlo_samples
        self.number_gp_realizations = number_gp_realizations
        self.number_bootstrap_samples = number_bootstrap_samples
        self.seed_bootstrap_samples = seed_bootstrap_samples
        self.first_order_estimator = first_order_estimator
        self.estimates_first_order = estimates_first_order
        self.estimates_second_order = estimates_second_order
        self.estimates_total_order = estimates_total_order

    @classmethod
    def from_config_create(cls, method_options, parameter_names):
        """Create estimator from problem description.

        Args:
            method_options (dict): dictionary with method options
            parameter_names (list): list of parameter names

        Returns:
            estimator: SobolIndexEstimator
        """
        number_monte_carlo_samples = method_options["number_monte_carlo_samples"]
        calculate_second_order = method_options.get("second_order", False)
        number_gp_realizations = method_options["number_gp_realizations"]
        number_bootstrap_samples = method_options["number_bootstrap_samples"]
        seed_bootstrap_samples = method_options.get("seed_bootstrap_samples", 42)
        _logger.info("Number of bootstrap samples = %i", number_bootstrap_samples)

        first_order_estimator = method_options.get("first_order_estimator", "Saltelli2010")
        _logger.info("First-order estimator %s", first_order_estimator)

        estimates_first_order, estimates_second_order, estimates_total_order = cls._init_dataset(
            number_gp_realizations,
            calculate_second_order,
            number_bootstrap_samples,
            parameter_names,
        )

        return cls(
            parameter_names=parameter_names,
            calculate_second_order=calculate_second_order,
            number_monte_carlo_samples=number_monte_carlo_samples,
            number_gp_realizations=number_gp_realizations,
            number_bootstrap_samples=number_bootstrap_samples,
            seed_bootstrap_samples=seed_bootstrap_samples,
            first_order_estimator=first_order_estimator,
            estimates_first_order=estimates_first_order,
            estimates_second_order=estimates_second_order,
            estimates_total_order=estimates_total_order,
        )

    def estimate(self, prediction, num_procs):
        """Estimate Sobol indices.

        Return estimate for first, second and total-order Sobol indices (for all realizations of
        the Gaussian process and all bootstrap samples).

        Args:
            prediction (xr.Array): prediction from Gaussian process
            num_procs (int): number of processors

        Returns:
            estimates (dict): dictionary of Sobol index estimates of different order
        """
        bootstrap_idx = self._draw_bootstrap_index()
        cross_parameter_names = self.parameter_names.copy()

        # start multiprocessing pool
        pool = mp.get_context("spawn").Pool(num_procs)

        for input_dim, parameter_name in enumerate(self.parameter_names):
            # adapt index so that for second-order indices redundant indices are not calculated
            # twice since S_ij == S_ji
            cross_parameter_names.remove(parameter_name)
            start_time = time.time()

            # calculate estimates in parallel (either over realizations or bootstrapping samples)
            estimate_function, input_list = self._setup_parallelization(
                prediction, input_dim, bootstrap_idx
            )
            raw_output = pool.starmap(estimate_function, input_list)

            # sort raw output from parallel processes
            self._sort_output(raw_output, parameter_name, cross_parameter_names)

            _logger.info("Time for parameter %s: %f", parameter_name, time.time() - start_time)

        pool.close()

        _logger.debug("First-order estimates: %s", self.estimates_first_order.values)
        _logger.debug("Total-order estimates: %s", self.estimates_total_order.values)

        estimates = {
            "first_order": self.estimates_first_order,
            "total_order": self.estimates_total_order,
            "second_order": self.estimates_second_order,
        }
        return estimates

    def _draw_bootstrap_index(self):
        """Draw random index for bootstrapping.

        Returns:
            bootstrap_idx (ndarray): index for bootstrapping
        """
        base = np.arange(self.number_monte_carlo_samples)
        rng = np.random.default_rng(seed=self.seed_bootstrap_samples)
        random = rng.integers(
            self.number_monte_carlo_samples,
            size=(self.number_bootstrap_samples - 1, self.number_monte_carlo_samples),
        )
        bootstrap_idx = np.concatenate((np.atleast_2d(base), random))
        _logger.debug("Bootstrap indices: %s", bootstrap_idx)
        return bootstrap_idx

    @classmethod
    def _init_dataset(
        cls,
        number_gp_realizations,
        calculate_second_order,
        number_bootstrap_samples,
        parameter_names,
    ):
        """Initialize data sets.

        Args:
            number_gp_realizations (int): number of Gaussian process realizations
            number_bootstrap_samples (int): number of bootstrap samples
            calculate_second_order (bool): true if second-order indices are calculated
            parameter_names (list): list of parameter names

        Returns:
            estimates_first_order (xr.DataArray): first-order index estimates
            estimates_second_order (xr.DataArray): second-order index estimates
            estimates_total_order (xr.DataArray): total-order index estimate
        """
        number_parameters = len(parameter_names)
        dimensions = ("gp_realization", "bootstrap", "parameter")
        coordinates = {
            "gp_realization": np.arange(number_gp_realizations),
            "bootstrap": np.arange(number_bootstrap_samples),
            "parameter": parameter_names,
        }

        estimates_first_order = xr.DataArray(
            data=np.empty((number_gp_realizations, number_bootstrap_samples, number_parameters)),
            dims=dimensions,
            coords=coordinates,
        )

        estimates_total_order = estimates_first_order.copy(deep=True)

        if calculate_second_order:
            data = np.empty(
                (
                    number_gp_realizations,
                    number_bootstrap_samples,
                    number_parameters,
                    number_parameters,
                )
            )
            data[:, :, :, :] = np.NaN
            estimates_second_order = xr.DataArray(
                data=data,
                dims=("gp_realization", "bootstrap", "parameter", "crossparameter"),
                coords={
                    "gp_realization": np.arange(number_gp_realizations),
                    "bootstrap": np.arange(number_bootstrap_samples),
                    "parameter": parameter_names,
                    "crossparameter": parameter_names,
                },
            )
        else:
            estimates_second_order = None

        return estimates_first_order, estimates_second_order, estimates_total_order

    def _setup_parallelization(self, prediction, input_dim, bootstrap_idx):
        """Setup parallelization for calculation of estimates.

        The parallelization scheme is chosen as follows:
        - If we sample realizations of the GP, parallelize over those realizations.
        - If we use the GP mean, parallelize over the bootstrap samples.

        Args:
            prediction (xr.DataArray): prediction
            input_dim (int): input parameter
            bootstrap_idx (ndarray): random index for bootstrapping

        Returns:
            estimate_function (obj): function object for estimate calculation
            input_list (list): list of input for estimate_function
        """
        if self.calculate_second_order:
            if self.number_gp_realizations == 1:
                estimate_function = calculate_indices_second_order_gp_mean
                input_list = [
                    (
                        prediction.data.squeeze(),
                        bootstrap_idx[b, :],
                        input_dim,
                        self.number_parameters,
                        self.first_order_estimator,
                    )
                    for b in np.arange(int(self.number_bootstrap_samples))
                ]
            else:
                estimate_function = calculate_indices_second_order_gp_realizations
                input_list = [
                    (
                        prediction.loc[{"gp_realization": k}].data,
                        bootstrap_idx,
                        input_dim,
                        self.number_bootstrap_samples,
                        self.number_parameters,
                        self.first_order_estimator,
                    )
                    for k in np.arange(self.number_gp_realizations)
                ]
        else:
            estimate_function = calculate_indices_first_total_order
            input_list = [
                (
                    prediction.loc[{"gp_realization": k}].data,
                    bootstrap_idx,
                    input_dim,
                    self.number_bootstrap_samples,
                    self.first_order_estimator,
                )
                for k in np.arange(self.number_gp_realizations)
            ]

        return estimate_function, input_list

    def _sort_output(self, raw_output, parameter_name, cross_parameter_names):
        """Sort raw output into DataArray.

        Args:
            raw_output (list): raw output data from parallel runs
            parameter_name (str): parameter name
            cross_parameter_names (list): cross-parameter names (for second-order indices)
        """
        if self.calculate_second_order and self.number_gp_realizations == 1:
            self._parallel_over_bootstrapping(raw_output, parameter_name, cross_parameter_names)
        else:
            self._parallel_over_realizations(raw_output, parameter_name, cross_parameter_names)

    def _parallel_over_bootstrapping(self, raw_output, i, j):
        """Output from parallelization over index b.

        Args:
            raw_output (list): raw output data from parallel runs
            i (str): index for S_i, S_Ti
            j (list): list of all index for S_ij (interactions with all other parameters)
        """
        for bootstrap_sample_id in np.arange(self.number_bootstrap_samples):
            output_index = {"parameter": i, "gp_realization": 0, "bootstrap": bootstrap_sample_id}
            self.estimates_first_order.loc[output_index] = raw_output[bootstrap_sample_id][0]
            self.estimates_total_order.loc[output_index] = raw_output[bootstrap_sample_id][1]

            if self.calculate_second_order:
                output_index = {
                    "parameter": i,
                    "gp_realization": 0,
                    "bootstrap": bootstrap_sample_id,
                    "crossparameter": j,
                }
                self.estimates_second_order.loc[output_index] = raw_output[bootstrap_sample_id][2]

    def _parallel_over_realizations(self, raw_output, i, j):
        """Output from parallelization over index k.

        Args:
            raw_output (list): raw output data from parallel runs
            i (str): index for S_i, S_Ti
            j (list): list of all index for S_ij (interactions with all other parameters)
        """
        for k in np.arange(self.number_gp_realizations):
            output_index = {"parameter": i, "gp_realization": k}
            self.estimates_first_order.loc[output_index] = raw_output[k][0]
            self.estimates_total_order.loc[output_index] = raw_output[k][1]

            if self.calculate_second_order:
                output_index = {"parameter": i, "gp_realization": k, "crossparameter": j}
                self.estimates_second_order.loc[output_index] = raw_output[k][2]


class SobolIndexEstimatorThirdOrder(SobolIndexEstimator):
    """SobolIndexEstimatorThirdOrder class.

    Attributes:
        calculate_third_order (bool): true if third-order indices only are calculated
        estimates_third_order (xr.DataArray): third-order index estimates
        third_order_parameters (list): list of parameter combination for third-order index
    """

    def __init__(
        self,
        parameter_names,
        calculate_second_order,
        number_monte_carlo_samples,
        number_gp_realizations,
        number_bootstrap_samples,
        first_order_estimator,
        calculate_third_order,
        third_order_parameters,
        seed_bootstrap_samples,
        estimates_first_order,
        estimates_total_order,
        estimates_second_order,
        estimates_third_order,
    ):
        """Initialize.

        Args:
            number_bootstrap_samples (int): number of bootstrap samples
            number_monte_carlo_samples (int): number of Monte-Carlo samples
            number_gp_realizations (int): number of Gaussian process realizations
            calculate_second_order (bool): true if second-order indices are calculated
            first_order_estimator (string): estimator for first-order Sobol indices
            parameter_names (list): list of parameter names
            calculate_third_order (bool): true if third-order indices only are calculated
            third_order_parameters (list): list of parameter combination for third-order index
            seed_bootstrap_samples (int): seed for bootstrap samples
            estimates_first_order (xr.DataArray): first-order index estimates
            estimates_second_order (xr.DataArray): second-order index estimates
            estimates_total_order (xr.DataArray): total-order index estimates
            estimates_third_order (xr.DataArray): third-order index estimates
        """
        super().__init__(
            parameter_names,
            calculate_second_order,
            number_monte_carlo_samples,
            number_gp_realizations,
            number_bootstrap_samples,
            seed_bootstrap_samples,
            first_order_estimator,
            estimates_first_order,
            estimates_second_order,
            estimates_total_order,
        )
        self.calculate_third_order = calculate_third_order
        self.third_order_parameters = third_order_parameters
        self.estimates_third_order = estimates_third_order

    @classmethod
    def from_config_create(cls, method_options, parameter_names):
        """Create estimator from problem description.

        Args:
            method_options (dict): dictionary with method options
            parameter_names (list): list of parameter names

        Returns:
            estimator: SobolIndexEstimatorThirdOrder
        """
        number_monte_carlo_samples = method_options["number_monte_carlo_samples"]
        calculate_second_order = method_options.get("second_order", False)
        number_gp_realizations = method_options["number_gp_realizations"]
        number_bootstrap_samples = method_options["number_bootstrap_samples"]
        seed_bootstrap_samples = method_options.get("seed_bootstrap_samples", 42)
        first_order_estimator = method_options.get("first_order_estimator", "Saltelli2010")

        calculate_third_order = method_options.get("third_order", False)
        third_order_parameters = method_options.get("third_order_parameters", None)

        estimates_first_order, estimates_total_order, estimates_second_order = cls._init_dataset(
            number_gp_realizations,
            calculate_second_order,
            number_bootstrap_samples,
            parameter_names,
        )

        estimates_third_order = cls._init_third_dataset(
            number_gp_realizations, number_bootstrap_samples
        )

        return cls(
            parameter_names=parameter_names,
            calculate_second_order=calculate_second_order,
            number_monte_carlo_samples=number_monte_carlo_samples,
            number_gp_realizations=number_gp_realizations,
            number_bootstrap_samples=number_bootstrap_samples,
            first_order_estimator=first_order_estimator,
            calculate_third_order=calculate_third_order,
            third_order_parameters=third_order_parameters,
            seed_bootstrap_samples=seed_bootstrap_samples,
            estimates_first_order=estimates_first_order,
            estimates_total_order=estimates_total_order,
            estimates_second_order=estimates_second_order,
            estimates_third_order=estimates_third_order,
        )

    def estimate(self, prediction, num_procs):
        """Estimate third-order Sobol index.

        Return estimate for third-order Sobol index (for all realizations of the Gaussian process
        and all bootstrap samples).

        Args:
            prediction (xr.Array): prediction from Gaussian process
            num_procs (int): number of processors

        Returns:
            estimates (dict): estimates for Sobol indices of different order
        """
        bootstrap_idx = self._draw_bootstrap_index()

        start_time = time.time()
        # calculate estimates in parallel over Gaussian process realizations
        estimate_function, input_list = self._setup_parallelization(prediction, 0, bootstrap_idx)

        # start multiprocessing pool
        pool = mp.get_context("spawn").Pool(num_procs)
        raw_output = pool.starmap(estimate_function, input_list)
        pool.close()

        # sort raw output from parallel processes
        self._sort_output(raw_output, "", [])

        _logger.info("Time for third-order indices: %f", time.time() - start_time)

        estimates = {
            "first_order": None,
            "total_order": None,
            "second_order": None,
            "third_order": self.estimates_third_order,
        }
        return estimates

    @classmethod
    def _init_third_dataset(cls, number_gp_realizations, number_bootstrap_samples):
        """Initialize data set for third-order indices.

        Args:
            number_gp_realizations (int): number of Gaussian process realizations
            number_bootstrap_samples (int): number of bootstrap samples

        Returns:
            estimates_third_order (xr.DataArray): third-order index estimates
        """
        dimensions = ("gp_realization", "bootstrap")
        coordinates = {
            "gp_realization": np.arange(number_gp_realizations),
            "bootstrap": np.arange(number_bootstrap_samples),
        }

        estimates_third_order = xr.DataArray(
            data=np.empty((number_gp_realizations, number_bootstrap_samples)),
            dims=dimensions,
            coords=coordinates,
        )
        return estimates_third_order

    def _setup_parallelization(self, prediction, input_dim, bootstrap_idx):
        """Setup parallelization for calculation of estimates.

        Args:
            prediction (xr.DataArray): prediction
            input_dim (int): input parameter
            bootstrap_idx (ndarray): random index for bootstrapping

        Returns:
            estimate_function (obj): function object for estimate calculation
            input_list (list): list of input for estimate_function
        """
        estimate_function = calculate_indices_third_order
        input_list = [
            (
                prediction.loc[{"gp_realization": k}].data,
                bootstrap_idx,
                self.number_bootstrap_samples,
                len(self.third_order_parameters),
                self.first_order_estimator,
            )
            for k in np.arange(self.number_gp_realizations)
        ]
        return estimate_function, input_list

    def _sort_output(self, raw_output, parameter_name, cross_parameter_names):
        """Sort raw output into DataArray.

        Args:
            raw_output (list): raw output data from parallel runs
            parameter_name (str): parameter name
            cross_parameter_names (list): cross-parameter names (for second-order indices)
        """
        for k in np.arange(self.number_gp_realizations):
            self.estimates_third_order.loc[{"gp_realization": k}] = raw_output[k]
