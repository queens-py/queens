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
"""Helper class for generating Monte-Carlo samples for Sobol indices."""

import logging

import numpy as np
import xarray as xr
from scipy.stats import qmc

from queens.utils.logger_settings import log_init_args

_logger = logging.getLogger(__name__)


class Sampler:
    """Sampler class.

    Draw Monte-Carlo samples and generate sample matrices A, B, AB, BA following

    Saltelli, A., et al. ‘Variance Based Sensitivity Analysis of Model Output. Design and
    Estimator for the Total Sensitivity Index’. Computer Physics Communications 181, no. 2
    (1 February 2010): 259–270.
    https://doi.org/10.1016/j.cpc.2009.09.018.

    Saltelli, A., ed. Global Sensitivity Analysis: The Primer. Chichester, England; Hoboken, NJ:
    John Wiley & Sons, Ltd., 2008.
    https://doi.org/10.1002/9780470725184.

    Attributes:
        number_parameters (int): number of input space dimensions
        number_monte_carlo_samples (int): number of Monte-Carlo samples
        sampling_approach (string): sampling approach (pseudo-random or quasi-random)
        calculate_second_order (bool): true if second-order indices are calculated
        parameters (list): information about distribution of random variables
        parameter_names (list): list of names of input parameters
        seed_monte_carlo (int): seed for random samples
    """

    @log_init_args
    def __init__(
        self,
        parameter_names,
        number_monte_carlo_samples,
        seed_monte_carlo,
        calculate_second_order,
        parameters,
        sampling_approach,
    ):
        """Initialize.

        Args:
            parameter_names (list): list of names of input parameters
            number_monte_carlo_samples (int): number of Monte-Carlo samples
            seed_monte_carlo (int): seed for random samples
            calculate_second_order (bool): true if second-order indices are calculated
            parameters (list): information about distribution of random variables
            sampling_approach (string): sampling approach (pseudo-random or quasi-random)
        """
        self.parameter_names = parameter_names
        self.number_parameters = len(self.parameter_names)
        self.number_monte_carlo_samples = number_monte_carlo_samples
        self.seed_monte_carlo = seed_monte_carlo
        self.calculate_second_order = calculate_second_order
        self.parameters = parameters
        self.sampling_approach = sampling_approach

    @classmethod
    def from_config_create(cls, method_options, parameter_names, parameters):
        """Create sampler from problem description.

        Args:
            method_options (dict): dictionary with method options
            parameter_names (list): list of parameter names
            parameters (list): information about distribution of random variables

        Returns:
            sampler: Sampler object
        """
        number_monte_carlo_samples = method_options["number_monte_carlo_samples"]
        _logger.info("Number of Monte-Carlo samples = %i", number_monte_carlo_samples)

        seed_monte_carlo = method_options.get("seed_monte_carlo", 42)

        calculate_second_order = method_options.get("second_order", False)
        sampling_approach = method_options.get("sampling_approach", "quasi_random")

        return cls(
            parameter_names=parameter_names,
            number_monte_carlo_samples=number_monte_carlo_samples,
            seed_monte_carlo=seed_monte_carlo,
            calculate_second_order=calculate_second_order,
            parameters=parameters,
            sampling_approach=sampling_approach,
        )

    def sample(self):
        """Generate sample matrices.

        Generate sample matrices A, B, AB, BA following

        Saltelli, A., et al. ‘Variance Based Sensitivity Analysis of Model Output. Design and
        Estimator for the Total Sensitivity Index’. Computer Physics Communications 181, no. 2
        (1 February 2010): 259–270.
        https://doi.org/10.1016/j.cpc.2009.09.018.

        Saltelli, A., ed. Global Sensitivity Analysis: The Primer. Chichester, England; Hoboken, NJ:
        John Wiley & Sons, Ltd., 2008.
        https://doi.org/10.1002/9780470725184.

        sample_matrix_a:  sample matrix A
        sample_matrix_b:  sample matrix B
        sample_matrix_ab:  separated from A by only change in X_i; from B by change in X_{~i}
        sample_matrix_ba:  separated from B by only change in X_i; from A by change in X_{~i}

        Returns:
            samples (xr.Array): Monte-Carlo samples
        """
        samples = self._init_samples()

        sample_matrix_a, sample_matrix_b = self._draw_base_samples()
        samples.loc[{"sample_matrix": "A"}] = sample_matrix_a
        samples.loc[{"sample_matrix": "B"}] = sample_matrix_b

        for i, parameter_name in enumerate(self.parameter_names):
            sample_matrix_ab = sample_matrix_a.copy()
            sample_matrix_ab[:, i] = sample_matrix_b[:, i].flatten()
            samples.loc[{"sample_matrix": "AB_" + parameter_name}] = sample_matrix_ab
            if self.calculate_second_order:
                sample_matrix_ba = sample_matrix_b.copy()
                sample_matrix_ba[:, i] = sample_matrix_a[:, i].flatten()
                samples.loc[{"sample_matrix": "BA_" + parameter_name}] = sample_matrix_ba

        return samples

    def _init_samples(self):
        """Initialize samples data-array.

        Returns:
            samples (xr.DataArray): Monte-Carlo samples (sample_matrix x M x D)
        """
        labels_sample_matrices = self._setup_sample_matrices_labels()
        dimensions = ("monte_carlo", "sample_matrix", "parameter")
        coordinates = {
            "monte_carlo": np.arange(self.number_monte_carlo_samples),
            "sample_matrix": labels_sample_matrices,
            "parameter": self.parameter_names,
        }

        samples = xr.DataArray(
            data=np.empty(
                (
                    self.number_monte_carlo_samples,
                    len(labels_sample_matrices),
                    self.number_parameters,
                )
            ),
            dims=dimensions,
            coords=coordinates,
        )

        return samples

    def _setup_sample_matrices_labels(self):
        """Set up labels of sample matrices.

        For further information on the sample matrices see

        Saltelli, A., et al. ‘Variance Based Sensitivity Analysis of Model Output. Design and
        Estimator for the Total Sensitivity Index’. Computer Physics Communications 181, no. 2
        (1 February 2010): 259–270.
        https://doi.org/10.1016/j.cpc.2009.09.018.

        Saltelli, A., ed. Global Sensitivity Analysis: The Primer. Chichester, England; Hoboken, NJ:
        John Wiley & Sons, Ltd., 2008.
        https://doi.org/10.1002/9780470725184.

        Examples:
            ['AB_x1', 'AB_x2', 'AB_x3', 'A', 'B'] (only first and total-order)
            ['AB_x1', 'AB_x2', 'AB_x3', 'BA_x1', 'BA_x2', 'BA_x3', 'A', 'B'] (include second-order)

        Returns:
            labels_sample_matrices (list): labels of sample matrices
        """
        labels_sample_matrices = [
            "AB_" + self.parameter_names[i] for i in range(self.number_parameters)
        ]
        if self.calculate_second_order:
            sample_matrices_second_order = [
                "BA_" + self.parameter_names[i] for i in range(self.number_parameters)
            ]
            labels_sample_matrices = labels_sample_matrices + sample_matrices_second_order
        labels_sample_matrices.extend(["A", "B"])

        # check size
        if self.calculate_second_order:
            number_of_sample_matrices = 2 * self.number_parameters + 2
        else:
            number_of_sample_matrices = self.number_parameters + 2

        if number_of_sample_matrices != len(labels_sample_matrices):
            raise AssertionError("Wrong size of sampling matrices.")

        return labels_sample_matrices

    def _draw_base_samples(self):
        """Draw base sample matrices A and B.

        Returns:
            sample_matrix_a (ndarray): Saltelli A sample matrix
            sample_matrix_b (ndarray): Saltelli B sample matrix
        """
        draw_sample = self._get_base_sample_method()
        base_sample_matrix_a, base_sample_matrix_b = draw_sample()

        sample_matrix_a = self.parameters.inverse_cdf_transform(base_sample_matrix_a)
        sample_matrix_b = self.parameters.inverse_cdf_transform(base_sample_matrix_b)

        return sample_matrix_a, sample_matrix_b

    def _get_base_sample_method(self):
        """Get sampling approach for base samples.

        Returns:
             function object (obj): function object for base sample type
        """
        if self.sampling_approach == "pseudo_random":
            return self._base_sample_pseudo_random
        if self.sampling_approach == "quasi_random":
            return self._base_sample_quasi_random

        raise ValueError(
            "Unknown sampling approach. Valid approaches are pseudo_random or quasi_random."
        )

    def _base_sample_quasi_random(self):
        """Sample with Sobol sequence.

        Get pseudo-random base samples based on Sobol sequence.

        For details on the Sobol sequence
        Owen, Art B. ‘On Dropping the First Sobol’ Point’. ArXiv:2008.08051 [Cs, Math, Stat], 2021.
        https://arxiv.org/abs/2008.08051v4.

        Returns:
            base_sample_matrix_a (ndarray): Saltelli A sample matrix in unit range
            base_sample_matrix_b (ndarray): Saltelli B sample matrix in unit range
        """
        sobol_engine = qmc.Sobol(
            d=2 * self.number_parameters, scramble=True, seed=self.seed_monte_carlo
        )
        base_sequence = sobol_engine.random(n=self.number_monte_carlo_samples)

        base_sample_matrix_a = base_sequence[:, : self.number_parameters]
        base_sample_matrix_b = base_sequence[:, self.number_parameters :]

        return base_sample_matrix_a, base_sample_matrix_b

    def _base_sample_pseudo_random(self):
        """Get pseudo-random base samples.

        Based on numpy random-number generator.

        Returns:
            base_sample_matrix_a (ndarray): Saltelli A sample matrix in unit range
            base_sample_matrix_b (ndarray): Saltelli B sample matrix in unit range
        """
        rng = np.random.default_rng(seed=self.seed_monte_carlo)
        base_sample_matrix_a = rng.uniform(
            0, 1, (self.number_monte_carlo_samples, self.number_parameters)
        )
        base_sample_matrix_b = rng.uniform(
            0, 1, (self.number_monte_carlo_samples, self.number_parameters)
        )

        return base_sample_matrix_a, base_sample_matrix_b


class ThirdOrderSampler(Sampler):
    """ThirdOrderSampler class.

    Draw Monte-Carlo samples and generate sample matrices following

    Saltelli, A., et al. ‘Variance Based Sensitivity Analysis of Model Output. Design and
    Estimator for the Total Sensitivity Index’. Computer Physics Communications 181, no. 2
    (1 February 2010): 259–270.
    https://doi.org/10.1016/j.cpc.2009.09.018.

    Saltelli, A., ed. Global Sensitivity Analysis: The Primer. Chichester, England; Hoboken, NJ:
    John Wiley & Sons, Ltd., 2008.
    https://doi.org/10.1002/9780470725184.

    Attributes:
        calculate_third_order (bool): true if third-order indices only are calculated
        third_order_parameters (list): list of parameter combination for third-order index
    """

    def __init__(
        self,
        parameter_names,
        number_monte_carlo_samples,
        seed_monte_carlo,
        calculate_second_order,
        parameters,
        sampling_approach,
        calculate_third_order,
        third_order_parameters,
    ):
        """Initialize.

        Args:
            parameter_names (list): list of names of input parameters
            number_monte_carlo_samples (int): number of Monte-Carlo samples
            seed_monte_carlo (int): seed for random samples
            calculate_second_order (bool): true if second-order indices are calculated
            parameters (list): information about distribution of random variables
            sampling_approach (string): sampling approach (pseudo-random or quasi-random)
            calculate_third_order (bool): true if third-order indices only are calculated
            third_order_parameters (list): list of parameter combination for third-order index
        """
        super().__init__(
            parameter_names=parameter_names,
            number_monte_carlo_samples=number_monte_carlo_samples,
            seed_monte_carlo=seed_monte_carlo,
            calculate_second_order=calculate_second_order,
            parameters=parameters,
            sampling_approach=sampling_approach,
        )
        self.calculate_third_order = calculate_third_order
        self.third_order_parameters = third_order_parameters

    @classmethod
    def from_config_create(cls, method_options, parameter_names, parameters):
        """Create third-order sampler from problem description.

        Args:
            method_options (dict): dictionary with method options
            parameter_names (list): list of parameter names
            parameters (list): information about distribution of random variables

        Returns:
            sampler: Sampler object
        """
        number_monte_carlo_samples = method_options["number_monte_carlo_samples"]
        _logger.info("Number of Monte-Carlo samples = %i", number_monte_carlo_samples)

        seed_monte_carlo = method_options.get("seed_monte_carlo", 42)

        calculate_second_order = method_options.get("second_order", False)
        sampling_approach = method_options.get("sampling_approach", "quasi_random")

        calculate_third_order = method_options.get("third_order", False)
        third_order_parameters = method_options.get("third_order_parameters", None)

        return cls(
            parameter_names=parameter_names,
            number_monte_carlo_samples=number_monte_carlo_samples,
            seed_monte_carlo=seed_monte_carlo,
            calculate_second_order=calculate_second_order,
            parameters=parameters,
            sampling_approach=sampling_approach,
            calculate_third_order=calculate_third_order,
            third_order_parameters=third_order_parameters,
        )

    def sample(self):
        """Generate sample matrices.

        Generate sample matrices A, B, AB, BA, AB_ijk following

        Saltelli, A., et al. ‘Variance Based Sensitivity Analysis of Model Output. Design and
        Estimator for the Total Sensitivity Index’. Computer Physics Communications 181, no. 2
        (1 February 2010): 259–270. https://doi.org/10.1016/j.cpc.2009.09.018.

        Saltelli, A., ed. Global Sensitivity Analysis: The Primer. Chichester, England; Hoboken, NJ:
        John Wiley & Sons, Ltd., 2008. https://doi.org/10.1002/9780470725184.

        sample_matrix_a:  sample matrix A
        sample_matrix_b:  sample matrix B
        sample_matrix_ab:  separated from A by only change in X_i; from B by change in X_{~i}
        sample_matrix_ba:  separated from B by only change in X_i; from A by change in X_{~i}
        sample_matrix_ab_ijk: separated from A by a change in X_i, X_j and X_k

        Returns:
            samples (xr.Array): Monte-Carlo sample matrices
        """
        samples = self._init_samples()

        third_order_parameter_indices = [
            self.parameter_names.index(parameter) for parameter in self.third_order_parameters
        ]

        sample_matrix_a, sample_matrix_b = self._draw_base_samples()
        samples.loc[{"sample_matrix": "A"}] = sample_matrix_a
        samples.loc[{"sample_matrix": "B"}] = sample_matrix_b

        for i, parameter_name in enumerate(self.third_order_parameters):
            sample_matrix_ab = sample_matrix_a.copy()
            sample_matrix_ab[:, third_order_parameter_indices[i]] = sample_matrix_b[
                :, third_order_parameter_indices[i]
            ].flatten()
            samples.loc[{"sample_matrix": "AB_" + parameter_name}] = sample_matrix_ab

            sample_matrix_ba = sample_matrix_b.copy()
            sample_matrix_ba[:, third_order_parameter_indices[i]] = sample_matrix_a[
                :, third_order_parameter_indices[i]
            ].flatten()
            samples.loc[{"sample_matrix": "BA_" + parameter_name}] = sample_matrix_ba

        # all columns from A except columns (i,j,k) for the third-order index interaction from B
        sample_matrix_ab_ijk = sample_matrix_a.copy()
        sample_matrix_ab_ijk[:, third_order_parameter_indices] = sample_matrix_b[
            :, third_order_parameter_indices
        ]

        samples.loc[{"sample_matrix": ["AB_" + "_".join(self.third_order_parameters)]}] = (
            np.expand_dims(sample_matrix_ab_ijk, axis=1)
        )

        return samples

    def _setup_sample_matrices_labels(self):
        """Set up labels of sample matrices.

        For further information on the sample matrices see

        Saltelli, A., et al. ‘Variance Based Sensitivity Analysis of Model Output. Design and
        Estimator for the Total Sensitivity Index’. Computer Physics Communications 181, no. 2
        (1 February 2010): 259–270.
        https://doi.org/10.1016/j.cpc.2009.09.018.

        Saltelli, A., ed. Global Sensitivity Analysis: The Primer. Chichester, England; Hoboken, NJ:
        John Wiley & Sons, Ltd., 2008.
        https://doi.org/10.1002/9780470725184.

        Example:
            ['AB_x1', 'AB_x2', 'AB_x3', 'BA_x1', 'BA_x2', 'BA_x3', 'AB_x1x2x3', 'A', 'B']

        Returns:
            labels_sample_matrices (list): labels of sample matrices
        """
        labels_sample_matrices = [
            "AB_" + parameter_name for parameter_name in self.third_order_parameters
        ]
        labels_sample_matrices.extend(
            ["BA_" + parameter_name for parameter_name in self.third_order_parameters]
        )
        labels_sample_matrices.extend(["AB_" + "_".join(self.third_order_parameters)])
        labels_sample_matrices.extend(["A", "B"])
        return labels_sample_matrices
