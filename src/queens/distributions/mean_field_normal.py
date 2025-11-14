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
"""Mean-field normal distribution."""

import numpy as np
import scipy.stats
from numpy.typing import ArrayLike
from scipy.special import erf  # pylint:disable=no-name-in-module

from queens.distributions._distribution import Continuous
from queens.utils.logger_settings import log_init_args


class MeanFieldNormal(Continuous):
    """Mean-field normal distribution.

    Attributes:
        standard_deviation: Standard deviation vector
    """

    @log_init_args
    def __init__(self, mean: ArrayLike, variance: ArrayLike, dimension: int) -> None:
        """Initialize mean-field normal distribution.

        Args:
            mean: Mean of the distribution
            variance: Variance of the distribution
            dimension: Dimensionality of the distribution
        """
        mean_array = np.array(mean)
        variance_array = np.array(variance)
        mean_array = MeanFieldNormal.get_check_array_dimension_and_reshape(mean_array, dimension)
        covariance = MeanFieldNormal.get_check_array_dimension_and_reshape(
            variance_array, dimension
        )
        self.standard_deviation = np.sqrt(covariance)
        super().__init__(mean_array, covariance, dimension)

    def update_variance(self, variance: np.ndarray) -> None:
        """Update the variance of the mean-field normal distribution.

        Args:
            variance: New variance vector
        """
        covariance = MeanFieldNormal.get_check_array_dimension_and_reshape(variance, self.dimension)
        self.covariance = covariance
        self.standard_deviation = np.sqrt(covariance)

    def update_mean(self, mean: np.ndarray) -> None:
        """Update the mean of the mean-field normal distribution.

        Args:
            mean: New mean vector
        """
        mean = MeanFieldNormal.get_check_array_dimension_and_reshape(mean, self.dimension)
        self.mean = mean

    def cdf(self, x: np.ndarray) -> np.ndarray:
        """Cumulative distribution function.

        Args:
            x: Positions at which the CDF is evaluated

        Returns:
            CDF at positions
        """
        z = (x - self.mean) / self.standard_deviation
        cdf = 0.5 * (1 + erf(z / np.sqrt(2)))
        cdf = np.prod(cdf, axis=1).reshape(x.shape[0], -1)
        return cdf

    def draw(self, num_draws: int = 1) -> np.ndarray:
        """Draw samples.

        Args:
            num_draws: Number of draws

        Returns:
            Drawn samples from the distribution
        """
        samples = np.random.randn(num_draws, self.dimension) * self.standard_deviation.reshape(
            1, -1
        ) + self.mean.reshape(1, -1)

        return samples

    def logpdf(self, x: np.ndarray) -> np.ndarray:
        """Log of the probability density function.

        Args:
            x: Positions at which the log-PDF is evaluated

        Returns:
            Log-PDF at positions
        """
        dist = x - self.mean
        logpdf = (
            -0.5 * self.dimension * np.log(2 * np.pi)
            - 0.5 * np.sum(np.log(self.covariance))
            - 0.5 * np.sum(dist**2 / self.covariance, axis=1)
        ).flatten()

        return logpdf

    def grad_logpdf(self, x: np.ndarray) -> np.ndarray:
        """Gradient of the log-PDF with respect to x.

        Args:
            x: Positions at which the gradient of log-PDF is evaluated

        Returns:
            Gradient of the log-PDF evaluated at positions
        """
        gradients_batch = -(x - self.mean) / self.covariance
        gradients_batch = gradients_batch.reshape(x.shape[0], -1)

        return gradients_batch

    def grad_logpdf_var(self, x: np.ndarray) -> np.ndarray:
        """Gradient of the log-PDF with respect to the variance vector.

        Args:
            x: Positions at which the gradient of the log-PDF is evaluated

        Returns:
            Gradient of the log-PDF w.r.t. the variance at given variance vector and position x
        """
        sample_batch = x.reshape(-1, self.dimension)

        part_1 = -0.5 * (1 / self.covariance)
        part_2 = 0.5 * ((sample_batch - self.mean) ** 2 / self.covariance**2)
        gradient_batch = part_1 + part_2

        grad_logpdf_var = gradient_batch.reshape(x.shape[0], -1)

        return grad_logpdf_var

    def ppf(self, quantiles: np.ndarray) -> np.ndarray:
        """Percent point function (inverse of cdf — quantiles).

        Args:
            quantiles: Quantiles at which the PPF is evaluated
        """
        self.check_1d()  # pylint: disable=duplicate-code
        ppf = scipy.stats.norm.ppf(
            quantiles, loc=self.mean, scale=self.covariance ** (1 / 2)
        ).reshape(-1)
        return ppf

    @staticmethod
    def get_check_array_dimension_and_reshape(
        input_array: np.ndarray, dimension: int
    ) -> np.ndarray:
        """Check dimensions and potentially reshape array.

        Args:
            input_array: Input array
            dimension: Dimension of the array

        Returns:
            Input array with correct dimension
        """
        if not isinstance(input_array, np.ndarray):
            raise TypeError("Input must be a numpy array.")

        # allow one dimensional inputs that update the entire array
        if input_array.size == 1:
            input_array = np.tile(input_array, dimension)

        # raise error in case of dimension mismatch
        if input_array.size != dimension:
            raise ValueError(
                "Dimension of input vector and dimension attribute do not match."
                f"Provided dimension of input vector: {input_array.size}."
                f"Provided dimension was: {dimension}."
            )
        return input_array
