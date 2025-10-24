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
"""Normal distribution."""

import numpy as np
import scipy.stats
from numpy.typing import ArrayLike

from queens.distributions._distribution import Continuous
from queens.utils.logger_settings import log_init_args
from queens.utils.numpy_array import at_least_2d
from queens.utils.numpy_linalg import safe_cholesky


class Normal(Continuous):
    """Normal distribution.

    Attributes:
        low_chol: Lower-triangular Cholesky factor of covariance matrix.
        precision: Precision matrix corresponding to covariance matrix.
        logpdf_const: Constant for evaluation of log-PDF.
    """

    @log_init_args
    def __init__(
        self,
        mean: ArrayLike,
        covariance: ArrayLike,
    ) -> None:
        """Initialize normal distribution.

        Args:
            mean: Mean of the distribution
            covariance: Covariance of the distribution
        """
        mean = np.array(mean).reshape(-1)
        covariance = at_least_2d(np.array(covariance))

        # sanity checks
        dimension = covariance.shape[0]
        if covariance.ndim != 2:
            raise ValueError(
                f"Provided covariance is not a matrix. "
                f"Provided covariance shape: {covariance.shape}"
            )
        if dimension != covariance.shape[1]:
            raise ValueError(
                "Provided covariance matrix is not quadratic. "
                f"Provided covariance shape: {covariance.shape}"
            )
        if not np.allclose(covariance.T, covariance):
            raise ValueError(
                "Provided covariance matrix is not symmetric. " f"Provided covariance: {covariance}"
            )
        if mean.shape[0] != dimension:
            raise ValueError(
                f"Dimension of mean vector and covariance matrix do not match. "
                f"Provided dimension of mean vector: {mean.shape[0]}. "
                f"Provided dimension of covariance matrix: {dimension}. "
            )

        low_chol, precision, logpdf_const = self._calculate_distribution_parameters(covariance)
        super().__init__(mean, covariance, dimension)
        self.low_chol = low_chol
        self.precision = precision
        self.logpdf_const = logpdf_const

    def cdf(self, x: np.ndarray) -> np.ndarray:
        """Cumulative distribution function.

        Args:
            x: Positions at which the CDF is evaluated

        Returns:
            CDF at positions
        """
        cdf = scipy.stats.multivariate_normal.cdf(
            x.reshape(-1, self.dimension), mean=self.mean, cov=self.covariance
        ).reshape(-1)
        return cdf

    def draw(self, num_draws: int = 1) -> np.ndarray:
        """Draw samples.

        Args:
            num_draws: Number of draws

        Returns:
            Drawn samples from the distribution
        """
        uncorrelated_vector = np.random.randn(self.dimension, num_draws)
        samples = self.mean + np.dot(self.low_chol, uncorrelated_vector).T
        return samples

    def logpdf(self, x: np.ndarray) -> np.ndarray:
        """Log of the probability density function.

        Args:
            x: Positions at which the log-PDF is evaluated

        Returns:
            Log-PDF at positions
        """
        dist = x.reshape(-1, self.dimension) - self.mean
        logpdf = self.logpdf_const - 0.5 * (np.dot(dist, self.precision) * dist).sum(axis=1)
        return logpdf

    def grad_logpdf(self, x: np.ndarray) -> np.ndarray:
        """Gradient of the log-PDF with respect to *x*.

        Args:
            x: Positions at which the gradient of log-PDF is evaluated

        Returns:
            Gradient of the log-PDF evaluated at positions
        """
        x = x.reshape(-1, self.dimension)
        grad_logpdf = np.dot(self.mean.reshape(1, -1) - x, self.precision)
        return grad_logpdf

    def ppf(self, quantiles: ArrayLike) -> np.ndarray:
        """Percent point function (inverse of CDF — quantiles).

        Args:
            quantiles: Quantiles at which the PPF is evaluated

        Returns:
            Positions which correspond to given quantiles
        """
        self.check_1d()
        ppf = scipy.stats.norm.ppf(
            quantiles, loc=self.mean, scale=self.covariance ** (1 / 2)
        ).reshape(-1)
        return ppf

    def update_covariance(self, covariance: np.ndarray) -> None:
        """Update covariance and dependent distribution parameters.

        Args:
            covariance: Covariance matrix
        """
        low_chol, precision, logpdf_const = self._calculate_distribution_parameters(covariance)
        self.covariance = covariance
        self.low_chol = low_chol
        self.precision = precision
        self.logpdf_const = logpdf_const

    @staticmethod
    def _calculate_distribution_parameters(
        covariance: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.float64]:
        """Calculate covariance dependent distribution parameters.

        Args:
            covariance: Covariance matrix

        Returns:
            low_chol: Lower-triangular Cholesky factor of covariance matrix
            precision: Precision matrix corresponding to covariance matrix
            logpdf_const: Constant for evaluation of log-PDF
        """
        dimension = covariance.shape[0]
        low_chol = safe_cholesky(covariance)

        # precision matrix Q and determinant of cov matrix
        chol_inv = np.linalg.inv(low_chol)
        precision = np.dot(chol_inv.T, chol_inv)

        # constant needed for pdf
        logpdf_const = -1 / 2 * (np.log(2.0 * np.pi) * dimension + np.linalg.slogdet(covariance)[1])
        return low_chol, precision, logpdf_const
