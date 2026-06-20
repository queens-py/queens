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
"""Truncated normal distribution."""

import numpy as np
import scipy.stats
from numpy.typing import ArrayLike

from queens.distributions._distribution import Continuous
from queens.utils.logger_settings import log_init_args


class TruncatedNormal(Continuous):
    """Truncated normal distribution.

    A one-dimensional normal distribution restricted to the finite interval
    [lower_bound, upper_bound]. The distribution is parametrized by the
    mean and std of the underlying (unbounded) normal distribution.

    Internally, scipy.stats.truncnorm is used with standardized bounds
    a = (lower_bound - unbounded_mean) / unbounded_std and
    b = (upper_bound - unbounded_mean) / unbounded_std.

    Attributes:
        unbounded_mean: Mean of the underlying (unbounded) normal distribution.
        unbounded_std: Standard deviation of the underlying (unbounded) normal distribution.
        lower_bound: Lower bound of the distribution.
        upper_bound: Upper bound of the distribution.
        scipy_truncnorm: Scipy truncated normal distribution object.
        mean: (inherited) Mean of the truncated distribution, computed via scipy.
        covariance: (inherited) Variance of the truncated distribution, computed via scipy.
        dimension: (inherited) Dimensionality of the distribution (always 1).
    """

    @log_init_args
    def __init__(
        self,
        unbounded_mean: ArrayLike,
        unbounded_std: ArrayLike,
        lower_bound: ArrayLike,
        upper_bound: ArrayLike,
    ) -> None:
        """Initialize truncated normal distribution.

        Args:
            unbounded_mean: Mean of the underlying (unbounded) normal distribution.
            unbounded_std: Standard deviation of the underlying (unbounded) normal distribution.
                Must be positive.
            lower_bound: Lower bound of the distribution. Must be smaller than upper_bound.
            upper_bound: Upper bound of the distribution.
        """
        unbounded_mean = np.array(unbounded_mean).reshape(-1)
        unbounded_std = np.array(unbounded_std).reshape(-1)
        lower_bound = np.array(lower_bound).reshape(-1)
        upper_bound = np.array(upper_bound).reshape(-1)

        if max(unbounded_mean.size, unbounded_std.size, lower_bound.size, upper_bound.size) != 1:
            raise NotImplementedError(
                "Only one-dimensional truncated normal distributions are supported."
            )

        super().check_positivity(unbounded_std=unbounded_std)
        super().check_bounds(lower_bound, upper_bound)

        a = (lower_bound - unbounded_mean) / unbounded_std
        b = (upper_bound - unbounded_mean) / unbounded_std
        scipy_truncnorm = scipy.stats.truncnorm(a, b, loc=unbounded_mean, scale=unbounded_std)

        self.unbounded_mean = unbounded_mean
        self.unbounded_std = unbounded_std
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.scipy_truncnorm = scipy_truncnorm

        super().__init__(
            mean=scipy_truncnorm.mean(),
            covariance=scipy_truncnorm.var(),
            dimension=1,
        )

    def cdf(self, x: np.ndarray) -> np.ndarray:
        """Cumulative distribution function.

        Args:
            x: Positions at which the CDF is evaluated

        Returns:
            CDF at positions
        """
        cdf = self.scipy_truncnorm.cdf(x).reshape(-1)
        return cdf

    def draw(self, num_draws: int = 1) -> np.ndarray:
        """Draw samples.

        Args:
            num_draws: Number of draws

        Returns:
            Drawn samples from the distribution
        """
        samples = self.scipy_truncnorm.rvs(size=num_draws).reshape(-1, 1)
        return samples

    def logpdf(self, x: np.ndarray) -> np.ndarray:
        """Log of the probability density function.

        Args:
            x: Positions at which the log-PDF is evaluated

        Returns:
            Log-PDF at positions
        """
        logpdf = self.scipy_truncnorm.logpdf(x).reshape(-1)
        return logpdf

    def grad_logpdf(self, x: np.ndarray) -> np.ndarray:
        """Gradient of the log-PDF with respect to x.

        Args:
            x: Positions at which the gradient of the log-PDF is evaluated

        Returns:
            Gradient of the log-PDF at positions
        """
        x = np.asarray(x).reshape(-1)
        grad_logpdf = (self.unbounded_mean - x) / self.unbounded_std**2
        return grad_logpdf

    def pdf(self, x: np.ndarray) -> np.ndarray:
        """Probability density function.

        Args:
            x: Positions at which the PDF is evaluated

        Returns:
            PDF at positions
        """
        pdf = self.scipy_truncnorm.pdf(x).reshape(-1)
        return pdf

    def ppf(self, quantiles: np.ndarray) -> np.ndarray:
        """Percent point function (inverse of CDF — quantiles).

        Args:
            quantiles: Quantiles at which the PPF is evaluated

        Returns:
            Positions which correspond to given quantiles
        """
        ppf = self.scipy_truncnorm.ppf(quantiles).reshape(-1)
        return ppf
