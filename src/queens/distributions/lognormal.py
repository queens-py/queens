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
"""LogNormal Distribution."""

import numpy as np
import scipy.linalg
import scipy.stats

from queens.distributions._distribution import Continuous
from queens.distributions.normal import Normal
from queens.utils.logger_settings import log_init_args


class LogNormal(Continuous):
    """LogNormal distribution.

    Support in (0, +inf).

    Attributes:
        normal_distribution (Normal): Underlying normal distribution.
    """

    @log_init_args
    def __init__(self, normal_mean, normal_covariance):
        """Initialize lognormal distribution.

        Args:
            normal_mean (array_like): mean of the normal distribution
            normal_covariance (array_like): covariance of the normal distribution
        """
        self.normal_distribution = Normal(normal_mean, normal_covariance)

        normal_covariance_diag = np.diag(self.normal_distribution.covariance)

        mean = np.exp(self.normal_distribution.mean + 0.5 * normal_covariance_diag)
        covariance = np.exp(
            self.normal_distribution.mean.reshape(-1, 1)
            + self.normal_distribution.mean.reshape(1, -1)
            + 0.5 * (normal_covariance_diag.reshape(-1, 1) + normal_covariance_diag.reshape(1, -1))
        ) * (np.exp(self.normal_distribution.covariance) - 1)

        super().__init__(
            mean=mean, covariance=covariance, dimension=self.normal_distribution.dimension
        )

    def cdf(self, x):
        """Cumulative distribution function.

        Args:
            x (np.ndarray): Positions at which the cdf is evaluated

        Returns:
            cdf (np.ndarray): cdf at evaluated positions
        """
        return self.normal_distribution.cdf(np.log(x))

    def draw(self, num_draws=1):
        """Draw samples.

        Args:
            num_draws (int, optional): Number of draws

        Returns:
            samples (np.ndarray): Drawn samples from the distribution
        """
        return np.exp(self.normal_distribution.draw(num_draws=num_draws))

    def logpdf(self, x):
        """Log of the probability density function.

        Args:
            x (np.ndarray): Positions at which the log pdf is evaluated

        Returns:
            logpdf (np.ndarray): pdf at evaluated positions
        """
        log_x = np.log(x).reshape(-1, self.dimension)
        dist = log_x - self.normal_distribution.mean
        logpdf = (
            self.normal_distribution.logpdf_const
            - np.sum(log_x, axis=1)
            - 0.5 * (np.dot(dist, self.normal_distribution.precision) * dist).sum(axis=1)
        )
        return logpdf

    def grad_logpdf(self, x):
        """Gradient of the log pdf with respect to *x*.

        Args:
            x (np.ndarray): Positions at which the gradient of log pdf is evaluated

        Returns:
            grad_logpdf (np.ndarray): Gradient of the log pdf evaluated at positions
        """
        x = x.reshape(-1, self.dimension)
        x[x == 0] = np.nan
        grad_logpdf = (
            1
            / x
            * (
                np.dot(
                    self.normal_distribution.mean.reshape(1, -1) - np.log(x),
                    self.normal_distribution.precision,
                )
                - 1
            )
        )
        return grad_logpdf

    def pdf(self, x):
        """Probability density function.

        Args:
            x (np.ndarray): Positions at which the pdf is evaluated

        Returns:
            pdf (np.ndarray): pdf at evaluated positions
        """
        return np.exp(self.logpdf(x))

    def ppf(self, quantiles):
        """Percent point function (inverse of cdf — quantiles).

        Args:
            quantiles (np.ndarray): Quantiles at which the ppf is evaluated

        Returns:
            ppf (np.ndarray): Positions which correspond to given quantiles
        """
        self.check_1d()
        ppf = scipy.stats.lognorm.ppf(
            quantiles,
            s=self.normal_distribution.covariance ** (1 / 2),
            scale=np.exp(self.normal_distribution.mean),
        ).reshape(-1)
        return ppf
