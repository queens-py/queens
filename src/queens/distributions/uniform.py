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
"""Uniform distribution."""

import numpy as np
import scipy.linalg
import scipy.stats

from queens.distributions._distribution import Continuous
from queens.utils.logger_settings import log_init_args


class Uniform(Continuous):
    """Uniform distribution class.

    Attributes:
        lower_bound (np.ndarray): Lower bound(s) of the distribution.
        upper_bound (np.ndarray): Upper bound(s) of the distribution.
        width (np.ndarray): Width(s) of the distribution.
        pdf_const (float): Constant for the evaluation of the pdf.
        logpdf_const (float): Constant for the evaluation of the log pdf.
    """

    @log_init_args
    def __init__(self, lower_bound, upper_bound):
        """Initialize uniform distribution.

        Args:
            lower_bound (array_like): Lower bound(s) of the distribution
            upper_bound (array_like): Upper bound(s) of the distribution
        """
        lower_bound = np.array(lower_bound).reshape(-1)
        upper_bound = np.array(upper_bound).reshape(-1)
        super().check_bounds(lower_bound, upper_bound)
        width = upper_bound - lower_bound

        mean = (lower_bound + upper_bound) / 2.0
        covariance = np.diag(width**2 / 12.0)
        dimension = mean.size

        pdf_const = 1.0 / np.prod(width)
        logpdf_const = np.log(pdf_const)
        super().__init__(mean=mean, covariance=covariance, dimension=dimension)
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.width = width
        self.pdf_const = pdf_const
        self.logpdf_const = logpdf_const

    def cdf(self, x):
        """Cumulative distribution function.

        Args:
            x (np.ndarray): Positions at which the cdf is evaluated

        Returns:
            cdf (np.ndarray): Cdf at evaluated positions
        """
        cdf = np.prod(
            np.clip(
                (x.reshape(-1, self.dimension) - self.lower_bound) / self.width,
                a_min=np.zeros(self.dimension),
                a_max=np.ones(self.dimension),
            ),
            axis=1,
        )
        return cdf

    def draw(self, num_draws=1):
        """Draw samples.

        Args:
            num_draws (int, optional): Number of draws

        Returns:
            samples (np.ndarray): Drawn samples from the distribution
        """
        samples = np.random.uniform(
            low=self.lower_bound, high=self.upper_bound, size=(num_draws, self.dimension)
        )
        return samples

    def logpdf(self, x):
        """Log of the probability density function.

        Args:
            x (np.ndarray): Positions at which the log pdf is evaluated

        Returns:
            logpdf (np.ndarray): log pdf at evaluated positions
        """
        x = x.reshape(-1, self.dimension)
        within_bounds = (x >= self.lower_bound).all(axis=1) * (x <= self.upper_bound).all(axis=1)
        logpdf = np.where(within_bounds, self.logpdf_const, -np.inf)
        return logpdf

    def grad_logpdf(self, x):
        """Gradient of the log pdf with respect to *x*.

        Args:
            x (np.ndarray): Positions at which the gradient of log pdf is evaluated

        Returns:
            grad_logpdf (np.ndarray): Gradient of the log pdf evaluated at positions
        """
        x = x.reshape(-1, self.dimension)
        grad_logpdf = np.zeros(x.shape)
        return grad_logpdf

    def pdf(self, x):
        """Probability density function.

        Args:
            x (np.ndarray): Positions at which the pdf is evaluated

        Returns:
            pdf (np.ndarray): pdf at evaluated positions
        """
        x = x.reshape(-1, self.dimension)
        # Check if positions are within bounds of the uniform distribution
        within_bounds = (x >= self.lower_bound).all(axis=1) * (x <= self.upper_bound).all(axis=1)
        pdf = within_bounds * self.pdf_const
        return pdf

    def ppf(self, quantiles):
        """Percent point function (inverse of cdf — quantiles).

        Args:
            quantiles (np.ndarray): Quantiles at which the ppf is evaluated

        Returns:
            ppf (np.ndarray): Positions which correspond to given quantiles
        """
        self.check_1d()
        ppf = scipy.stats.uniform.ppf(q=quantiles, loc=self.lower_bound, scale=self.width).reshape(
            -1
        )
        return ppf
