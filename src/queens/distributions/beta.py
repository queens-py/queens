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
"""Beta Distribution."""

import numpy as np
import scipy.stats

from queens.distributions._distribution import Continuous
from queens.utils.logger_settings import log_init_args


class Beta(Continuous):
    """Beta distribution.

    A generalized one-dimensional beta distribution based on scipy stats. The generalized beta
    distribution has a lower bound and an upper bound.
    The parameters *a* and *b* determine the shape of the distribution within these bounds.

    Attributes:
        lower_bound: Lower bound of the beta distribution.
        upper_bound: Upper bound of the beta distribution.
        a: Shape parameter of the beta distribution, must be greater than 0.
        b: Shape parameter of the beta distribution, must be greater than 0.
        scipy_beta: Scipy beta distribution object.
    """

    @log_init_args
    def __init__(
        self, lower_bound: float | np.ndarray, upper_bound: float | np.ndarray, a: float, b: float
    ) -> None:
        """Initialize Beta distribution.

        Args:
            lower_bound: Lower bound of the beta distribution.
            upper_bound: Upper bound of the beta distribution.
            a: Shape parameter of the beta distribution, must be > 0.
            b: Shape parameter of the beta distribution, must be > 0.
        """
        lower_bound = np.array(lower_bound).reshape(-1)
        upper_bound = np.array(upper_bound).reshape(-1)
        self.a = np.array(a)
        self.b = np.array(b)
        super().check_positivity(a=self.a, b=self.b)
        super().check_bounds(lower_bound, upper_bound)
        scale = upper_bound - lower_bound
        scipy_beta = scipy.stats.beta(scale=scale, loc=lower_bound, a=self.a, b=self.b)
        mean = scipy_beta.mean()
        covariance = scipy_beta.var()

        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.scipy_beta = scipy_beta

        super().__init__(mean=mean, covariance=covariance, dimension=1)

    def cdf(self, x: np.ndarray) -> np.ndarray:
        """Cumulative distribution function.

        Args:
            x: Positions at which the CDF is evaluated

        Returns:
            CDF at positions
        """
        cdf = self.scipy_beta.cdf(x).reshape(-1)
        return cdf

    def draw(self, num_draws: int = 1) -> np.ndarray:
        """Draw samples.

        Args:
            num_draws: Number of draws

        Returns:
            Drawn samples from the distribution
        """
        samples = self.scipy_beta.rvs(size=num_draws).reshape(-1, 1)
        return samples

    def logpdf(self, x: np.ndarray) -> np.ndarray:
        """Log of the probability density function.

        Args:
            x: Positions at which the log-PDF is evaluated

        Returns:
            Log-PDF at positions
        """
        logpdf = self.scipy_beta.logpdf(x).reshape(-1)
        return logpdf

    def grad_logpdf(self, x: np.ndarray) -> np.ndarray:
        """Gradient of the log-PDF with respect to *x*.

        Args:
            x: Positions at which the gradient of the log-PDF is evaluated
        """
        raise NotImplementedError(
            "This method is currently not implemented for the beta distribution."
        )

    def pdf(self, x: np.ndarray) -> np.ndarray:
        """Probability density function.

        Args:
            x: Positions at which the PDF is evaluated

        Returns:
            PDF at positions
        """
        pdf = self.scipy_beta.pdf(x).reshape(-1)
        return pdf

    def ppf(self, quantiles: np.ndarray) -> np.ndarray:
        """Percent point function (inverse of CDF — quantiles).

        Args:
            quantiles: Quantiles at which the PPF is evaluated

        Returns:
            Positions which correspond to given quantiles
        """
        ppf = self.scipy_beta.ppf(quantiles).reshape(-1)
        return ppf
