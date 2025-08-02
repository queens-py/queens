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
"""Multinomial distribution."""

import numpy as np
from scipy.stats import multinomial

from queens.distributions._distribution import Discrete
from queens.utils.logger_settings import log_init_args


class Multinomial(Discrete):
    """Multinomial distribution."""

    @log_init_args
    def __init__(self, n_trials, probabilities):
        """Initialize discrete uniform distribution.

        Args:
            probabilities (np.ndarray): Probabilities associated to all the events in the sample
                                        space
            n_trials (int): Number of trials, i.e. the value to which every multivariate sample
                            adds up to.
        """
        if not isinstance(n_trials, int) or n_trials <= 0:
            raise ValueError(f"n_trials was set to {n_trials} needs to be a positive integer.")

        self.n_trials = n_trials

        # we misuse the sample_space attribute of the base class to store the number of trials
        sample_space = np.ones((len(probabilities), 1)) * self.n_trials
        super().__init__(probabilities, sample_space, dimension=len(probabilities))
        self.scipy_multinomial = multinomial(self.n_trials, self.probabilities)

    def _compute_mean_and_covariance(self):
        """Compute the mean value and covariance of the mixture model.

        Returns:
            mean (np.ndarray): Mean value of the distribution
            covariance (np.ndarray): Covariance of the distribution
        """
        n_trials = self.sample_space[0]
        mean = n_trials * self.probabilities
        covariance = n_trials * (
            np.diag(self.probabilities) - np.outer(self.probabilities, self.probabilities)
        )
        return mean, covariance

    def draw(self, num_draws=1):
        """Draw samples.

        Args:
            num_draws (int, optional): Number of draws
        """
        return np.random.multinomial(self.n_trials, self.probabilities, size=num_draws)

    def logpdf(self, x):
        """Log of the probability mass function.

        Args:
            x (np.ndarray): Positions at which the log pdf is evaluated
        """
        return self.scipy_multinomial.logpmf(x)

    def pdf(self, x):
        """Probability mass function.

        Args:
            x (np.ndarray): Positions at which the pdf is evaluated
        """
        return self.scipy_multinomial.pmf(x)

    def cdf(self, x):
        """Cumulative distribution function.

        Args:
            x (np.ndarray): Positions at which the cdf is evaluated
        """
        super().check_1d()

    def ppf(self, quantiles):
        """Percent point function (inverse of cdf - quantiles).

        Args:
            quantiles (np.ndarray): Quantiles at which the ppf is evaluated
        """
        super().check_1d()
