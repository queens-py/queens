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
from numpy.typing import ArrayLike
from scipy.stats import multinomial

from queens.distributions._distribution import Discrete
from queens.utils.logger_settings import log_init_args


class Multinomial(Discrete):
    """Multinomial distribution."""

    @log_init_args
    def __init__(self, n_trials: int, probabilities: ArrayLike) -> None:
        """Initialize discrete uniform distribution.

        Args:
            n_trials: Number of trials, i.e., the value to which every multivariate sample adds up
                to.
            probabilities: Probabilities associated with all the events in the sample space.
        """
        if not isinstance(n_trials, int) or n_trials <= 0:
            raise ValueError(f"n_trials was set to {n_trials} needs to be a positive integer.")

        self.n_trials = n_trials
        probabilities_array = np.array(probabilities)

        # we misuse the sample_space attribute of the base class to store the number of trials
        sample_space = np.ones((len(probabilities_array), 1)) * self.n_trials
        super().__init__(probabilities_array, sample_space, dimension=len(probabilities_array))
        self.scipy_multinomial = multinomial(self.n_trials, self.probabilities)

    def _compute_mean_and_covariance(self) -> tuple[np.ndarray, np.ndarray]:
        """Compute the mean value and covariance of the mixture model.

        Returns:
            Mean value of the distribution
            Covariance of the distribution
        """
        n_trials = self.sample_space[0]
        mean = n_trials * self.probabilities
        covariance = n_trials * (
            np.diag(self.probabilities) - np.outer(self.probabilities, self.probabilities)
        )
        return mean, covariance

    def draw(self, num_draws: int = 1) -> np.ndarray:
        """Draw samples.

        Args:
            num_draws: Number of draws

        Returns:
            Drawn samples
        """
        return np.random.multinomial(self.n_trials, self.probabilities, size=num_draws)

    def logpdf(self, x: np.ndarray) -> np.ndarray:
        """Log of the probability mass function.

        Args:
            x: Positions at which the log-PMF is evaluated

        Returns:
            Log-PMF at positions
        """
        return self.scipy_multinomial.logpmf(x)

    def pdf(self, x: np.ndarray) -> np.ndarray:
        """Probability mass function.

        Args:
            x: Positions at which the PMF is evaluated

        Returns:
            PMF at positions
        """
        return self.scipy_multinomial.pmf(x)

    def cdf(self, x: np.ndarray) -> None:
        """Cumulative distribution function.

        Args:
            x: Positions at which the CDF is evaluated
        """
        super().check_1d()

    def ppf(self, quantiles: np.ndarray) -> None:
        """Percent point function (inverse of CDF - quantiles).

        Args:
            quantiles: Quantiles at which the PPF is evaluated
        """
        super().check_1d()
