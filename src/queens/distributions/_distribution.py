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
"""Distribution."""

import abc
import logging
from abc import abstractmethod

import numpy as np

from queens.utils.printing import get_str_table

_logger = logging.getLogger(__name__)


class Distribution(abc.ABC):
    """Base class for probability distributions."""

    @abstractmethod
    def draw(self, num_draws=1):
        """Draw samples.

        Args:
            num_draws (int, optional): Number of draws
        """

    @abstractmethod
    def logpdf(self, x):
        """Log of the probability *mass* function.

        In order to keep the interfaces unified the PMF is also accessed via the pdf.

        Args:
            x (np.ndarray): Positions at which the log pdf is evaluated
        """

    @abstractmethod
    def pdf(self, x):
        """Probability density function.

        Args:
            x (np.ndarray): Positions at which the pdf is evaluated
        """

    def export_dict(self):
        """Create a dict of the distribution.

        Returns:
            export_dict (dict): Dict containing distribution information
        """
        export_dict = vars(self)
        export_dict = {"type": self.__class__.__name__, **export_dict}
        return export_dict

    def __str__(self):
        """Get string for the given distribution.

        Returns:
            str: Table with distribution information
        """
        return get_str_table(type(self).__name__, self.export_dict())

    @staticmethod
    def check_positivity(**parameters):
        """Check if parameters are positive.

        Args:
            parameters (dict): Checked parameters
        """
        for name, value in parameters.items():
            if (np.array(value) <= 0).any():
                raise ValueError(
                    f"The parameter '{name}' has to be positive. " f"You specified {name}={value}."
                )


class Continuous(Distribution):
    """Base class for continuous probability distributions.

    Attributes:
        mean (np.ndarray): Mean of the distribution.
        covariance (np.ndarray): Covariance of the distribution.
        dimension (int): Dimensionality of the distribution.
    """

    def __init__(self, mean, covariance, dimension):
        """Initialize distribution.

        Args:
            mean (np.ndarray): mean of the distribution
            covariance (np.ndarray): covariance of the distribution
            dimension (int): dimensionality of the distribution
        """
        self.mean = mean
        self.covariance = covariance
        self.dimension = dimension

    @abstractmethod
    def cdf(self, x):
        """Cumulative distribution function.

        Args:
            x (np.ndarray): Positions at which the cdf is evaluated
        """

    @abstractmethod
    def draw(self, num_draws=1):
        """Draw samples.

        Args:
            num_draws (int, optional): Number of draws
        """

    @abstractmethod
    def logpdf(self, x):
        """Log of the probability density function.

        Args:
            x (np.ndarray): Positions at which the log pdf is evaluated
        """

    @abstractmethod
    def grad_logpdf(self, x):
        """Gradient of the log pdf with respect to *x*.

        Args:
            x (np.ndarray): Positions at which the gradient of log pdf is evaluated
        """

    def pdf(self, x):
        """Probability density function.

        Args:
            x (np.ndarray): Positions at which the pdf is evaluated

        Returns:
            pdf (np.ndarray): pdf at evaluated positions
        """
        pdf = np.exp(self.logpdf(x))
        return pdf

    @abstractmethod
    def ppf(self, quantiles):
        """Percent point function (inverse of cdf — quantiles).

        Args:
            quantiles (np.ndarray): Quantiles at which the ppf is evaluated
        """

    def check_1d(self):
        """Check if distribution is one-dimensional."""
        if self.dimension != 1:
            raise ValueError("Method does not support multivariate distributions!")

    @staticmethod
    def check_bounds(lower_bound, upper_bound):
        """Check sanity of bounds.

        Args:
            lower_bound (np.ndarray): Lower bound(s) of distribution
            upper_bound (np.ndarray): Upper bound(s) of distribution
        """
        if (upper_bound <= lower_bound).any():
            raise ValueError(
                f"Lower bound must be smaller than upper bound. "
                f"You specified lower_bound={lower_bound} and upper_bound={upper_bound}"
            )


class Discrete(Distribution):
    """Discrete distribution base class.

    Attributes:
        mean (np.ndarray): Mean of the distribution.
        covariance (np.ndarray): Covariance of the distribution.
        dimension (int): Dimensionality of the distribution.
        probabilities (np.ndarray): Probabilities associated to all the events in the sample space
        sample_space (np.ndarray): Samples, i.e. possible outcomes of sampling the distribution
    """

    def __init__(self, probabilities, sample_space, dimension=None):
        """Initialize the discrete distribution.

        Args:
            probabilities (np.ndarray): Probabilities associated to all the events in the sample
                                        space
            sample_space (np.ndarray): Samples, i.e. possible outcomes of sampling the distribution
            dimension (int): Dimension of a sample event
        """
        if len({len(d) for d in sample_space}) != 1:
            raise ValueError("Dimensions of the sample events do not match.")

        sample_space = np.array(sample_space).reshape(len(sample_space), -1)
        probabilities = np.array(probabilities)
        if dimension is None:
            self.dimension = sample_space[0].shape[0]
        else:
            if not isinstance(dimension, int) or dimension <= 0:
                raise ValueError(f"Dimension has to be a positive integer not {dimension}.")
            self.dimension = dimension

        if len(sample_space) != len(probabilities):
            raise ValueError(
                f"The number of probabilities {len(probabilities)} does not match the number of"
                f" events in the sample space {len(sample_space)}"
            )

        super().check_positivity(probabilities=probabilities)

        if not np.isclose(np.sum(probabilities), 1, atol=0):
            _logger.info("Probabilities do not sum up to one, they are going to be normalized.")
            probabilities = probabilities / np.sum(probabilities)

        # Sort the sample events
        if self.dimension == 1:
            indices = np.argsort(sample_space.flatten())
            self.probabilities = probabilities[indices]
            self.sample_space = sample_space[indices]
        else:
            self.probabilities = probabilities
            self.sample_space = sample_space

        self.mean, self.covariance = self._compute_mean_and_covariance()

    @abstractmethod
    def draw(self, num_draws=1):
        """Draw samples.

        Args:
            num_draws (int, optional): Number of draws
        """

    @abstractmethod
    def logpdf(self, x):
        """Log of the probability *mass* function.

        In order to keep the interfaces unified the PMF is also accessed via the pdf.

        Args:
            x (np.ndarray): Positions at which the log pdf is evaluated
        """

    @abstractmethod
    def pdf(self, x):
        """Probability density function.

        Args:
            x (np.ndarray): Positions at which the pdf is evaluated
        """

    @abstractmethod
    def cdf(self, x):
        """Cumulative distribution function.

        Args:
            x (np.ndarray): Positions at which the cdf is evaluated
        """

    @abstractmethod
    def ppf(self, quantiles):
        """Percent point function (inverse of cdf - quantiles).

        Args:
            quantiles (np.ndarray): Quantiles at which the ppf is evaluated
        """

    @abstractmethod
    def _compute_mean_and_covariance(self):
        """Compute the mean value and covariance of the distribution.

        Returns:
            mean (np.ndarray): Mean value of the distribution
            covariance (np.ndarray): Covariance of the distribution
        """

    def check_1d(self):
        """Check if distribution is one-dimensional."""
        if self.dimension != 1:
            raise ValueError("Method does not support multivariate distributions!")

    @staticmethod
    def check_duplicates_in_sample_space(sample_space):
        """Check for duplicate events in the sample space.

        Args:
            sample_space (np.ndarray): Samples, i.e. possible outcomes of sampling the distribution
        """
        if len(sample_space) != len(np.unique(sample_space, axis=0)):
            raise ValueError("The sample space contains duplicate events, this is not possible.")
