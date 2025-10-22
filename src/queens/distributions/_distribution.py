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
from collections.abc import Sequence, Sized
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from queens.utils.printing import get_str_table

_logger = logging.getLogger(__name__)


class Distribution(abc.ABC):
    """Base class for probability distributions."""

    @abstractmethod
    def draw(self, num_draws: int = 1) -> Any:
        """Draw samples.

        Args:
            num_draws: Number of draws
        """

    @abstractmethod
    def logpdf(self, x: np.ndarray) -> Any:
        """Log of the probability *mass* function.

        In order to keep the interfaces unified the PMF is also accessed via the PDF.

        Args:
            x: Positions at which the log-PDF is evaluated
        """

    @abstractmethod
    def pdf(self, x: np.ndarray) -> Any:
        """Probability density function.

        Args:
            x: Positions at which the PDF is evaluated
        """

    def export_dict(self) -> dict:
        """Create a dict of the distribution.

        Returns:
            Dict containing distribution information
        """
        export_dict = vars(self)
        export_dict = {"type": self.__class__.__name__, **export_dict}
        return export_dict

    def __str__(self) -> str:
        """Get string for the given distribution.

        Returns:
            Table with distribution information
        """
        return get_str_table(type(self).__name__, self.export_dict())

    @staticmethod
    def check_positivity(**parameters: ArrayLike) -> None:
        """Check if parameters are positive.

        Args:
            parameters: Checked parameters
        """
        for name, value in parameters.items():
            if (np.array(value) <= 0).any():
                raise ValueError(
                    f"The parameter '{name}' has to be positive. You specified {name}={value}."
                )


class Continuous(Distribution):
    """Base class for continuous probability distributions.

    Attributes:
        mean: Mean of the distribution.
        covariance: Covariance of the distribution.
        dimension: Dimensionality of the distribution.
    """

    def __init__(self, mean: np.ndarray, covariance: np.ndarray, dimension: int) -> None:
        """Initialize distribution.

        Args:
            mean: Mean of the distribution
            covariance: Covariance of the distribution
            dimension: Dimensionality of the distribution
        """
        self.mean = mean
        self.covariance = covariance
        self.dimension = dimension

    @abstractmethod
    def cdf(self, x: np.ndarray) -> Any:
        """Cumulative distribution function.

        Args:
            x: Positions at which the CDF is evaluated
        """

    @abstractmethod
    def draw(self, num_draws: int = 1) -> Any:
        """Draw samples.

        Args:
            num_draws: Number of draws
        """

    @abstractmethod
    def logpdf(self, x: np.ndarray) -> Any:
        """Log of the probability density function.

        Args:
            x: Positions at which the log-PDF is evaluated
        """

    @abstractmethod
    def grad_logpdf(self, x: np.ndarray) -> Any:
        """Gradient of the log-PDF with respect to *x*.

        Args:
            x: Positions at which the gradient of log-PDF is evaluated
        """

    def pdf(self, x: np.ndarray) -> Any:
        """Probability density function.

        Args:
            x: Positions at which the PDF is evaluated

        Returns:
            PDF at positions
        """
        pdf = np.exp(self.logpdf(x))
        return pdf

    @abstractmethod
    def ppf(self, quantiles: np.ndarray) -> Any:
        """Percent point function (inverse of CDF — quantiles).

        Args:
            quantiles: Quantiles at which the PPF is evaluated
        """

    def check_1d(self) -> None:
        """Check if distribution is one-dimensional."""
        if self.dimension != 1:
            raise ValueError("Method does not support multivariate distributions!")

    @staticmethod
    def check_bounds(lower_bound: np.ndarray, upper_bound: np.ndarray) -> None:
        """Check sanity of bounds.

        Args:
            lower_bound: Lower bound(s) of distribution
            upper_bound: Upper bound(s) of distribution
        """
        if (upper_bound <= lower_bound).any():
            raise ValueError(
                f"Lower bound must be smaller than upper bound. "
                f"You specified lower_bound={lower_bound} and upper_bound={upper_bound}"
            )


class Discrete(Distribution):
    """Discrete distribution base class.

    Attributes:
        mean: Mean of the distribution.
        covariance: Covariance of the distribution.
        dimension: Dimensionality of the distribution.
        probabilities: Probabilities associated to all the events in the sample space
        sample_space: Samples, i.e. possible outcomes of sampling the distribution
    """

    def __init__(
        self,
        probabilities: ArrayLike,
        sample_space: np.ndarray | Sequence[Sized],
        dimension: int | None = None,
    ) -> None:
        """Initialize the discrete distribution.

        Args:
            probabilities: Probabilities associated to all the events in the sample space
            sample_space: Samples, i.e. possible outcomes of sampling the distribution
            dimension: Dimension of a sample event
        """
        if len({len(d) for d in sample_space}) != 1:
            raise ValueError("Dimensions of the sample events do not match.")

        sample_space_array = np.array(sample_space).reshape(len(sample_space), -1)
        probabilities_array = np.array(probabilities)

        if dimension is None:
            self.dimension = sample_space_array[0].shape[0]
        else:
            if not isinstance(dimension, int) or dimension <= 0:
                raise ValueError(f"Dimension has to be a positive integer not {dimension}.")
            self.dimension = dimension

        if len(sample_space_array) != len(probabilities_array):
            raise ValueError(
                f"The number of probabilities {len(probabilities_array)} does not match the number "
                f"of events in the sample space {len(sample_space_array)}"
            )

        super().check_positivity(probabilities=probabilities_array)

        if not np.isclose(np.sum(probabilities_array), 1, atol=0):
            _logger.info("Probabilities do not sum up to one, they are going to be normalized.")
            probabilities_array = probabilities_array / np.sum(probabilities_array)

        # Sort the sample events
        if self.dimension == 1:
            indices = np.argsort(sample_space_array.flatten())
            self.probabilities = probabilities_array[indices]
            self.sample_space = sample_space_array[indices]
        else:
            self.probabilities = probabilities_array
            self.sample_space = sample_space_array

        self.mean, self.covariance = self._compute_mean_and_covariance()

    @abstractmethod
    def draw(self, num_draws: int = 1) -> Any:
        """Draw samples.

        Args:
            num_draws: Number of draws
        """

    @abstractmethod
    def logpdf(self, x: np.ndarray) -> Any:
        """Log of the probability *mass* function.

        In order to keep the interfaces unified, the PMF is also accessed via the PDF.

        Args:
            x: Positions at which the log-PDF is evaluated
        """

    @abstractmethod
    def pdf(self, x: np.ndarray) -> Any:
        """Probability density function.

        Args:
            x: Positions at which the PDF is evaluated
        """

    @abstractmethod
    def cdf(self, x: np.ndarray) -> Any:
        """Cumulative distribution function.

        Args:
            x: Positions at which the CDF is evaluated
        """

    @abstractmethod
    def ppf(self, quantiles: np.ndarray) -> Any:
        """Percent point function (inverse of CDF - quantiles).

        Args:
            quantiles: Quantiles at which the PPF is evaluated
        """

    @abstractmethod
    def _compute_mean_and_covariance(self) -> tuple[np.ndarray, np.ndarray]:
        """Compute the mean value and covariance of the distribution.

        Returns:
            Mean value of the distribution
            Covariance of the distribution
        """

    def check_1d(self) -> None:
        """Check if distribution is one-dimensional."""
        if self.dimension != 1:
            raise ValueError("Method does not support multivariate distributions!")

    @staticmethod
    def check_duplicates_in_sample_space(sample_space: np.ndarray) -> None:
        """Check for duplicate events in the sample space.

        Args:
            sample_space: Samples, i.e. possible outcomes of sampling the distribution
        """
        if len(sample_space) != len(np.unique(sample_space, axis=0)):
            raise ValueError("The sample space contains duplicate events, this is not possible.")
