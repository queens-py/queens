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
"""General categorical distribution.

Disclaimer: Most of our iterators are not able to handle categorical distributions.
"""

import itertools
import logging

import numpy as np

from queens.distributions._distribution import Distribution
from queens.utils.logger_settings import log_init_args

_logger = logging.getLogger(__name__)


class Categorical(Distribution):
    """General categorical distribution.

    Attributes:
        probabilities: Probabilities associated with the categories
        categories: Categories
    """

    @log_init_args
    def __init__(self, probabilities: np.ndarray, categories: np.ndarray) -> None:
        """Initialize categorical distribution.

        Args:
            probabilities: Probabilities associated with the categories
            categories: Categories
        """
        categories = np.array(categories, dtype=object)
        probabilities = np.array(probabilities)

        if len(categories) != len(probabilities):
            raise ValueError(
                f"The number of probabilities {len(probabilities)} does not match the number of"
                f" categories {len(categories)}"
            )

        super().check_positivity(probabilities=probabilities)

        if not np.isclose(np.sum(probabilities), 1, atol=0):
            _logger.info("Probabilities do not sum up to one, they are going to be normalized.")
            probabilities = probabilities / np.sum(probabilities)

        self.probabilities = probabilities
        self.categories = categories

    def draw(self, num_draws: int = 1) -> np.ndarray:
        """Draw samples.

        Args:
            num_draws: Number of draws

        Returns:
            Samples of the categorical distribution
        """
        samples_per_category = np.random.multinomial(num_draws, self.probabilities)
        samples_tuple = (
            [
                [self.categories[category]] * repetitions
                for category, repetitions in enumerate(samples_per_category)
                if repetitions
            ],
        )
        samples = np.array(
            list(itertools.chain.from_iterable(*samples_tuple)),
            dtype=object,
        )
        np.random.shuffle(samples)
        return samples.reshape(-1, 1)

    def logpdf(self, x: np.ndarray) -> np.ndarray:
        """Log of the probability *mass* function.

        Args:
            x: Positions at which the log-PMF is evaluated

        Returns:
            Log PMF at positions
        """
        return np.log(self.pdf(x))

    def pdf(self, x: np.ndarray) -> np.ndarray:
        """Probability *mass* function.

        Args:
            x: Positions at which the PMF is evaluated

        Returns:
            PMF at positions
        """
        index = np.array([np.argwhere(self.categories == xi) for xi in x]).flatten()
        return self.probabilities[index]
