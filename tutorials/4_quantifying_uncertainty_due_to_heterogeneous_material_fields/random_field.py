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
"""CustomRandomField module for tutorial 4."""
from collections.abc import Callable

import numpy as np

from queens.distributions._distribution import Continuous, Discrete
from queens.parameters.random_fields._random_field import RandomField


class CustomRandomField(RandomField):
    """Random field defined by an explicit expansion function.

    Attributes:
            latent_distribution: QUEENS distribution object of random field
            expansion: Callable mapping ``(theta, coords) -> field``.
            coordinates: Coordinates at which the random field is evaluated.
    """

    def __init__(
        self, coords: dict, latent_distribution: Continuous | Discrete, expansion: Callable
    ) -> None:
        """Initialize the custom random field.

        Args:
            coords: Dictionary with coordinates at which the random field is evaluated.
            latent_distribution: Distribution of the latent parameters ``theta``.
            expansion: Function computing field values from ``(theta, coords)``.
        """
        super().__init__(coords, latent_distribution, latent_distribution.dimension)
        self.latent_distribution = latent_distribution
        self.expansion = expansion
        self.coordinates = coords["coords"]

    def draw(self, num_samples: int) -> np.ndarray:
        """Draw ``num_samples`` samples of the latent space.

        Args:
            num_samples (int): Batch size of samples to draw
        """
        return self.latent_distribution.draw(num_samples)

    def expanded_representation(self, samples: np.ndarray) -> np.ndarray:
        """Expand the random field realization.

        Args:
            samples (np.array): Latent space variables to be expanded into a random field
        """
        if samples.ndim == 1:
            return self.expansion(samples, self.coordinates)

        expansions = []

        for sample in samples:
            expansions.append(self.expansion(sample, self.coordinates))

        return np.array(expansions)

    def logpdf(self, samples: np.ndarray) -> np.ndarray:
        """Get joint logpdf of latent space.

        Args:
            samples (np.array): Sample to evaluate logpdf
        """
        return self.latent_distribution.logpdf(samples)

    def grad_logpdf(self, samples: np.ndarray) -> np.ndarray:
        """Get gradient of joint logpdf of latent space.

        Args:
            samples (np.array): Sample to evaluate gradient of logpdf
        """
        raise NotImplementedError()

    def latent_gradient(self, upstream_gradient: np.ndarray) -> np.ndarray:
        """Gradient of the field with respect to the latent variables."""
        raise NotImplementedError()
