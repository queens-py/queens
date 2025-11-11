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
"""Piece-wise random fields class."""

from copy import deepcopy

import numpy as np

from queens.distributions._distribution import Continuous, Discrete
from queens.parameters.parameters import HasGradLogPDF
from queens.parameters.random_fields._random_field import RandomField


class PieceWise(RandomField):
    """Piece-wise random field class.

    The field is constructed at each node or element which is given by the
    coordinates. Each piece is assumed independently. Equals creating
    random variables for each point in coords.

    Attributes:
            distribution: Dummy distribution with correct dimension
            latent_1d_distribution: QUEENS distribution object of latent space variables
    """

    def __init__(self, coords: dict, latent_1d_distribution: Continuous | Discrete):
        """Initialize RF object.

        Args:
            coords: Dictionary with coordinates of discretized random field and the corresponding
                keys
            latent_1d_distribution: Latent 1d distribution that is used for all variables.
        """
        super().__init__(coords, deepcopy(latent_1d_distribution), len(coords["keys"]))

        self.latent_1d_distribution = latent_1d_distribution
        self.distribution.dimension = self.dimension

    def draw(self, num_samples: int) -> np.ndarray:
        """Draw samples from the latent representation of the random field.

        Args:
            num_samples: Number of draws of latent random samples
        Returns:
            Drawn samples
        """
        samples = self.latent_1d_distribution.draw(num_samples * self.dimension).reshape(
            num_samples, self.dimension
        )
        return samples

    def logpdf(self, samples: np.ndarray) -> np.ndarray:
        """Get joint log-PDF of latent space.

        Args:
            samples: Latent space samples

        Returns:
            Log-PDF of the samples
        """
        return (
            self.latent_1d_distribution.logpdf(samples.reshape(-1, 1))
            .reshape(samples.shape)
            .sum(axis=1)
        )

    def grad_logpdf(self, samples: np.ndarray) -> np.ndarray:
        """Get gradient of joint log-PDF of latent space.

        Args:
            samples: Latent space samples

        Returns:
            Gradient of the log-PDF of the samples
        """
        if not isinstance(self.latent_1d_distribution, HasGradLogPDF):
            raise TypeError(
                f"The latent 1D distribution {self.latent_1d_distribution} does not have a "
                "grad_logpdf function."
            )

        return self.latent_1d_distribution.grad_logpdf(samples.reshape(-1, 1)).reshape(
            samples.shape
        )

    def expanded_representation(self, samples: np.ndarray) -> np.ndarray:
        """Expand latent representation of samples.

        Args:
            samples: Latent representation of samples

        Returns:
            Expanded representation of samples
        """
        return samples

    def latent_gradient(self, upstream_gradient: np.ndarray) -> np.ndarray:
        """Gradient of the field with respect to the latent parameters.

        Args:
            upstream_gradient: Gradient with respect to all coords of the field

        Returns:
            Gradient of the field with respect to the latent parameters
        """
        return upstream_gradient
