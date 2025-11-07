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
"""Random fields module."""

import abc
from typing import TYPE_CHECKING

import numpy as np

from queens.utils.numpy_array import at_least_2d

if TYPE_CHECKING:
    from queens.distributions._distribution import Continuous, Discrete


class RandomField(metaclass=abc.ABCMeta):
    """RandomField meta class.

    Attributes:
            dimension: Dimension of the latent space.
            coords: Coordinates at which the random field is evaluated.
            dim_coords: Dimension of the random field (number of coordinates)
            distribution: QUEENS distribution object of latent space variables
    """

    def __init__(self, coords: dict, distribution: "Continuous | Discrete", dimension: int) -> None:
        """Initialize random field object.

        Args:
            coords: Dictionary with coordinates of discretized random field and the corresponding
                keys
            distribution: QUEENS distribution object of latent space variables
            dimension: Dimension of the latent space.
        """
        self.coords = self._convert_coords_to_2d_array(coords)
        self.distribution = distribution
        self.dim_coords = len(coords["keys"])
        self.dimension = dimension

    @abc.abstractmethod
    def draw(self, num_samples: int) -> np.ndarray:
        """Draw samples of the latent space.

        Args:
            num_samples (int): Batch size of samples to draw

        Returns:
            Drawn samples
        """

    @abc.abstractmethod
    def expanded_representation(self, samples: np.ndarray) -> np.ndarray:
        """Expand the random field realization.

        Args:
            samples (np.array): Latent space variables to be expanded into a random field

        Returns:
            Expanded representation of samples
        """

    @abc.abstractmethod
    def logpdf(self, samples: np.ndarray) -> np.ndarray:
        """Get joint log-PDF of latent space.

        Args:
            samples: Samples to evaluate log-PDF

        Returns:
            Log-PDF of the samples
        """

    @abc.abstractmethod
    def grad_logpdf(self, samples: np.ndarray) -> np.ndarray:
        """Get gradient of joint log-PDF of latent space.

        Args:
            samples: Samples to evaluate gradient of log-PDF

        Returns:
            Gradient of the log-PDF
        """

    def latent_gradient(self, upstream_gradient: np.ndarray) -> np.ndarray:
        """Gradient of the field with respect to the latent parameters.

        Args:
            upstream_gradient: Gradient with respect to all coords of the field

        Returns:
            Gradient of the field with respect to the latent parameters
        """
        raise NotImplementedError("latent_gradient method not implemented for this random field.")

    @staticmethod
    def _convert_coords_to_2d_array(coords: dict) -> dict:
        """Convert the coords to a 2D numpy array.

        Args:
            coords: Dictionary containing the coordinates of the random field.

        Returns:
            Dictionary with 2D numpy array coordinates.
        """
        # ensure that coordinates are an ndarray
        coords["coords"] = np.array(coords["coords"], copy=False)

        # ensure correct shape:
        # convert coords to a 2D column vector if necessary
        coords["coords"] = at_least_2d(coords["coords"])

        return coords
