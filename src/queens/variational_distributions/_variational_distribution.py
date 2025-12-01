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
"""Variational Distribution."""

import abc
from typing import Any

import numpy as np

from queens.utils.type_hinting import ArrayN, ArrayNxM


class Variational:
    """Base class for probability distributions for variational inference.

    Attributes:
        dimension: Dimension of the distribution
        n_parameters: Number of variational parameters
    """

    def __init__(self, dimension: int, n_parameters: int) -> None:
        """Initialize variational distribution.

        Args:
            dimension: Dimension of the variational distribution
            n_parameters: Number of variational parameters
        """
        self.dimension = dimension
        self.n_parameters = n_parameters

    @abc.abstractmethod
    def construct_variational_parameters(self, *args: Any) -> np.ndarray:
        """Construct variational parameters from distribution parameters.

        Args:
            args: Distribution parameters

        Returns:
            Variational parameters
        """

    @abc.abstractmethod
    def reconstruct_distribution_parameters(self, variational_parameters: ArrayN) -> Any:
        """Reconstruct distribution parameters from variational parameters.

        Args:
            variational_parameters: Variational parameters

        Returns:
            Distribution parameters
        """

    @abc.abstractmethod
    def draw(
        self,
        variational_parameters: ArrayN,
        n_draws: int = 1,
    ) -> ArrayNxM:
        """Draw *n_draws* samples from distribution.

        Args:
            variational_parameters: Variational parameters of shape (n_params,)
            n_draws: Number of samples

        Returns:
            Drawn samples of shape (n_draws, n_dim)
        """

    @abc.abstractmethod
    def logpdf(
        self,
        variational_parameters: ArrayN,
        x: ArrayNxM,
    ) -> np.ndarray:
        """Evaluate the natural logarithm of the PDF.

        Args:
            variational_parameters: Variational parameters of shape (n_params,)
            x: Locations to evaluate of shape (n_samples, n_dim)

        Returns:
            Row vector of the Log-PDF values
        """

    @abc.abstractmethod
    def pdf(
        self,
        variational_parameters: ArrayN,
        x: ArrayNxM,
    ) -> np.ndarray:
        """Evaluate the probability density function (PDF).

        Args:
            variational_parameters: Variational parameters of shape (n_params,)
            x: Locations to evaluate of shape (n_samples, n_dim)

        Returns:
            Row vector of the PDF values
        """

    @abc.abstractmethod
    def grad_params_logpdf(
        self,
        variational_parameters: ArrayN,
        x: ArrayNxM,
    ) -> np.ndarray:
        """Log-PDF gradient w.r.t. the variational parameters.

        Evaluated at samples  *x*. Also known as the score function.

        Args:
            variational_parameters: Variational parameters of shape (n_params,)
            x: Locations to evaluate of shape (n_samples, n_dim)

        Returns:
            Gradient of the log-PDF w.r.t. the variational parameters
        """

    @abc.abstractmethod
    def fisher_information_matrix(self, variational_parameters: ArrayN) -> ArrayNxM:
        """Compute the fisher information matrix.

        Depends on the variational distribution for the given
        parameterization.

        Args:
            variational_parameters: Variational parameters of shape (n_params,)

        Returns:
            Fisher information matrix of shape (n_params, n_params)
        """

    @abc.abstractmethod
    def initialize_variational_parameters(self, random: bool = False) -> ArrayN:
        """Initialize variational parameters.

        Args:
            random: If True, a random initialization is used. Otherwise the default is selected.

        Returns:
            Variational parameters of shape (n_params,)
        """

    @abc.abstractmethod
    def export_dict(self, variational_parameters: np.ndarray) -> dict:
        """Create a dict of the distribution based on the given parameters.

        Args:
            variational_parameters: Variational parameters

        Returns:
            Dictionary containing distribution information
        """
