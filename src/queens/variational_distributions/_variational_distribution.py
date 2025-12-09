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
from typing import Any, Literal, TypeAlias, TypeVar

import numpy as np

# pylint: disable=invalid-name

NDims = TypeVar("NDims", bound=int)
NSamples = TypeVar("NSamples", bound=int)
NParams = TypeVar("NParams", bound=int)
NParamsComponent = TypeVar("NParamsComponent", bound=int)

# Vectors
ArrayNDims: TypeAlias = np.ndarray[tuple[NDims], np.dtype[np.floating]]
ArrayNParams: TypeAlias = np.ndarray[tuple[NParams], np.dtype[np.floating]]
ArrayNParamsComponent: TypeAlias = np.ndarray[tuple[NParamsComponent], np.dtype[np.floating]]
ArrayNSamples: TypeAlias = np.ndarray[tuple[NSamples], np.dtype[np.floating]]

# Matrices
Array1XNParams: TypeAlias = np.ndarray[tuple[Literal[1], NParams], np.dtype[np.floating]]
ArrayNDimsX1: TypeAlias = np.ndarray[tuple[NDims, Literal[1]], np.dtype[np.floating]]
ArrayNDimsXNDims: TypeAlias = np.ndarray[tuple[NDims, NDims], np.dtype[np.floating]]
ArrayNParamsXNParams: TypeAlias = np.ndarray[tuple[NParams, NParams], np.dtype[np.floating]]
ArrayNParamsXNSamples: TypeAlias = np.ndarray[tuple[NParams, NSamples], np.dtype[np.floating]]
ArrayNSamplesXNDims: TypeAlias = np.ndarray[tuple[NSamples, NDims], np.dtype[np.floating]]
ArrayNSamplesXNParams: TypeAlias = np.ndarray[tuple[NSamples, NParams], np.dtype[np.floating]]


class Variational:
    """Base class for probability distributions for variational inference.

    Attributes:
        dimension: Dimension of the distribution
        n_parameters: Number of variational parameters
    """

    def __init__(self, dimension: NDims, n_parameters: NParams) -> None:
        """Initialize variational distribution.

        Args:
            dimension: Dimension of the variational distribution
            n_parameters: Number of variational parameters
        """
        self.dimension = dimension
        self.n_parameters = n_parameters

    @abc.abstractmethod
    def construct_variational_parameters(self, *args: Any) -> ArrayNParams:
        """Construct variational parameters from distribution parameters.

        Args:
            args: Distribution parameters

        Returns:
            Variational parameters
        """

    @abc.abstractmethod
    def reconstruct_distribution_parameters(self, variational_parameters: ArrayNParams) -> Any:
        """Reconstruct distribution parameters from variational parameters.

        Args:
            variational_parameters: Variational parameters

        Returns:
            Distribution parameters
        """

    @abc.abstractmethod
    def draw(self, variational_parameters: ArrayNParams, n_draws: NSamples) -> ArrayNSamplesXNDims:
        """Draw *n_draws* samples from distribution.

        Args:
            variational_parameters: Variational parameters
            n_draws: Number of samples

        Returns:
            Drawn samples
        """

    @abc.abstractmethod
    def logpdf(
        self,
        variational_parameters: ArrayNParams,
        x: ArrayNSamplesXNDims,
    ) -> ArrayNSamples:
        """Evaluate the natural logarithm of the PDF.

        Args:
            variational_parameters: Variational parameters
            x: Locations to evaluate

        Returns:
            Log-PDF values
        """

    @abc.abstractmethod
    def pdf(
        self,
        variational_parameters: ArrayNParams,
        x: ArrayNSamplesXNDims,
    ) -> ArrayNSamples:
        """Evaluate the probability density function (PDF).

        Args:
            variational_parameters: Variational parameters
            x: Locations to evaluate

        Returns:
            PDF values
        """

    @abc.abstractmethod
    def grad_params_logpdf(
        self,
        variational_parameters: ArrayNParams,
        x: ArrayNSamplesXNDims,
    ) -> ArrayNParamsXNSamples:
        """Log-PDF gradient w.r.t. the variational parameters.

        Evaluated at samples  *x*. Also known as the score function.

        Args:
            variational_parameters: Variational parameters
            x: Locations to evaluate

        Returns:
            Gradient of the log-PDF w.r.t. the variational parameters
        """

    @abc.abstractmethod
    def fisher_information_matrix(
        self, variational_parameters: ArrayNParams
    ) -> ArrayNParamsXNParams:
        """Compute the Fisher information matrix.

        Depends on the variational distribution for the given
        parameterization.

        Args:
            variational_parameters: Variational parameters

        Returns:
            Fisher information matrix
        """

    @abc.abstractmethod
    def initialize_variational_parameters(self, random: bool = False) -> ArrayNParams:
        """Initialize variational parameters.

        Args:
            random: If True, a random initialization is used. Otherwise the default is selected.

        Returns:
            Variational parameters
        """

    @abc.abstractmethod
    def export_dict(self, variational_parameters: ArrayNParams) -> dict:
        """Create a dict of the distribution based on the given parameters.

        Args:
            variational_parameters: Variational parameters

        Returns:
            Dictionary containing distribution information
        """
