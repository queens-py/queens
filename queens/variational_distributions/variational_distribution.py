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


class Variational:
    """Base class for probability distributions for variational inference.

    Attributes:
        dimension (int): dimension of the distribution
    """

    def __init__(self, dimension):
        """Initialize variational distribution."""
        self.dimension = dimension

    @abc.abstractmethod
    def reconstruct_distribution_parameters(self, variational_parameters):
        """Reconstruct distribution parameters from variational parameters.

        Args:
            variational_parameters (np.ndarray): Variational parameters
        """

    @abc.abstractmethod
    def draw(self, variational_parameters, n_draws=1):
        """Draw *n_draws* samples from distribution.

        Args:
           variational_parameters (np.ndarray):  variational parameters (1 x n_params)
           n_draws (int): Number of samples
        """

    @abc.abstractmethod
    def logpdf(self, variational_parameters, x):
        """Evaluate the natural logarithm of the logpdf at sample.

        Args:
            variational_parameters (np.ndarray):  variational parameters (1 x n_params)
            x (np.ndarray): Locations to evaluate (n_samples x n_dim)
        """

    @abc.abstractmethod
    def pdf(self, variational_parameters, x):
        """Evaluate the probability density function (pdf) at sample.

        Args:
            variational_parameters (np.ndarray):  variational parameters (1 x n_params)
            x (np.ndarray): Locations to evaluate (n_samples x n_dim)
        """

    @abc.abstractmethod
    def grad_params_logpdf(self, variational_parameters, x):
        """Logpdf gradient w.r.t. the variational parameters.

        Evaluated at samples  *x*. Also known as the score function.

        Args:
            variational_parameters (np.ndarray):  variational parameters (1 x n_params)
            x (np.ndarray): Locations to evaluate (n_samples x n_dim)
        """

    @abc.abstractmethod
    def fisher_information_matrix(self, variational_parameters):
        """Compute the fisher information matrix.

        Depends on the variational distribution for the given
        parameterization.

        Args:
            variational_parameters (np.ndarray):  variational parameters (1 x n_params)
        """

    @abc.abstractmethod
    def initialize_variational_parameters(self, random=False):
        """Initialize variational parameters.

        Args:
            random (bool, optional): If True, a random initialization is used. Otherwise the
                                     default is selected

        Returns:
            variational_parameters (np.ndarray):  variational parameters (1 x n_params)
        """

    @abc.abstractmethod
    def export_dict(self, variational_parameters):
        """Create a dict of the distribution based on the given parameters.

        Args:
            variational_parameters (np.ndarray): Variational parameters

        Returns:
            export_dict (dictionary): Dict containing distribution information
        """
