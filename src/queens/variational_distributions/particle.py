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
"""Particle Variational Distribution."""

from typing import Sequence, Sized

import numpy as np

from queens.distributions.particle import Particle as ParticleDistribution
from queens.utils.type_hinting import ArrayN, ArrayNxM
from queens.variational_distributions._variational_distribution import Variational


class Particle(Variational):
    r"""Variational distribution for particle distributions.

    The probabilities of the distribution are parameterized by softmax:
    :math:`p_i=p(\lambda_i)=\frac{\exp(\lambda_i)}{\sum_k exp(\lambda_k)}`

    Attributes:
        particles_obj: Particle distribution object
        dimension: Number of random variables
    """

    def __init__(self, sample_space: np.ndarray | Sequence[Sized]) -> None:
        """Initialize variational distribution.

        Args:
            sample_space: Sample space of the variational distribution
        """
        self.particles_obj = ParticleDistribution(np.ones(len(sample_space)), sample_space)
        super().__init__(self.particles_obj.dimension, n_parameters=len(sample_space))

    def construct_variational_parameters(  # pylint: disable=arguments-differ
        self, probabilities: ArrayN, sample_space: np.ndarray
    ) -> ArrayN:
        """Construct the variational parameters from the probabilities.

        Args:
            probabilities: Probabilities of the distribution
            sample_space: Sample space of the distribution

        Returns:
            Variational parameters
        """
        self.particles_obj = ParticleDistribution(probabilities, sample_space)
        variational_parameters = np.log(probabilities).flatten()
        return variational_parameters

    def initialize_variational_parameters(self, random: bool = False) -> ArrayN:
        r"""Initialize variational parameters.

        Default initialization:
            :math:`w_i=\frac{1}{N_\text{sample space}}`

        Random intialization:
            :math:`w_i=\frac{s}{N_\text{experiments}}` where :math:`s` is a sample of a multinomial
            distribution with :math:`N_\text{experiments}`

        Args:
            random: If True, a random initialization is used. Otherwise the default is selected

        Returns:
            Variational parameters of shape (n_params,)
        """
        if random:
            variational_parameters = (
                np.random.multinomial(100, [1 / self.n_parameters] * self.n_parameters) / 100
            )
            variational_parameters = np.log(variational_parameters)
        else:
            variational_parameters = np.log(np.ones(self.n_parameters) / self.n_parameters)

        return variational_parameters

    def reconstruct_distribution_parameters(
        self, variational_parameters: ArrayN
    ) -> tuple[ArrayN, np.ndarray]:
        """Reconstruct probabilities from the variational parameters.

        Args:
            variational_parameters: Variational parameters

        Returns:
            Probabilities of the distribution
            Sample space of the distribution
        """
        probabilities = np.exp(variational_parameters)
        probabilities /= np.sum(probabilities)
        self.particles_obj = ParticleDistribution(probabilities, self.particles_obj.sample_space)
        return probabilities, self.particles_obj.sample_space

    def draw(self, variational_parameters: ArrayN, n_draws: int = 1) -> ArrayNxM:
        """Draw *n_draws* samples from distribution.

        Args:
            variational_parameters: Variational parameters of the distribution
            n_draws: Number of samples

        Returns:
            Samples of shape (n_draws, n_dim)
        """
        self.reconstruct_distribution_parameters(variational_parameters)
        return self.particles_obj.draw(n_draws)

    def logpdf(self, variational_parameters: ArrayN, x: ArrayNxM) -> np.ndarray:
        """Evaluate the natural logarithm of the PDF.

        Args:
            variational_parameters: Variational parameters of the distribution
            x: Locations at which to evaluate the distribution of shape (n_samples, n_dim)

        Returns:
            Log-PDF values at the locations x
        """
        self.reconstruct_distribution_parameters(variational_parameters)
        return self.particles_obj.logpdf(x)

    def pdf(self, variational_parameters: ArrayN, x: ArrayNxM) -> np.ndarray:
        """Evaluate the probability density function (PDF).

        Args:
            variational_parameters: Variational parameters of the distribution
            x: Locations at which to evaluate the distribution of shape (n_samples, n_dim)

        Returns:
            Row vector of the PDF values
        """
        self.reconstruct_distribution_parameters(variational_parameters)
        return self.particles_obj.pdf(x)

    def grad_params_logpdf(self, variational_parameters: ArrayN, x: ArrayNxM) -> np.ndarray:
        r"""Log-PDF gradient w.r.t. the variational parameters.

        Evaluated at samples  *x*. Also known as the score function.

        For the given parameterization, the score function yields:
        :math:`\nabla_{\lambda_i}\ln p(\theta_j | \lambda)=\delta_{ij}-p_i`

        Args:
            variational_parameters: Variational parameters of the distribution
            x: Locations at which to evaluate the distribution of shape (n_samples, n_dim)

        Returns:
            Score functions at the locations x
        """
        self.reconstruct_distribution_parameters(variational_parameters)
        index = np.array(
            [(self.particles_obj.sample_space == xi).all(axis=1).nonzero()[0] for xi in x]
        ).flatten()

        if len(index) != len(x):
            raise ValueError(
                f"At least one event is not part of the sample space "
                f"{self.particles_obj.sample_space}"
            )
        sample_scores = np.eye(len(variational_parameters)) - np.exp(
            variational_parameters
        ) / np.sum(np.exp(variational_parameters))
        # Get the samples
        return sample_scores[index].T

    def fisher_information_matrix(self, variational_parameters: ArrayN) -> ArrayNxM:
        r"""Compute the Fisher information matrix.

        For the given parameterization, the Fisher information yields:
        :math:`\text{FIM}_{ij}=\delta_{ij} p_i -p_i p_j`

        Args:
            variational_parameters: Variational parameters of the distribution

        Returns:
            Fisher information matrix of shape (n_parameters, n_parameters)
        """
        probabilities, _ = self.reconstruct_distribution_parameters(variational_parameters)
        fim = np.diag(probabilities) - np.outer(probabilities, probabilities)
        return fim

    def export_dict(self, variational_parameters: ArrayN) -> dict:
        """Create a dict of the distribution based on the given parameters.

        Args:
            variational_parameters: Variational parameters

        Returns:
            Dictionary containing distribution information
        """
        self.reconstruct_distribution_parameters(variational_parameters)
        export_dict = {
            "type": type(self),
            "variational_parameters": variational_parameters,
        }
        export_dict.update(self.particles_obj.export_dict())
        return export_dict
