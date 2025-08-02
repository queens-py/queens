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

import numpy as np

from queens.distributions.particle import Particle as ParticleDistribution
from queens.variational_distributions._variational_distribution import Variational


class Particle(Variational):
    r"""Variational distribution for particle distributions.

    The probabilities of the distribution are parameterized by softmax:
    :math:`p_i=p(\lambda_i)=\frac{\exp(\lambda_i)}{\sum_k exp(\lambda_k)}`

    Attributes:
        particles_obj (Particle): Particle distribution object
        dimension (int): Number of random variables
    """

    def __init__(self, sample_space):
        """Initialize variational distribution."""
        self.particles_obj = ParticleDistribution(np.ones(len(sample_space)), sample_space)
        super().__init__(self.particles_obj.dimension)
        self.n_parameters = len(sample_space)

    def construct_variational_parameters(self, probabilities, sample_space):
        """Construct the variational parameters from the probabilities.

        Args:
            probabilities (np.ndarray): Probabilities of the distribution
            sample_space (np.ndarray): Sample space of the distribution

        Returns:
            variational_parameters (np.ndarray): Variational parameters
        """
        self.particles_obj = ParticleDistribution(probabilities, sample_space)
        variational_parameters = np.log(probabilities).flatten()
        return variational_parameters

    def initialize_variational_parameters(self, random=False):
        r"""Initialize variational parameters.

        Default initialization:
            :math:`w_i=\frac{1}{N_\text{sample space}}`

        Random intialization:
            :math:`w_i=\frac{s}{N_\text{experiments}}` where :math:`s` is a sample of a multinomial
            distribution with :math:`N_\text{experiments}`

        Args:
            random (bool, optional): If True, a random initialization is used. Otherwise the
                                     default is selected

        Returns:
            variational_parameters (np.ndarray):  variational parameters (1 x n_params)
        """
        if random:
            variational_parameters = (
                np.random.multinomial(100, [1 / self.n_parameters] * self.n_parameters) / 100
            )
            variational_parameters = np.log(variational_parameters)
        else:
            variational_parameters = np.log(np.ones(self.n_parameters) / self.n_parameters)

        return variational_parameters

    def reconstruct_distribution_parameters(self, variational_parameters):
        """Reconstruct probabilities from the variational parameters.

        Args:
            variational_parameters (np.ndarray): Variational parameters

        Returns:
            probabilities (np.ndarray): Probabilities of the distribution
        """
        probabilities = np.exp(variational_parameters)
        probabilities /= np.sum(probabilities)
        self.particles_obj = ParticleDistribution(probabilities, self.particles_obj.sample_space)
        return probabilities, self.particles_obj.sample_space

    def draw(self, variational_parameters, n_draws=1):
        """Draw *n_draws* samples from distribution.

        Args:
            variational_parameters (np.ndarray): Variational parameters of the distribution
            n_draws (int): Number of samples

        Returns:
            samples (np.ndarray): samples (n_draws x n_dim)
        """
        self.reconstruct_distribution_parameters(variational_parameters)
        return self.particles_obj.draw(n_draws)

    def logpdf(self, variational_parameters, x):
        """Evaluate the natural logarithm of the logpdf at sample.

        Args:
            variational_parameters (np.ndarray): Variational parameters of the distribution
            x (np.ndarray): Locations at which to evaluate the distribution (n_samples x n_dim)

        Returns:
            logpdf (np.ndarray): Logpdfs at the locations x
        """
        self.reconstruct_distribution_parameters(variational_parameters)
        return self.particles_obj.logpdf(x)

    def pdf(self, variational_parameters, x):
        """Evaluate the probability density function (pdf) at sample.

        Args:
            variational_parameters (np.ndarray): Variational parameters of the distribution
            x (np.ndarray): Locations at which to evaluate the distribution (n_samples x n_dim)

        Returns:
            logpdf (np.ndarray): Pdfs at the locations x
        """
        self.reconstruct_distribution_parameters(variational_parameters)
        return self.particles_obj.pdf(x)

    def grad_params_logpdf(self, variational_parameters, x):
        r"""Logpdf gradient w.r.t. the variational parameters.

        Evaluated at samples  *x*. Also known as the score function.

        For the given parameterization, the score function yields:
        :math:`\nabla_{\lambda_i}\ln p(\theta_j | \lambda)=\delta_{ij}-p_i`

        Args:
            variational_parameters (np.ndarray): Variational parameters of the distribution
            x (np.ndarray): Locations at which to evaluate the distribution (n_samples x n_dim)

        Returns:
            score_function (np.ndarray): Score functions at the locations x
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

    def fisher_information_matrix(self, variational_parameters):
        r"""Compute the fisher information matrix.

        For the given parameterization, the Fisher information yields:
        :math:`\text{FIM}_{ij}=\delta_{ij} p_i -p_i p_j`

        Args:
            variational_parameters (np.ndarray): Variational parameters of the distribution

        Returns:
            fim (np.ndarray): Fisher information matrix (n_params x n_params)
        """
        probabilities, _ = self.reconstruct_distribution_parameters(variational_parameters)
        fim = np.diag(probabilities) - np.outer(probabilities, probabilities)
        return fim

    def export_dict(self, variational_parameters):
        """Create a dict of the distribution based on the given parameters.

        Args:
            variational_parameters (np.ndarray): Variational parameters

        Returns:
            export_dict (dictionary): Dict containing distribution information
        """
        self.reconstruct_distribution_parameters(variational_parameters)
        export_dict = {
            "type": type(self),
            "variational_parameters": variational_parameters,
        }
        export_dict.update(self.particles_obj.export_dict())
        return export_dict
