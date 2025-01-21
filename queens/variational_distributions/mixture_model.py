#
# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (c) 2025, QUEENS contributors.
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
"""Mixture Model Variational Distribution."""

import numpy as np

from queens.variational_distributions.variational_distribution import VariationalDistribution


class MixtureModelVariational(VariationalDistribution):
    r"""Mixture model variational distribution class.

    Every component is a member of the same distribution family. Uses the parameterization:
    :math:`parameters=[\lambda_0,\lambda_1,...,\lambda_{C},\lambda_{weights}]`
    where :math:`C` is the number of components, :math:`\\lambda_i` are the variational parameters
    of the ith component and :math:`\\lambda_{weights}` parameters such that the component weights
    are obtained by:
    :math:`weight_i=\frac{exp(\lambda_{weights,i})}{\sum_{j=1}^{C}exp(\lambda_{weights,j})}`

    This allows the weight parameters :math:`\lambda_{weights}` to be unconstrained.

    Attributes:
        n_components (int): Number of mixture components.
        base_distribution: Variational distribution object for the components.
        n_parameters (int): Number of parameters used in the parameterization.
    """

    def __init__(self, base_distribution, dimension, n_components):
        """Initialize mixture model.

        Args:
            dimension (int): Dimension of the random variable
            n_components (int): Number of mixture components
            base_distribution: Variational distribution object for the components
        """
        super().__init__(dimension)
        self.n_components = n_components
        self.base_distribution = base_distribution
        self.n_parameters = n_components * base_distribution.n_parameters

    def initialize_variational_parameters(self, random=False):
        r"""Initialize variational parameters.

        Default weights initialization:
            :math:`w_i=\frac{1}{N_\text{sample space}}`

        Random weights intialization:
            :math:`w_i=\frac{s}{N_\text{experiments}}` where :math:`s` is a sample of a multinomial
            distribution with :math:`N_\text{experiments}`

        The component initialization is handle by the component itself.

        Args:
            random (bool, optional): If True, a random initialization is used. Otherwise the
                                     default is selected

        Returns:
            variational_parameters (np.ndarray):  variational parameters (1 x n_params)
        """
        variational_parameters_components = (
            self.base_distribution.initialize_variational_parameters(random)
        )
        # Repeat for each component

        variational_parameters_components = np.tile(
            variational_parameters_components, self.n_components
        )
        if random:
            variational_parameters_weights = (
                np.random.multinomial(100, [1 / self.n_parameters] * self.n_parameters) / 100
            )
            variational_parameters_weights = np.log(variational_parameters_weights)
        else:
            variational_parameters_weights = np.log(np.ones(self.n_parameters) / self.n_parameters)

        return np.concatenate([variational_parameters_components, variational_parameters_weights])

    def construct_variational_parameters(self, component_parameters_list, weights):
        """Construct the variational parameters from the probabilities.

        Args:
            component_parameters_list (list): List of the component parameters of the components
            weights (np.ndarray): Probabilities of the distribution

        Returns:
            variational_parameters (np.ndarray): Variational parameters
        """
        variational_parameters = []
        for component_parameters in component_parameters_list:
            variational_parameters.append(
                self.base_distribution.construct_variational_parameters(*component_parameters)
            )
        variational_parameters.append(np.log(weights).flatten())
        return np.concatenate(variational_parameters)

    def _construct_component_variational_parameters(self, variational_parameters):
        """Reconstruct the weights and parameters of the mixture components.

        Creates a list containing the variational parameters of the different components.

        The list is nested, each entry correspond to the parameters of a component.

        Args:
            variational_parameters (np.ndarray): Variational parameters

        Returns:
            variational_parameters_list (list): List of the variational parameters of the components
            weights (np.ndarray): Weights of the mixture
        """
        n_parameters_comp = self.base_distribution.n_parameters
        variational_parameters_list = []
        for j in range(self.n_components):
            params_comp = variational_parameters[
                n_parameters_comp * j : n_parameters_comp * (j + 1)
            ]
            variational_parameters_list.append(params_comp)
        # Compute the weights from the weight parameters
        weights = np.exp(variational_parameters[-self.n_components :])
        weights = weights / np.sum(weights)
        return variational_parameters_list, weights

    def reconstruct_distribution_parameters(self, variational_parameters):
        """Reconstruct the weights and parameters of the mixture components.

        The list is nested, each entry correspond to the parameters of a component.

        Args:
            variational_parameters (np.ndarray): Variational parameters

        Returns:
            distribution_parameters_list (list): List of the distribution parameters of the
                                                 components
            weights (np.ndarray): Weights of the mixture
        """
        n_parameters_comp = self.base_distribution.n_parameters
        distribution_parameters_list = []
        for j in range(self.n_components):
            params_comp = variational_parameters[
                n_parameters_comp * j : n_parameters_comp * (j + 1)
            ]
            distribution_parameters_list.append(
                self.base_distribution.reconstruct_distribution_parameters(params_comp)
            )

        # Compute the weights from the weight parameters
        weights = np.exp(variational_parameters[-self.n_components :])
        weights = weights / np.sum(weights)
        return distribution_parameters_list, weights

    def draw(self, variational_parameters, n_draws=1):
        """Draw *n_draw* samples from the variational distribution.

        Uses a two-step process:
            1. From a multinomial distribution, based on the weights, select a component
            2. Sample from the selected component

        Args:
            variational_parameters (np.ndarray): Variational parameters
            n_draws (int): Number of samples to draw

        Returns:
            samples (np.ndarray): Row wise samples of the variational distribution
        """
        parameters_list, weights = self._construct_component_variational_parameters(
            variational_parameters
        )
        samples = []
        for _ in range(n_draws):
            # Select component to draw from
            component = np.argmax(np.random.multinomial(1, weights))
            # Draw a sample of this component
            sample = self.base_distribution.draw(parameters_list[component], 1)
            samples.append(sample)
        samples = np.concatenate(samples, axis=0)
        return samples

    def logpdf(self, variational_parameters, x):
        """Logpdf evaluated using the variational parameters at samples *x*.

        Is a general implementation using the logpdf function of the components. Uses the
        log-sum-exp trick [1] in order to reduce floating point issues.

        References:
        [1] :  David M. Blei, Alp Kucukelbir & Jon D. McAuliffe (2017) Variational Inference: A
               Review for Statisticians, Journal of the American Statistical Association, 112:518

        Args:
            variational_parameters (np.ndarray): Variational parameters
            x (np.ndarray): Row-wise samples

        Returns:
            logpdf (np.ndarray): Row vector of the logpdfs
        """
        parameters_list, weights = self._construct_component_variational_parameters(
            variational_parameters
        )
        logpdf = []
        x = np.atleast_2d(x)
        # Parameter for the log-sum-exp trick
        max_logpdf = -np.inf * np.ones(len(x))
        for j in range(self.n_components):
            logpdf.append(np.log(weights[j]) + self.base_distribution.logpdf(parameters_list[j], x))
            max_logpdf = np.maximum(max_logpdf, logpdf[-1])
        logpdf = np.array(logpdf) - np.tile(max_logpdf, (self.n_components, 1))
        logpdf = np.sum(np.exp(logpdf), axis=0)
        logpdf = np.log(logpdf) + max_logpdf
        return logpdf

    def pdf(self, variational_parameters, x):
        """Pdf evaluated using the variational parameters at given samples `x`.

        Args:
            variational_parameters (np.ndarray): Variational parameters
            x (np.ndarray): Row-wise samples

        Returns:
            pdf (np.ndarray): Row vector of the pdfs
        """
        pdf = np.exp(self.logpdf(variational_parameters, x))
        return pdf

    def grad_params_logpdf(self, variational_parameters, x):
        """Logpdf gradient w.r.t. the variational parameters.

        Evaluated at samples *x*. Also known as the score function.
        Is a general implementation using the score functions of
        the components.

        Args:
            variational_parameters (np.ndarray): Variational parameters
            x (np.ndarray): Row-wise samples

        Returns:
            score (np.ndarray): Column-wise scores
        """
        parameters_list, weights = self._construct_component_variational_parameters(
            variational_parameters
        )
        x = np.atleast_2d(x)
        # Jacobian of the weights w.r.t. weight parameters
        jacobian_weights = np.diag(weights) - np.outer(weights, weights)
        # Score function entries due to the parameters of the components
        component_block = []
        # Score function entries due to the weight parameterization
        weights_block = np.zeros((self.n_components, len(x)))
        logpdf = self.logpdf(variational_parameters, x)
        for j in range(self.n_components):
            # coefficient for the score term of every component
            precoeff = np.exp(self.base_distribution.logpdf(parameters_list[j], x) - logpdf)
            # Score function of the jth component
            score_comp = self.base_distribution.grad_params_logpdf(parameters_list[j], x)
            component_block.append(
                weights[j] * np.tile(precoeff, (len(score_comp), 1)) * score_comp
            )
            weights_block += np.tile(precoeff, (self.n_components, 1)) * jacobian_weights[
                :, j
            ].reshape(-1, 1)
        score = np.vstack((np.concatenate(component_block, axis=0), weights_block))
        return score

    def fisher_information_matrix(self, variational_parameters, n_samples=10000):
        """Approximate the Fisher information matrix using Monte Carlo.

        Args:
            variational_parameters (np.ndarray): Variational parameters
            n_samples (int, optional): number of samples for a MC FIM estimation

        Returns:
            FIM (np.ndarray): Matrix (num parameters x num parameters)
        """
        samples = self.draw(variational_parameters, n_samples)
        scores = self.grad_params_logpdf(variational_parameters, samples)
        fim = 0
        for j in range(n_samples):
            fim = fim + np.outer(scores[:, j], scores[:, j])
        fim = fim / n_samples
        return fim

    def export_dict(self, variational_parameters):
        """Create a dict of the distribution based on the given parameters.

        Args:
            variational_parameters (np.ndarray): Variational parameters

        Returns:
            export_dict (dictionary): Dict containing distribution information
        """
        parameters_list, weights = self._construct_component_variational_parameters(
            variational_parameters
        )
        export_dict = {
            "type": "mixture_model",
            "dimension": self.dimension,
            "n_components": self.n_components,
            "weights": weights,
            "variational_parameters": variational_parameters,
        }
        # Loop over the components
        for j in range(self.n_components):
            component_dict = self.base_distribution.export_dict(parameters_list[j])
            component_key = "component_" + str(j)
            export_dict.update({component_key: component_dict})
        return export_dict
