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
"""Joint Variational Distribution."""

import numpy as np
import scipy

from queens.variational_distributions.variational_distribution import VariationalDistribution


class JointVariational(VariationalDistribution):
    r"""Joint variational distribution class.

    This distribution allows to join distributions in an independent fashion:
    :math:`q(\theta|\lambda)=\prod_{i=1}^{N}q_i(\theta_i | \lambda_i)`

    NOTE: :math:`q_i(\theta_i | \lambda_i)` can be multivariate or of different families. Hence it
    is a generalization of the mean field distribution

    Attributes:
        distributions (list): List of variational distribution objects for the different
                              independent distributions.
        n_parameters (int): Total number of parameters used in the parameterization.
        distributions_n_parameters (np.ndarray): Number of parameters per distribution
        distributions_dimension (np.ndarray): Number of dimension per distribution
    """

    def __init__(self, distributions, dimension):
        """Initialize joint distribution.

        Args:
            dimension (int): Dimension of the random variable
            distributions (list): List of variational distribution objects for the different
                                  independent distributions.
        """
        super().__init__(dimension)
        self.distributions = distributions

        self.distributions_n_parameters = np.array(
            [distribution.n_parameters for distribution in distributions]
        ).astype(int)

        self.n_parameters = int(np.sum(self.distributions_n_parameters))

        self.distributions_dimension = np.array(
            [distribution.dimension for distribution in distributions]
        ).astype(int)

        if dimension != np.sum(self.distributions_dimension):
            raise ValueError(
                f"The provided total dimension {dimension} of the distribution does not match the "
                f"dimensions of the subdistributions {np.sum(self.distributions_dimension)}"
            )

    def initialize_variational_parameters(self, random=False):
        r"""Initialize variational parameters.

        The distribution initialization is handle by the component itself.

        Args:
            random (bool, optional): If True, a random initialization is used. Otherwise the
                                     default is selected

        Returns:
            variational_parameters (np.ndarray):  variational parameters (1 x n_params)
        """
        variational_parameters = np.concatenate(
            [
                distribution.initialize_variational_parameters(random)
                for distribution in self.distributions
            ]
        )

        return variational_parameters

    def construct_variational_parameters(self, distributions_parameters_list):
        """Construct the variational parameters from the distribution list.

        Args:
            distributions_parameters_list (list): List of the parameters of the distributions

        Returns:
            variational_parameters (np.ndarray): Variational parameters
        """
        variational_parameters = []
        for parameters, distribution in zip(
            distributions_parameters_list, self.distributions, strict=True
        ):
            variational_parameters.append(
                distribution.construct_variational_parameters(*parameters)
            )
        return np.concatenate(variational_parameters)

    def _construct_distributions_variational_parameters(self, variational_parameters):
        """Reconstruct the parameters of the distributions.

        Creates a list containing the variational parameters of the different components.

        The list is nested, each entry correspond to the parameters of a distribution.

        Args:
            variational_parameters (np.ndarray): Variational parameters

        Returns:
            variational_parameters_list (list): List of the variational parameters of the components
        """
        variational_parameters_list = split_array_by_chunk_sizes(
            variational_parameters, self.distributions_n_parameters
        )
        return variational_parameters_list

    def reconstruct_distribution_parameters(self, variational_parameters):
        """Reconstruct the parameters of distributions.

        The list is nested, each entry correspond to the parameters of a distribution.

        Args:
            variational_parameters (np.ndarray): Variational parameters

        Returns:
            distribution_parameters_list (list): List of the distribution parameters of the
                                                 components
        """
        distribution_parameters_list = []

        for parameters, distribution in self._zip_variational_parameters_distributions(
            variational_parameters
        ):
            distribution_parameters_list.append(
                distribution.reconstruct_distribution_parameters(parameters)
            )

        return [distribution_parameters_list]

    def _zip_variational_parameters_distributions(self, variational_parameters):
        """Zip parameters and distributions.

        This helper function creates a generator for variational parameters and subdistribution.

        Args:
            variational_parameters (np.ndarray): Variational parameters

        Returns:
            zip: of variational parameters and distributions
        """
        return zip(
            split_array_by_chunk_sizes(variational_parameters, self.distributions_n_parameters),
            self.distributions,
            strict=True,
        )

    def _zip_variational_parameters_distributions_samples(self, variational_parameters, samples):
        """Zip parameters, samples and distributions.

        This helper function creates a generator for variational parameters, samples and
        subdistribution.

        Args:
            variational_parameters (np.ndarray): Variational parameters
            samples (np.ndarray): Row-wise samples

        Returns:
            zip: of variational parameters, samples and distributions
        """
        return zip(
            split_array_by_chunk_sizes(variational_parameters, self.distributions_n_parameters),
            split_array_by_chunk_sizes(samples, self.distributions_dimension),
            self.distributions,
            strict=True,
        )

    def draw(self, variational_parameters, n_draws=1):
        """Draw *n_draw* samples from the variational distribution.

        Args:
            variational_parameters (np.ndarray): Variational parameters
            n_draws (int): Number of samples to draw

        Returns:
            samples (np.ndarray): Row wise samples of the variational distribution
        """
        sample_array = []
        for parameters, distribution in self._zip_variational_parameters_distributions(
            variational_parameters
        ):
            sample_array.append(distribution.draw(parameters, n_draws))
        return np.column_stack(sample_array)

    def logpdf(self, variational_parameters, x):
        """Logpdf evaluated using the variational parameters at samples *x*.

        Args:
            variational_parameters (np.ndarray): Variational parameters
            x (np.ndarray): Row-wise samples

        Returns:
            logpdf (np.ndarray): Row vector of the logpdfs
        """
        logpdf = 0
        for (
            parameters,
            samples,
            distribution,
        ) in self._zip_variational_parameters_distributions_samples(variational_parameters, x):
            logpdf += distribution.logpdf(parameters, samples)
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
        score = []
        for (
            parameters,
            samples,
            distribution,
        ) in self._zip_variational_parameters_distributions_samples(variational_parameters, x):
            score.append(distribution.grad_params_logpdf(parameters, samples))

        return np.row_stack(score)

    def fisher_information_matrix(self, variational_parameters):
        """Approximate the Fisher information matrix using Monte Carlo.

        Args:
            variational_parameters (np.ndarray): Variational parameters

        Returns:
            FIM (np.ndarray): Matrix (num parameters x num parameters)
        """
        fim = []
        for parameters, distribution in self._zip_variational_parameters_distributions(
            variational_parameters
        ):
            fim.append(distribution.fisher_information_matrix(parameters))

        return scipy.linalg.block_diag(*fim)

    def export_dict(self, variational_parameters):
        """Create a dict of the distribution based on the given parameters.

        Args:
            variational_parameters (np.ndarray): Variational parameters

        Returns:
            export_dict (dictionary): Dict containing distribution information
        """
        export_dict = {
            "type": "joint",
            "dimension": self.dimension,
            "variational_parameters": variational_parameters,
        }
        for i, (parameters, distribution) in enumerate(
            self._zip_variational_parameters_distributions(variational_parameters)
        ):
            component_dict = distribution.export_dict(parameters)
            component_key = f"subdistribution_{i}"
            export_dict.update({component_key: component_dict})
        return export_dict


def split_array_by_chunk_sizes(array, chunk_sizes):
    """Split up array by a list of chunk sizes.

    Args:
        array (np.ndarray): Array to be split
        chunk_sizes (np.ndarray): List of chunk sizes

    Returns:
        list:  with the chunks
    """
    if array.ndim > 2:
        raise ValueError(
            f"Can only split 1d or 2d arrays but you provided ab array of dim {array.ndim}"
        )

    total_dimension = np.atleast_2d(array).shape[1]
    if np.sum(chunk_sizes) != total_dimension:
        raise ValueError(
            f"The chunk sizes do not sum up ({np.sum(chunk_sizes)}) to the dimension of the"
            f" array {total_dimension}!"
        )

    chunked_array = np.split(array, np.cumsum(chunk_sizes)[:-1], axis=array.ndim - 1)
    return chunked_array
