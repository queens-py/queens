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
"""Joint Variational Distribution."""

import numpy as np
import scipy

from queens.utils.type_hinting import ArrayN, ArrayNxM
from queens.variational_distributions._variational_distribution import Variational


class Joint(Variational):
    r"""Joint variational distribution class.

    This distribution allows to join distributions in an independent fashion:
    :math:`q(\theta|\lambda)=\prod_{i=1}^{N}q_i(\theta_i | \lambda_i)`

    NOTE: :math:`q_i(\theta_i | \lambda_i)` can be multivariate or of different families. Hence it
    is a generalization of the mean field distribution

    Attributes:
        distributions: Variational distribution objects for the different independent distributions.
        n_parameters: Total number of parameters used in the parameterization.
        distributions_n_parameters: Number of parameters per distribution
        distributions_dimension: Number of dimension per distribution
    """

    def __init__(self, distributions: list, dimension: int) -> None:
        """Initialize joint distribution.

        Args:
            distributions: Variational distribution objects for the different independent
                distributions.
            dimension: Dimension of the random variable
        """
        self.distributions = distributions
        self.distributions_n_parameters = np.array(
            [distribution.n_parameters for distribution in self.distributions]
        ).astype(int)

        super().__init__(dimension, n_parameters=int(np.sum(self.distributions_n_parameters)))

        self.distributions_dimension = np.array(
            [distribution.dimension for distribution in self.distributions]
        ).astype(int)

        if dimension != np.sum(self.distributions_dimension):
            raise ValueError(
                f"The provided total dimension {dimension} of the distribution does not match the "
                f"dimensions of the subdistributions {np.sum(self.distributions_dimension)}"
            )

    def initialize_variational_parameters(self, random: bool = False) -> ArrayN:
        r"""Initialize variational parameters.

        The distribution initialization is handle by the component itself.

        Args:
            random: If True, a random initialization is used. Otherwise the default is selected

        Returns:
            Variational parameters of shape (n_params,)
        """
        variational_parameters = np.concatenate(
            [
                distribution.initialize_variational_parameters(random)
                for distribution in self.distributions
            ]
        )

        return variational_parameters

    def construct_variational_parameters(  # pylint: disable=arguments-differ
        self, distributions_parameters: list
    ) -> ArrayN:
        """Construct the variational parameters from the distribution list.

        Args:
            distributions_parameters: Parameters of the distributions

        Returns:
            Variational parameters
        """
        variational_parameters = []
        for parameters, distribution in zip(
            distributions_parameters, self.distributions, strict=True
        ):
            variational_parameters.append(
                distribution.construct_variational_parameters(*parameters)
            )
        return np.concatenate(variational_parameters)

    def _construct_distributions_variational_parameters(
        self, variational_parameters: ArrayN
    ) -> list:
        """Reconstruct the parameters of the distributions.

        Creates a list containing the variational parameters of the different components.

        The list is nested, each entry correspond to the parameters of a distribution.

        Args:
            variational_parameters: Variational parameters

        Returns:
            Variational parameters of the components
        """
        variational_parameters_list = split_array_by_chunk_sizes(
            variational_parameters, self.distributions_n_parameters
        )
        return variational_parameters_list

    def reconstruct_distribution_parameters(self, variational_parameters: ArrayN) -> list[list]:
        """Reconstruct the parameters of distributions.

        The list is nested, each entry correspond to the parameters of a distribution.

        Args:
            variational_parameters: Variational parameters

        Returns:
            Distribution parameters of the components
        """
        distribution_parameters_list = []

        for parameters, distribution in self._zip_variational_parameters_distributions(
            variational_parameters
        ):
            distribution_parameters_list.append(
                distribution.reconstruct_distribution_parameters(parameters)
            )
        return [distribution_parameters_list]

    def _zip_variational_parameters_distributions(self, variational_parameters: ArrayN) -> zip:
        """Zip parameters and distributions.

        This helper function creates a generator for variational parameters and subdistribution.

        Args:
            variational_parameters: Variational parameters

        Returns:
            Zip of variational parameters and distributions
        """
        return zip(
            split_array_by_chunk_sizes(variational_parameters, self.distributions_n_parameters),
            self.distributions,
            strict=True,
        )

    def _zip_variational_parameters_distributions_samples(
        self, variational_parameters: ArrayN, samples: np.ndarray
    ) -> zip:
        """Zip parameters, samples and distributions.

        This helper function creates a generator for variational parameters, samples and
        subdistribution.

        Args:
            variational_parameters: Variational parameters
            samples: Row-wise samples

        Returns:
            Zip of variational parameters, samples and distributions
        """
        return zip(
            split_array_by_chunk_sizes(variational_parameters, self.distributions_n_parameters),
            split_array_by_chunk_sizes(samples, self.distributions_dimension),
            self.distributions,
            strict=True,
        )

    def draw(self, variational_parameters: ArrayN, n_draws: int = 1) -> ArrayNxM:
        """Draw *n_draw* samples from the variational distribution.

        Args:
            variational_parameters: Variational parameters
            n_draws: Number of samples to draw

        Returns:
            Samples of shape (n_draws, n_dim)
        """
        sample_array = []
        for parameters, distribution in self._zip_variational_parameters_distributions(
            variational_parameters
        ):
            sample_array.append(distribution.draw(parameters, n_draws))
        return np.column_stack(sample_array)

    def logpdf(self, variational_parameters: ArrayN, x: np.ndarray) -> np.ndarray:
        """Log-PDF evaluated using the variational parameters at samples *x*.

        Args:
            variational_parameters: Variational parameters
            x: Row-wise samples

        Returns:
            Row vector of the Log-PDF values
        """
        logpdf = np.zeros_like(x[:, 0], dtype=float)
        for (
            parameters,
            samples,
            distribution,
        ) in self._zip_variational_parameters_distributions_samples(variational_parameters, x):
            logpdf += distribution.logpdf(parameters, samples)
        return logpdf

    def pdf(self, variational_parameters: ArrayN, x: np.ndarray) -> np.ndarray:
        """Pdf evaluated using the variational parameters at given samples `x`.

        Args:
            variational_parameters: Variational parameters
            x: Row-wise samples

        Returns:
            Row vector of the PDF values
        """
        pdf = np.exp(self.logpdf(variational_parameters, x))
        return pdf

    def grad_params_logpdf(self, variational_parameters: ArrayN, x: np.ndarray) -> np.ndarray:
        """Log-PDF gradient w.r.t. the variational parameters.

        Evaluated at samples *x*. Also known as the score function.
        Is a general implementation using the score functions of
        the components.

        Args:
            variational_parameters: Variational parameters
            x: Row-wise samples

        Returns:
            Column-wise scores
        """
        score = []
        for (
            parameters,
            samples,
            distribution,
        ) in self._zip_variational_parameters_distributions_samples(variational_parameters, x):
            score.append(distribution.grad_params_logpdf(parameters, samples))

        return np.row_stack(score)

    def fisher_information_matrix(self, variational_parameters: ArrayN) -> np.ndarray:
        """Approximate the Fisher information matrix using Monte Carlo.

        Args:
            variational_parameters: Variational parameters

        Returns:
            Fisher information matrix (num parameters x num parameters)
        """
        fim = []
        for parameters, distribution in self._zip_variational_parameters_distributions(
            variational_parameters
        ):
            fim.append(distribution.fisher_information_matrix(parameters))

        return scipy.linalg.block_diag(*fim)

    def export_dict(self, variational_parameters: ArrayN) -> dict:
        """Create a dict of the distribution based on the given parameters.

        Args:
            variational_parameters: Variational parameters

        Returns:
            Dictionary containing distribution information
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


def split_array_by_chunk_sizes(array: np.ndarray, chunk_sizes: np.ndarray) -> list:
    """Split up array by a list of chunk sizes.

    Args:
        array: Array to be split
        chunk_sizes: Chunk sizes

    Returns:
        Chunks of the array
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
