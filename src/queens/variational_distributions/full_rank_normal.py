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
"""Full-Rank Normal Variational Distribution."""

import numpy as np
import scipy
from numba import njit

from queens.utils.logger_settings import log_init_args
from queens.utils.type_hinting import ArrayN, ArrayNx1, ArrayNxM
from queens.variational_distributions._variational_distribution import Variational


class FullRankNormal(Variational):
    r"""Full-rank multivariate normal distribution.

    Uses the parameterization (as in [1])
    :math:`parameters=[\mu, \lambda]`, where :math:`\mu` are the mean values and
    :math:`\lambda` is an array containing the nonzero entries of the lower Cholesky
    decomposition of the covariance matrix :math:`L`:
    :math:`\lambda=[L_{00},L_{10},L_{11},L_{20},L_{21},L_{22}, ...]`.
    This allows the parameters :math:`\lambda` to be unconstrained.

    References:
        [1]: Kucukelbir, Alp, et al. "Automatic differentiation variational inference."
             The Journal of Machine Learning Research 18.1 (2017): 430-474.

    Attributes:
        n_parameters: Number of parameters used in the parameterization.
    """

    @log_init_args
    def __init__(self, dimension: int) -> None:
        """Initialize variational distribution.

        Args:
            dimension: Dimension of the RV
        """
        n_parameters = (dimension * (dimension + 1)) // 2 + dimension
        super().__init__(dimension, n_parameters)

    def initialize_variational_parameters(self, random: bool = False) -> ArrayN:
        r"""Initialize variational parameters.

        Default initialization:
            :math:`\mu=0` and :math:`L=diag(1)` where :math:`\Sigma=LL^T`

        Random intialization:
            :math:`\mu=Uniform(-0.1,0.1)` :math:`L=diag(Uniform(0.9,1.1))` where :math:`\Sigma=LL^T`

        Args:
            random: If True, a random initialization is used. Otherwise the default is selected.

        Returns:
            Variational parameters of shape (n_params,)
        """
        if random:
            cholesky_covariance = np.eye(self.dimension) + 0.1 * (
                -0.5 + np.diag(np.random.rand(self.dimension))
            )
            variational_parameters = np.zeros(self.dimension) + 0.1 * (
                -0.5 + np.random.rand(self.dimension)
            )
            for j in range(len(cholesky_covariance)):
                variational_parameters = np.hstack(
                    (variational_parameters, cholesky_covariance[j, : j + 1])
                )
        else:
            mean = np.zeros(self.dimension)
            cholesky = np.ones((self.dimension * (self.dimension + 1)) // 2)
            variational_parameters = np.concatenate([mean, cholesky])

        return variational_parameters

    def construct_variational_parameters(  # pylint: disable=arguments-differ
        self, mean: ArrayNx1 | ArrayN, covariance: ArrayNxM
    ) -> ArrayN:
        """Construct the variational parameters from mean and covariance.

        Args:
            mean: Mean values of the distribution of shape (n_dim, 1) or (n_dim,)
            covariance: Covariance matrix of the distribution of shape (n_dim, n_dim)

        Returns:
            Variational parameters
        """
        if len(mean) == len(covariance):
            cholesky_covariance = np.linalg.cholesky(covariance)
            variational_parameters = mean.flatten()
            for j in range(len(cholesky_covariance)):
                variational_parameters = np.hstack(
                    (variational_parameters, cholesky_covariance[j, : j + 1])
                )
        else:
            raise ValueError(
                f"Dimension of the mean value {len(mean)} does not equal covariance dimension"
                f"{covariance.shape}"
            )
        return variational_parameters

    def reconstruct_distribution_parameters(
        self, variational_parameters: ArrayN
    ) -> tuple[ArrayNx1, ArrayNxM]:
        """Reconstruct mean value, covariance and its Cholesky decomposition.

        Args:
            variational_parameters: Variational parameters
        Returns:
            Mean value of the distribution of shape (n_dim, 1)
            Covariance of the distribution of shape (n_dim, n_dim)
        """
        mean, cov, _ = self.reconstruct_distribution_parameters_with_cholesky(
            variational_parameters
        )
        return mean, cov

    def reconstruct_distribution_parameters_with_cholesky(
        self, variational_parameters: ArrayN
    ) -> tuple[ArrayNx1, ArrayNxM, ArrayNxM]:
        """Reconstruct mean value, covariance and its Cholesky decomposition.

        Args:
            variational_parameters: Variational parameters
        Returns:
            Mean value of the distribution of shape (n_dim, 1)
            Covariance of the distribution of shape (n_dim, n_dim)
            Cholesky decomposition of the covariance matrix of shape (n_dim, n_dim)
        """
        mean = variational_parameters[: self.dimension].reshape(-1, 1)
        cholesky_covariance_array = variational_parameters[self.dimension :]
        cholesky_covariance = np.zeros((self.dimension, self.dimension))
        idx = np.tril_indices(self.dimension, k=0, m=self.dimension)
        cholesky_covariance[idx] = cholesky_covariance_array
        cov = np.matmul(cholesky_covariance, cholesky_covariance.T)

        return mean, cov, cholesky_covariance

    def _grad_reconstruct_distribution_parameters(self) -> np.ndarray:
        """Gradient of the parameter reconstruction.

        Returns:
            Gradient vector of the reconstruction w.r.t. the variational parameters
        """
        grad_reconstruct_params = np.ones((1, self.n_parameters))
        return grad_reconstruct_params

    def draw(self, variational_parameters: ArrayN, n_draws: int = 1) -> ArrayNxM:
        """Draw *n_draw* samples from the variational distribution.

        Args:
            variational_parameters: Variational parameters
            n_draws: Number of samples to draw

        Returns:
            Samples of shape (n_draws, n_dim)
        """
        mean, _, cholesky = self.reconstruct_distribution_parameters_with_cholesky(
            variational_parameters
        )
        sample = np.dot(cholesky, np.random.randn(self.dimension, n_draws)).T + mean.reshape(1, -1)
        return sample

    def logpdf(self, variational_parameters: ArrayN, x: np.ndarray) -> np.ndarray:
        """Log-PDF evaluated at the samples *x*.

        Args:
            variational_parameters: Variational parameters
            x: Row-wise samples

        Returns:
            Row vector of the Log-PDF values
        """
        mean, cov, cholesky = self.reconstruct_distribution_parameters_with_cholesky(
            variational_parameters
        )
        x = np.atleast_2d(x)
        u = np.linalg.solve(cov, (x.T - mean))

        def col_dot_prod(x: np.ndarray, y: np.ndarray) -> np.ndarray:
            return np.sum(x * y, axis=0)

        logpdf = (
            -0.5 * self.dimension * np.log(2 * np.pi)
            - np.sum(np.log(np.abs(np.diag(cholesky))))
            - 0.5 * col_dot_prod(x.T - mean, u)
        )
        return logpdf.flatten()

    def pdf(self, variational_parameters: ArrayN, x: np.ndarray) -> np.ndarray:
        """PDF of evaluated at given samples *x*.

        First computes the log-PDF, which is numerically more stable for exponential distributions.

        Args:
            variational_parameters: Variational parameters
            x: Row-wise samples

        Returns:
            Row vector of the PDF values
        """
        pdf = np.exp(self.logpdf(variational_parameters, x))
        return pdf

    def grad_params_logpdf(self, variational_parameters: ArrayN, x: np.ndarray) -> np.ndarray:
        """Log-PDF gradient w.r.t. to the variational parameters.

        Evaluated at samples *x*. Also known as the score function.

        Args:
            variational_parameters: Variational parameters
            x: Row-wise samples

        Returns:
            Column-wise scores
        """
        mean, cov, cholesky = self.reconstruct_distribution_parameters_with_cholesky(
            variational_parameters
        )
        x = np.atleast_2d(x)
        # Helper variable
        q = np.linalg.solve(cov, x.T - mean)
        dlogpdf_dmu = q.copy()
        diag_indx = np.cumsum(np.arange(1, self.dimension + 1)) - 1
        n_params_chol = (self.dimension * (self.dimension + 1)) // 2
        dlogpdf_dsigma = np.zeros((n_params_chol, 1))
        # Term due to determinant
        dlogpdf_dsigma[diag_indx] = -1.0 / (np.diag(cholesky).reshape(-1, 1))
        dlogpdf_dsigma = np.tile(dlogpdf_dsigma, (1, len(x)))
        # Term due to normalization
        b = np.matmul(cholesky.T, q)
        indx = 0
        f = np.zeros(dlogpdf_dsigma.shape)
        for r in range(0, self.dimension):
            for s in range(0, r + 1):
                dlogpdf_dsigma[indx, :] += q[r, :] * b[s, :]
                f[indx, :] += q[r, :] * b[s, :]
                indx += 1
        score = np.vstack((dlogpdf_dmu, dlogpdf_dsigma))
        return score

    def total_grad_params_logpdf(
        self, variational_parameters: ArrayN, standard_normal_sample_batch: np.ndarray
    ) -> np.ndarray:
        """Total log-PDF reparameterization gradient.

        Total log-PDF reparameterization gradient w.r.t. the variational parameters.

        Args:
            variational_parameters: Variational parameters
            standard_normal_sample_batch: Standard normal distributed sample batch

        Returns:
            Total log-PDF reparameterization gradient
        """
        idx = np.tril_indices(self.dimension, k=0, m=self.dimension)
        cholesky_diagonal_idx = np.where(np.equal(*idx))[0] + self.dimension
        total_grad = np.zeros((standard_normal_sample_batch.shape[0], variational_parameters.size))
        total_grad[:, cholesky_diagonal_idx] = -1 / variational_parameters[cholesky_diagonal_idx]
        return total_grad

    def grad_sample_logpdf(
        self, variational_parameters: ArrayN, sample_batch: np.ndarray
    ) -> np.ndarray:
        """Computes the gradient of the log-PDF w.r.t. to the sample *x*.

        Args:
            variational_parameters: Variational parameters
            sample_batch: Row-wise samples

        Returns:
            Gradients of the log-pdf w.r.t. the sample *x*. The first dimension of the array
                corresponds to the different samples. The second dimension to different dimensions
                within one sample. (Third dimension is empty and just added to keep slices
                two-dimensional.)
        """
        mean, cov = self.reconstruct_distribution_parameters(variational_parameters)
        gradient_lst = []
        for sample in sample_batch:
            gradient_lst.append(
                np.dot(np.linalg.inv(cov), -(sample.reshape(-1, 1) - mean)).reshape(-1, 1)
            )

        gradients_batch = np.array(gradient_lst)
        return gradients_batch.reshape(sample_batch.shape)

    def fisher_information_matrix(self, variational_parameters: ArrayN) -> np.ndarray:
        """Compute the Fisher information matrix analytically.

        Args:
            variational_parameters: Variational parameters

        Returns:
            Fisher information matrix (num parameters x num parameters)
        """
        _, cov, cholesky = self.reconstruct_distribution_parameters_with_cholesky(
            variational_parameters
        )

        def fim_blocks(dimension: int) -> tuple[np.ndarray, np.ndarray]:
            """Compute the blocks of the FIM."""
            mu_block = np.linalg.inv(cov + 1e-8 * np.eye(len(cov)))
            n_params_chol = (dimension * (dimension + 1)) // 2
            sigma_block = np.zeros((n_params_chol, n_params_chol))
            matrix_list = []
            # Improvements of this implementation are welcomed!
            for r in range(0, dimension):
                for s in range(0, r + 1):
                    matrix_q = np.zeros(cholesky.shape)
                    matrix_q[r, s] = 1
                    matrix_q = cholesky @ matrix_q.T
                    matrix_q = np.linalg.solve(cov, matrix_q + matrix_q.T)
                    matrix_list.append(matrix_q)
            for p in range(n_params_chol):
                for q in range(p + 1):
                    val = 0.5 * np.trace(matrix_list[p] @ matrix_list[q])
                    sigma_block[p, q] = val
                    sigma_block[q, p] = val
            return mu_block, sigma_block

        # Using jit is useful in higher dimensional cases but introduces an computational overhead
        # for lowerdimensional cases. Doing some tests showed that the break evenpoint is reached
        # at around dimension 35
        if self.dimension < 35:
            mu_block, sigma_block = fim_blocks(self.dimension)
        else:
            mu_block, sigma_block = njit(fim_blocks)(self.dimension)

        return scipy.linalg.block_diag(mu_block, sigma_block)

    def export_dict(self, variational_parameters: ArrayN) -> dict:
        """Create a dict of the distribution based on the given parameters.

        Args:
            variational_parameters: Variational parameters
        Returns:
            Dictionary containing distribution information
        """
        mean, cov = self.reconstruct_distribution_parameters(variational_parameters)
        export_dict = {
            "type": "fullrank_Normal",
            "mean": mean,
            "covariance": cov,
            "variational_parameters": variational_parameters,
        }
        return export_dict

    def conduct_reparameterization(
        self, variational_parameters: np.ndarray, n_samples: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Conduct a reparameterization.

        Args:
            variational_parameters: Array with variational parameters
            n_samples: Number of samples for current batch

        Returns:
            Actual samples from the variational distribution
            Standard normal distributed samples used for the reparameterization
        """
        standard_normal_sample_batch = np.random.normal(0, 1, size=(n_samples, self.dimension))
        mean, _, cholesky = self.reconstruct_distribution_parameters_with_cholesky(
            variational_parameters
        )
        samples_mat = mean + np.dot(cholesky, standard_normal_sample_batch.T)

        return samples_mat.T, standard_normal_sample_batch

    def grad_params_reparameterization(
        self,
        variational_parameters: ArrayN,
        standard_normal_sample_batch: np.ndarray,
        upstream_gradient: np.ndarray,
    ) -> np.ndarray:
        r"""Calculate the gradient of the reparameterization.

        Args:
            variational_parameters: Variational parameters
            standard_normal_sample_batch: Standard normal distributed sample batch
            upstream_gradient: Upstream gradient

        Returns:
            Gradient of the upstream function w.r.t. the variational parameters.

        **Note:**
            We assume that *grad_reconstruct_params* is a row-vector containing the partial
            derivatives of the reconstruction mapping of the actual distribution parameters
            w.r.t. the variational parameters.

            The variable *jacobi_parameters* is the (n_parameters :math:`\times` dim_sample)
            Jacobi matrix of the reparameterization w.r.t. the distribution parameters,
            with differentiating after the distribution
            parameters in different rows and different output dimensions of the sample per
            column.
        """
        # pylint: disable=unused-argument
        grad_reconstruct_params = self._grad_reconstruct_distribution_parameters()

        indices_1, indices_2 = np.tril_indices(self.dimension)

        gradient = (
            np.hstack(
                (
                    upstream_gradient,
                    upstream_gradient[:, indices_1] * standard_normal_sample_batch[:, indices_2],
                )
            )
            * grad_reconstruct_params
        )

        return gradient
