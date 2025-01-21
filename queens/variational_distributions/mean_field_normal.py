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
"""Mean-Field Normal Variational Distribution."""

import numpy as np

from queens.utils.logger_settings import log_init_args
from queens.variational_distributions.variational_distribution import VariationalDistribution


class MeanFieldNormalVariational(VariationalDistribution):
    r"""Mean field multivariate normal distribution.

    Uses the parameterization (as in [1]):  :math:`parameters=[\mu, \lambda]`
    where :math:`\mu` are the mean values and :math:`\sigma^2=exp(2 \lambda)`
    the variances allowing for :math:`\lambda` to be unconstrained.

    References:
        [1]: Kucukelbir, Alp, et al. "Automatic differentiation variational inference."
             The Journal of Machine Learning Research 18.1 (2017): 430-474.

    Attributes:
        n_parameters (int): Number of parameters used in the parameterization.
    """

    @log_init_args
    def __init__(self, dimension):
        """Initialize variational distribution.

        Args:
            dimension (int): Dimension of RV.
        """
        super().__init__(dimension)
        self.n_parameters = 2 * dimension

    def initialize_variational_parameters(self, random=False):
        r"""Initialize variational parameters.

        Default initialization:
            :math:`\mu=0` and :math:`\sigma^2=1`

        Random intialization:
            :math:`\mu=Uniform(-0.1,0.1)` and :math:`\sigma^2=Uniform(0.9,1.1)`

        Args:
            random (bool, optional): If True, a random initialization is used. Otherwise the
                                     default is selected

        Returns:
            variational_parameters (np.ndarray):  variational parameters (1 x n_params)
        """
        if random:
            variational_parameters = np.hstack(
                (
                    0.1 * (-0.5 + np.random.rand(self.dimension)),
                    0.5 + np.log(1 + 0.1 * (-0.5 + np.random.rand(self.dimension))),
                )
            )
        else:
            variational_parameters = np.zeros(self.n_parameters)

        return variational_parameters

    @staticmethod
    def construct_variational_parameters(mean, covariance):
        """Construct the variational parameters from mean and covariance.

        Args:
            mean (np.ndarray): Mean values of the distribution (n_dim x 1)
            covariance (np.ndarray): Covariance matrix of the distribution (n_dim x n_dim)

        Returns:
            variational_parameters (np.ndarray): Variational parameters
        """
        if len(mean) == len(covariance):
            variational_parameters = np.hstack((mean.flatten(), 0.5 * np.log(np.diag(covariance))))
        else:
            raise ValueError(
                f"Dimension of the mean value {len(mean)} does not equal covariance dimension"
                f"{covariance.shape}"
            )
        return variational_parameters

    def reconstruct_distribution_parameters(self, variational_parameters):
        """Reconstruct mean and covariance from the variational parameters.

        Args:
            variational_parameters (np.ndarray): Variational parameters

        Returns:
            mean (np.ndarray): Mean value of the distribution (n_dim x 1)
            cov (np.ndarray): Covariance matrix of the distribution (n_dim x n_dim)
        """
        mean, cov = (
            variational_parameters[: self.dimension],
            np.exp(2 * variational_parameters[self.dimension :]),
        )
        return mean.reshape(-1, 1), np.diag(cov)

    def _grad_reconstruct_distribution_parameters(self, variational_parameters):
        """Gradient of the parameter reconstruction.

        Args:
            variational_parameters (np.ndarray): Variational parameters

        Returns:
            grad_reconstruct_params (np.ndarray): Gradient vector of the reconstruction
                                                w.r.t. the variational parameters
        """
        grad_mean = np.ones((1, self.dimension))
        grad_std = (np.exp(variational_parameters[self.dimension :])).reshape(1, -1)
        grad_reconstruct_params = np.hstack((grad_mean, grad_std))
        return grad_reconstruct_params

    def draw(self, variational_parameters, n_draws=1):
        """Draw *n_draw* samples from the variational distribution.

        Args:
            variational_parameters (np.ndarray): Variational parameters
            n_draws (int): Number of samples to draw

        Returns:
            samples (np.ndarray): Row-wise samples of the variational distribution
        """
        mean, cov = self.reconstruct_distribution_parameters(variational_parameters)
        samples = np.random.randn(n_draws, self.dimension) * np.sqrt(np.diag(cov)).reshape(
            1, -1
        ) + mean.reshape(1, -1)
        return samples

    def logpdf(self, variational_parameters, x):
        """Logpdf evaluated using the variational parameters at samples `x`.

        Args:
            variational_parameters (np.ndarray): Variational parameters
            x (np.ndarray): Row-wise samples

        Returns:
            logpdf (np.ndarray): Row vector of the logpdfs
        """
        mean, cov = self.reconstruct_distribution_parameters(variational_parameters)
        mean = mean.flatten()
        cov = np.diag(cov)
        x = np.atleast_2d(x)
        logpdf = (
            -0.5 * self.dimension * np.log(2 * np.pi)
            - np.sum(variational_parameters[self.dimension :])
            - 0.5 * np.sum((x - mean) ** 2 / cov, axis=1)
        )
        return logpdf.flatten()

    def pdf(self, variational_parameters, x):
        """Pdf of the variational distribution evaluated at samples *x*.

        First computes the logpdf, which is numerically more stable for exponential distributions.

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

        Args:
            variational_parameters (np.ndarray): Variational parameters
            x (np.ndarray): Row-wise samples

        Returns:
            score (np.ndarray): Column-wise scores
        """
        mean, cov = self.reconstruct_distribution_parameters(variational_parameters)
        mean = mean.flatten()
        cov = np.diag(cov)
        dlogpdf_dmu = (x - mean) / cov
        dlogpdf_dsigma = (x - mean) ** 2 / cov - np.ones(x.shape)
        score = np.concatenate(
            [
                dlogpdf_dmu.T.reshape(self.dimension, len(x)),
                dlogpdf_dsigma.T.reshape(self.dimension, len(x)),
            ]
        )
        return score

    def total_grad_params_logpdf(self, variational_parameters, standard_normal_sample_batch):
        """Total logpdf reparameterization gradient.

        Total logpdf reparameterization gradient w.r.t. the variational parameters.

        Args:
            variational_parameters (np.ndarray): Variational parameters
            standard_normal_sample_batch (np.ndarray): Standard normal distributed sample batch

        Returns:
            total_grad (np.ndarray): Total Logpdf reparameterization gradient
        """
        total_grad = np.zeros((standard_normal_sample_batch.shape[0], variational_parameters.size))
        total_grad[:, self.dimension :] = -1.0
        return total_grad

    def grad_sample_logpdf(self, variational_parameters, sample_batch):
        """Computes the gradient of the logpdf w.r.t. to the *x*.

        Args:
            variational_parameters (np.ndarray): Variational parameters
            sample_batch (np.ndarray): Row-wise samples

        Returns:
            gradients_batch (np.ndarray): Gradients of the log-pdf w.r.t. the
            sample *x*. The first dimension of the array corresponds to
            the different samples. The second dimension to different dimensions
            within one sample. (Third dimension is empty and just added to
            keep slices two dimensional.)
        """
        mean, cov = self.reconstruct_distribution_parameters(variational_parameters)
        gradients_batch = -(sample_batch - mean.reshape(1, self.dimension)) / np.diag(cov).reshape(
            1, self.dimension
        )
        return gradients_batch

    def fisher_information_matrix(self, variational_parameters):
        r"""Compute the Fisher information matrix analytically.

        Args:
            variational_parameters (np.ndarray): Variational parameters

        Returns:
            FIM (np.ndarray): Matrix (n_parameters x n_parameters)
        """
        fisher_diag = np.exp(-2 * variational_parameters[self.dimension :])
        fisher_diag = np.hstack((fisher_diag, 2 * np.ones(self.dimension)))
        return np.diag(fisher_diag)

    def export_dict(self, variational_parameters):
        """Create a dict of the distribution based on the given parameters.

        Args:
            variational_parameters (np.ndarray): Variational parameters

        Returns:
            export_dict (dictionary): Dict containing distribution information
        """
        mean, cov = self.reconstruct_distribution_parameters(variational_parameters)
        sd = cov**0.5
        export_dict = {
            "type": "meanfield_Normal",
            "mean": mean,
            "covariance": cov,
            "standard_deviation": sd,
            "variational_parameters": variational_parameters,
        }
        return export_dict

    def conduct_reparameterization(self, variational_parameters, n_samples):
        """Conduct a reparameterization.

        Args:
            variational_parameters (np.ndarray): Array with variational parameters
            n_samples (int): Number of samples for current batch

        Returns:
            * samples_mat (np.ndarray): Array of actual samples from the
              variational distribution
            * standard_normal_sample_batch (np.ndarray): Standard normal
              distributed sample batch
        """
        standard_normal_sample_batch = np.random.normal(0, 1, size=(n_samples, self.dimension))
        mean, cov = self.reconstruct_distribution_parameters(variational_parameters)
        samples_mat = mean.flatten() + np.sqrt(np.diag(cov)) * standard_normal_sample_batch

        return samples_mat, standard_normal_sample_batch

    def grad_params_reparameterization(
        self, variational_parameters, standard_normal_sample_batch, upstream_gradient
    ):
        r"""Calculate the gradient of the reparameterization.

        Args:
            variational_parameters (np.ndarray): Variational parameters
            standard_normal_sample_batch (np.ndarray): Standard normal distributed sample batch
            upstream_gradient (np.array): Upstream gradient

        Returns:
            gradient (np.ndarray): Gradient of the upstream function w.r.t. the variational
                                   parameters.

        Note:
            We assume that *grad_reconstruct_params* is a row-vector containing the partial
            derivatives of the reconstruction mapping of the actual distribution parameters
            w.r.t. the variational parameters.

            The variable *jacobi_parameters* is the (n_parameters :math:`\times` dim_sample)
            Jacobi matrix of the reparameterization w.r.t. the distribution parameters,
            with differentiating after the distribution
            parameters in different rows and different output dimensions of the sample per
            column.
        """
        grad_reconstruct_params = self._grad_reconstruct_distribution_parameters(
            variational_parameters
        )
        gradient = (
            np.hstack((upstream_gradient, upstream_gradient * standard_normal_sample_batch))
            * grad_reconstruct_params
        )
        return gradient
