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
"""Karhunen-Loève Random fields class."""

import logging

import numpy as np
from scipy.spatial.distance import pdist, squareform

from queens.distributions.mean_field_normal import MeanFieldNormal
from queens.parameters.random_fields.random_field import RandomField

_logger = logging.getLogger(__name__)


class KarhunenLoeve(RandomField):
    """Karhunen Loeve RandomField class.

    Attributes:
            nugget_variance (float): Nugget variance for the random field (lower bound for
                                        diagonal values of the covariance matrix).
            explained_variance (float): Explained variance by the eigen decomposition.
            std (float): Hyperparameter for standard-deviation of random field
            corr_length (float): Hyperparameter for the correlation length
            cut_off (float): Lower value limit of covariance matrix entries
            mean (np.array): Mean at coordinates of random field, can be a single constant
            cov_matrix (np.array): Covariance matrix to compute eigendecomposition on
            eigenbasis (np.array): Eigenvectors of covariance matrix, weighted by the eigenvalues
            eigenvalues (np.array): Eigenvalues of covariance matrix
            eigenvectors (np.array): Eigenvectors of covariance matrix
            dimension (int): Dimension of the latent space
    """

    def __init__(
        self,
        coords,
        mean=0.0,
        std=1.0,
        corr_length=0.3,
        explained_variance=None,
        latent_dimension=None,
        cut_off=0.0,
    ):
        """Initialize KL object.

        Args:
            coords (dict): Dictionary with coordinates of discretized random field and the
            mean (np.array): Mean at coordinates of random field, can be a single constant
            std (float): Hyperparameter for standard-deviation of random field
            corr_length (float): Hyperparameter for the correlation length
            explained_variance (float): Explained variance of by the eigen decomposition,
                                        mutually exclusive argument with latent_dimension
            latent_dimension (int): Dimension of the latent space,
                                    mutually exclusive argument with explained_variance
            cut_off (float): Lower value limit of covariance matrix entries
        """
        super().__init__(coords)
        self.nugget_variance = 1e-9
        self.explained_variance = explained_variance
        self.std = std
        self.corr_length = corr_length
        self.cut_off = cut_off
        self.mean = mean
        self.cov_matrix = None
        self.eigenbasis = None
        self.eigenvalues = None
        self.eigenvectors = None

        if (latent_dimension is None and explained_variance is None) or (
            latent_dimension is not None and explained_variance is not None
        ):
            raise KeyError("Specify either dimension or explained variance")

        if latent_dimension is not None:
            self.dimension = latent_dimension
        else:
            self.dimension = None

        self.calculate_covariance_matrix()
        self.eigendecomp_cov_matrix()

        self.distribution = MeanFieldNormal(mean=0, variance=1, dimension=self.dimension)

    def draw(self, num_samples):
        """Draw samples from the latent representation of the random field.

        Args:
            num_samples: Number of draws of latent random samples
        Returns:
            samples (np.ndarray): Drawn samples
        """
        return self.distribution.draw(num_samples)

    def logpdf(self, samples):
        """Get joint logpdf of latent space.

        Args:
            samples (np.array): Samples for evaluating the logpdf

        Returns:
            logpdf (np.array): Logpdf of the samples
        """
        return self.distribution.logpdf(samples)

    def grad_logpdf(self, samples):
        """Get gradient of joint logpdf of latent space.

        Args:
            samples (np.array): Samples for evaluating the gradient of the logpdf

        Returns:
            gradient (np.array): Gradient of the logpdf
        """
        return self.distribution.grad_logpdf(samples)

    def expanded_representation(self, samples):
        """Expand latent representation of sample.

        Args:
            samples (np.ndarray): Latent representation of sample

        Returns:
            samples_expanded (np.ndarray): Expanded representation of sample
        """
        samples_expanded = self.mean + np.matmul(samples, self.eigenbasis.T)
        return samples_expanded

    def latent_gradient(self, upstream_gradient):
        """Gradient of the field with respect to the latent parameters.

        Args:
            upstream_gradient (np.ndarray): Gradient with respect to all coords of the field

        Returns:
            latent_grad (np.ndarray): Graident of the field with respect to the latent
            parameters
        """
        latent_grad = np.matmul(upstream_gradient, self.eigenbasis)
        return latent_grad

    def calculate_covariance_matrix(self):
        """Calculate discretized covariance matrix.

        Based on the kernel description of the random field, build its
        covariance matrix using the external geometry and coordinates.
        """
        # assume squared exponential kernel
        distance = squareform(pdist(self.coords["coords"], "sqeuclidean"))
        covariance = (self.std**2) * np.exp(-distance / (2 * self.corr_length**2))
        covariance[covariance < self.cut_off] = 0
        self.cov_matrix = covariance + self.nugget_variance * np.eye(self.dim_coords)

    def eigendecomp_cov_matrix(self):
        """Decompose and then truncate the random field.

        According to desired variance fraction that should be
        covered/explained by the truncation.
        """
        # compute eigendecomposition
        eig_val, eig_vec = np.linalg.eigh(self.cov_matrix)
        eigenvalues = np.flip(eig_val)
        eigenvectors = np.flip(eig_vec, axis=1)

        if self.dimension is None:
            eigenvalues_normed = eigenvalues / np.sum(eigenvalues)
            dimension = (np.cumsum(eigenvalues_normed) < self.explained_variance).argmin() + 1
            if dimension == 1 and eigenvalues_normed[0] <= self.explained_variance:
                raise ValueError("Expansion failed.")

            self.dimension = dimension

        # truncated eigenfunction base
        self.eigenvalues = eigenvalues[: self.dimension]
        self.eigenvectors = eigenvectors[:, : self.dimension]

        if self.explained_variance is None:
            self.explained_variance = np.sum(self.eigenvalues) / np.sum(eigenvalues)
            _logger.info("Explained variance is %f", self.explained_variance)

        # weight the eigenbasis with the eigenvalues
        self.eigenbasis = self.eigenvectors * np.sqrt(self.eigenvalues)
