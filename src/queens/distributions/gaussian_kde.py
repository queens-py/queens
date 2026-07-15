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
"""Gaussian kernel density estimate distribution."""

from __future__ import annotations

from typing import Any, override

import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import gaussian_kde
from sklearn.cluster import KMeans

from queens.distributions._distribution import Continuous
from queens.utils.logger_settings import log_init_args


class GaussianKDE(Continuous):
    """(Multivariate) Gaussian kernel density estimate distribution.

    This wraps ``scipy.stats.gaussian_kde`` to allow its use in QUEENS.
    """

    @log_init_args
    def __init__(
        self,
        samples: ArrayLike,
        weights: ArrayLike | None = None,
        bandwidth: Any | None = None,
    ) -> None:
        """Initialize a Gaussian kernel density estimate distribution.

        Args:
            samples: Samples used to estimate the density, with one sample per row.
            weights: Non-negative sample weights. They don't need to be normalized.
            bandwidth: KDE bandwidth method. See ``scipy.stats.gaussian_kde`` for available options.
        """
        samples = np.asarray(samples)
        if samples.ndim == 1:
            samples = samples.reshape(-1, 1)
        if samples.ndim != 2:
            raise ValueError("Samples must be a one- or two-dimensional array.")
        self._check_number_of_kernels(samples.shape[0], samples.shape[1])

        weights_array = None if weights is None else np.asarray(weights).reshape(-1)
        if weights_array is not None:
            if weights_array.size != samples.shape[0]:
                raise ValueError("Number of weights does not match the number of samples.")

        self.samples = samples
        self.bandwidth = bandwidth
        # create the scipy gaussian_kde object
        self.scipy_kde = gaussian_kde(
            samples.T,
            weights=weights_array,
            bw_method=bandwidth,
        )
        # get the normalized weights
        self.weights = self.scipy_kde.weights

        mean = np.average(samples, axis=0, weights=self.weights)
        sample_covariance = np.einsum("n,ni,nj->ij", self.weights, samples - mean, samples - mean)
        covariance = sample_covariance + self.scipy_kde.covariance
        super().__init__(mean=mean, covariance=covariance, dimension=samples.shape[1])

    @override
    def cdf(self, x: np.ndarray) -> np.ndarray:
        x = self._as_points(x)
        lower = np.full(self.dimension, -np.inf)
        return np.array(
            [self.scipy_kde.integrate_box(low_bounds=lower, high_bounds=point) for point in x]
        )

    @override
    def draw(self, num_draws: int = 1) -> np.ndarray:
        return self.scipy_kde.resample(size=num_draws).T

    @override
    def logpdf(self, x: np.ndarray) -> np.ndarray:
        x = self._as_points(x)
        return self.scipy_kde.logpdf(x.T)

    @override
    def pdf(self, x: np.ndarray) -> np.ndarray:
        x = self._as_points(x)
        return self.scipy_kde.pdf(x.T)

    @override
    def grad_logpdf(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError(
            "Gradient of logpdf not available for Gaussian KDE distributions."
        )

    @override
    def ppf(self, quantiles: np.ndarray) -> np.ndarray:
        raise NotImplementedError("ppf not available for Gaussian KDE distributions.")

    def downsample(
        self,
        max_kernels: int,
        random_state: int | None = None,
    ) -> GaussianKDE:
        """Downsample the KDE using weighted cluster centers.

        Args:
            max_kernels: Maximum number of kernels in the reduced KDE.
            random_state: Seed used for the k-means initialization.

        Returns:
            Weighted reduced Gaussian KDE, or this KDE if it already has at most ``max_kernels``
            kernels.
        """
        self._check_number_of_kernels(max_kernels, self.dimension)
        if self.samples.shape[0] <= max_kernels:
            return self

        kmeans = KMeans(n_clusters=max_kernels, random_state=random_state, n_init="auto")
        kmeans.fit(self.samples, sample_weight=self.weights)

        reduced_samples = []
        reduced_weights = []
        for cluster_label in range(max_kernels):
            samples_in_cluster_mask = kmeans.labels_ == cluster_label
            if not np.any(samples_in_cluster_mask):
                continue  # empty cluster

            cluster_weights = self.weights[samples_in_cluster_mask]
            # average over all samples in the cluster, weighted by their original weights
            # to determine the new reduced kernel location of this cluster
            reduced_samples.append(
                np.average(self.samples[samples_in_cluster_mask], axis=0, weights=cluster_weights)
            )
            reduced_weights.append(np.sum(cluster_weights))

        return GaussianKDE(
            np.array(reduced_samples),
            weights=np.array(reduced_weights),
            bandwidth=self.scipy_kde.factor,
        )

    @staticmethod
    def _check_number_of_kernels(number_of_kernels: int, dimension: int) -> None:
        if number_of_kernels <= dimension:
            raise ValueError("Number of kernels must be larger than the distribution dimension.")

    def _as_points(self, x: ArrayLike) -> np.ndarray:
        """Convert one point or a batch of points to row-wise shape."""
        x = np.asarray(x, dtype=float)
        if x.ndim == 0 and self.dimension == 1:
            return x.reshape(1, 1)
        if x.ndim == 1:
            if self.dimension == 1:
                return x.reshape(-1, 1)
            if x.size == self.dimension:
                return x.reshape(1, self.dimension)
        elif x.ndim == 2 and x.shape[1] == self.dimension:
            return x

        raise ValueError(
            "Evaluation points must have shape "
            f"({self.dimension},) or (n_points, {self.dimension})."
        )
