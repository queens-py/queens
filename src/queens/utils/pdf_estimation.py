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
"""Kernel density estimation (KDE).

Estimation of the probability density function based on samples from the
distribution.
"""

import logging

import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity

_logger = logging.getLogger(__name__)


def estimate_bandwidth_for_kde(samples, min_samples, max_samples, kernel="gaussian"):
    """Estimate optimal bandwidth for kde of pdf.

    Args:
        samples (np.ndarray):  Samples for which to estimate pdf
        min_samples (float): Smallest value
        max_samples (float): Largest value
        kernel (str,optional):        Kernel type

    Returns:
        float: Estimate for optimal *kernel_bandwidth*
    """
    kernel_bandwidth_upper_bound = np.log10((max_samples - min_samples) / 2.0)
    kernel_bandwidth_lower_bound = np.log10((max_samples - min_samples) / 30.0)

    # do 30-fold cross-validation and use all cores available to speed-up the process
    # we use a log grid to emphasize the smaller bandwidth values
    grid = GridSearchCV(
        KernelDensity(kernel=kernel),
        {"bandwidth": np.logspace(kernel_bandwidth_lower_bound, kernel_bandwidth_upper_bound, 40)},
        cv=30,
        n_jobs=-1,
    )

    grid.fit(samples.reshape(-1, 1))
    kernel_bandwidth = grid.best_params_["bandwidth"]
    _logger.info("bandwidth = %s", kernel_bandwidth)

    return kernel_bandwidth


def estimate_pdf(samples, kernel_bandwidth, support_points=None, kernel="gaussian"):
    """Estimate pdf using kernel density estimation.

    Args:
        samples (np.array):         Samples for which to estimate pdf
        kernel_bandwidth (float):   Kernel width to use in kde
        support_points (np.array):  Points where to evaluate pdf
        kernel (str, optional):               Kernel type

    Returns:
        np.ndarray, np.ndarray: *pdf_estimate* at support points
    """
    # make sure that we have at least 2 D column vectors but do not change correct 2D format
    samples = np.atleast_2d(samples).T

    # support points given
    if support_points is None:
        min_samples = np.amin(samples)
        max_samples = np.amax(samples)
        support_points = np.linspace(min_samples, max_samples, 100)
        support_points = np.meshgrid(*[support_points[:, None]] * samples.shape[1])
        points = support_points[0].reshape(-1, 1)
        if len(points.shape) > 1:
            for col in range(1, samples.shape[1]):
                points = np.hstack(
                    (points, support_points[col].reshape(-1, 1))
                )  # reshape matrix to vector with all combinations
        support_points = np.atleast_2d(points)
    else:
        support_points = np.atleast_2d(support_points).T

        # no support points given
    kde = KernelDensity(kernel=kernel, bandwidth=kernel_bandwidth).fit(samples)

    y_density = np.exp(kde.score_samples(support_points))
    return y_density, support_points
