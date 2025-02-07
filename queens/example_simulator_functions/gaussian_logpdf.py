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
"""Gaussian distributions."""

import numpy as np

from queens.distributions.normal import NormalDistribution

# 1d standard Gaussian
STANDARD_NORMAL = NormalDistribution(0.0, 1)

# 2d Gaussian
DIM = 2

MEAN_2D = [0.0, 0.0]
COV_2D = [[1.0, 0.5], [0.5, 1.0]]

A = np.eye(DIM, DIM)
B = np.zeros(DIM)

GAUSSIAN_2D = NormalDistribution(MEAN_2D, COV_2D)

# 4d Gaussian
COV_4D = [
    [2.691259143915389, 1.465825570809310, 0.347698874175537, 0.140030644426489],
    [1.465825570809310, 4.161662217930926, 0.423882544003853, 1.357386322235196],
    [0.347698874175537, 0.423882544003853, 2.928845742295657, 0.484200164430076],
    [0.140030644426489, 1.357386322235196, 0.484200164430076, 3.350315448057768],
]

MEAN_4D = [0.806500709319150, 2.750827521892630, -3.388270291505472, 1.293259980552181]

GAUSSIAN_4D = NormalDistribution(MEAN_4D, COV_4D)


def gaussian_1d_logpdf(x):
    """1D Gaussian likelihood model.

    Used as a basic test function for MCMC methods.

    Returns:
        float: The logpdf evaluated at *x*
    """
    y = STANDARD_NORMAL.logpdf(x)
    return y


def gaussian_2d_logpdf(samples):
    """2D Gaussian logpdf.

    Args:
        samples (np.ndarray): Samples to be evaluated

    Returns:
        np.ndarray: logpdf
    """
    model_data = np.dot(A, samples.T).T + B
    y = GAUSSIAN_2D.logpdf(model_data)
    return y


def gaussian_4d_logpdf(samples):
    """4D Gaussian logpdf.

    Args:
        samples (np.ndarray): Samples to be evaluated

    Returns:
        np.ndarray: logpdf
    """
    y = GAUSSIAN_4D.logpdf(samples)
    return y
