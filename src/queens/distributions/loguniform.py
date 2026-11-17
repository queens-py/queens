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
"""Log-uniform distribution."""

from typing import override

import numpy as np
from numpy.typing import ArrayLike

from queens.distributions._distribution import Continuous
from queens.utils.logger_settings import log_init_args


class LogUniform(Continuous):
    r"""Log-uniform (or reciprocal) distribution class.

    For :math:`0<a<b`, the one-dimensional density is defined as [1]

    .. math::

        f(x; a, b) = \frac{1}{x\ln(b/a)}, \qquad a \le x \le b.

    The density is zero outside these bounds.

    For multiple dimensions, the coordinates are considered independent, i.e. the joint density is
    the product of the one-dimensional densities.

    [1] https://en.wikipedia.org/wiki/Reciprocal_distribution

    Attributes:
        lower_bound: Lower bound(s) of the distribution.
        upper_bound: Upper bound(s) of the distribution.
        log_width: Width(s) in log-space.
        logpdf_const: Constant for the evaluation of the log-PDF.
    """

    @log_init_args
    def __init__(self, lower_bound: ArrayLike, upper_bound: ArrayLike) -> None:
        """Initialize log-uniform distribution.

        Args:
            lower_bound: Lower bound(s) of the distribution
            upper_bound: Upper bound(s) of the distribution
        """
        lower_bound = np.array(lower_bound, dtype=float).reshape(-1)
        upper_bound = np.array(upper_bound, dtype=float).reshape(-1)
        super().check_positivity(lower_bound=lower_bound, upper_bound=upper_bound)
        super().check_bounds(lower_bound, upper_bound)

        log_width = np.log(upper_bound / lower_bound)
        mean = (upper_bound - lower_bound) / log_width
        variance = (upper_bound**2 - lower_bound**2) / (2.0 * log_width) - mean**2

        super().__init__(mean=mean, covariance=np.diag(variance), dimension=mean.size)
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.log_width = log_width
        self.logpdf_const = np.sum(np.log(log_width))

    @override
    def cdf(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float).reshape(-1, self.dimension)
        x_clipped = np.clip(x, self.lower_bound, self.upper_bound)
        return np.prod(np.log(x_clipped / self.lower_bound) / self.log_width, axis=1)

    @override
    def draw(self, num_draws: int = 1) -> np.ndarray:
        return self.lower_bound * np.exp(
            np.random.uniform(size=(num_draws, self.dimension)) * self.log_width
        )

    @override
    def logpdf(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float).reshape(-1, self.dimension)
        within_bounds = (x >= self.lower_bound).all(axis=1) & (x <= self.upper_bound).all(axis=1)
        # outside the bounds, the pdf is zero, so logpdf is -inf.
        logpdf = np.full(x.shape[0], -np.inf)
        logpdf[within_bounds] = -np.sum(np.log(x[within_bounds]), axis=1) - self.logpdf_const
        return logpdf

    @override
    def grad_logpdf(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float).reshape(-1, self.dimension)
        within_bounds = (x >= self.lower_bound).all(axis=1) & (x <= self.upper_bound).all(axis=1)
        grad_logpdf = np.full(x.shape, np.nan)
        grad_logpdf[within_bounds] = -1.0 / x[within_bounds]
        return grad_logpdf

    @override
    def ppf(self, quantiles: np.ndarray) -> np.ndarray:
        self.check_1d()
        quantiles = np.asarray(quantiles, dtype=float).reshape(-1)
        positions = self.lower_bound * np.exp(quantiles * self.log_width)
        return np.where((quantiles >= 0) & (quantiles <= 1), positions, np.nan)
