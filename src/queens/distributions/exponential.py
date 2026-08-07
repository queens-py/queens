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
"""Exponential distribution."""

from typing import override

import numpy as np

from queens.distributions._distribution import Continuous
from queens.utils.logger_settings import log_init_args


class Exponential(Continuous):
    r"""Exponential distribution class.

    For a multivariate distribution the components are assumed to be independent.

    Attributes:
        rate: Rate parameter(s) of the distribution.
        scale: Scale parameters(s) of the distribution
                            (:math:`scale = \frac{1}{rate}`) .
        pdf_const: Constant for the evaluation of the PDF.
        logpdf_const: Constant for the evaluation of the log-PDF.
    """

    @log_init_args
    def __init__(self, rate: float | np.ndarray) -> None:
        """Initialize exponential distribution.

        Args:
            rate: rate parameter(s) of the distribution
        """
        rate = np.array(rate).reshape(-1)
        super().check_positivity(rate=rate)
        scale = 1 / rate

        mean = scale
        covariance = np.diag(scale**2)
        dimension = mean.size

        pdf_const = np.prod(rate)
        logpdf_const = np.sum(np.log(rate))

        super().__init__(mean=mean, covariance=covariance, dimension=dimension)
        self.rate = rate
        self.scale = scale
        self.pdf_const = pdf_const
        self.logpdf_const = logpdf_const

    @override
    def cdf(self, x: np.ndarray) -> np.ndarray:
        x = x.reshape(-1, self.dimension)
        condition = (x >= 0).all(axis=1)
        cdf = np.where(condition, np.prod(1 - np.exp(-self.rate * x), axis=1), 0)
        return cdf

    @override
    def draw(self, num_draws: int = 1) -> np.ndarray:
        samples = np.random.exponential(scale=self.scale, size=(num_draws, self.dimension))
        return samples

    @override
    def logpdf(self, x: np.ndarray) -> np.ndarray:
        x = x.reshape(-1, self.dimension)
        condition = (x >= 0).all(axis=1)
        logpdf = self.logpdf_const + np.where(condition, np.sum(-self.rate * x, axis=1), -np.inf)
        return logpdf

    @override
    def grad_logpdf(self, x: np.ndarray) -> np.ndarray:
        x = x.reshape(-1, self.dimension)
        condition = (x >= 0).all(axis=1).reshape(-1, 1)
        grad_logpdf = np.where(condition, -self.rate, np.nan)
        return grad_logpdf

    @override
    def ppf(self, quantiles: np.ndarray) -> np.ndarray:
        self.check_1d()
        ppf = -self.scale * np.log(1 - quantiles)
        return ppf
