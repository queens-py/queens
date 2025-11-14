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
"""Free Variable."""

from typing import Any

import numpy as np

from queens.distributions._distribution import Continuous
from queens.utils.logger_settings import log_init_args


class FreeVariable(Continuous):
    """Free variable class.

    This is not a proper distribution class. It is used for variables
    with no underlying distribution.
    """

    @log_init_args
    def __init__(self, dimension: int) -> None:
        """Initialize FreeVariable object.

        Args:
            dimension: Dimensionality of the variable
        """
        super().__init__(mean=np.array([]), covariance=np.array([]), dimension=dimension)

    def cdf(self, _: Any) -> np.ndarray:
        """Cumulative distribution function."""
        raise ValueError("cdf method is not supported for FreeVariable.")

    def draw(self, _: int = 1) -> np.ndarray:
        """Draw samples."""
        raise ValueError("draw method is not supported for FreeVariable.")

    def logpdf(self, _: Any) -> np.ndarray:
        """Log of the probability density function."""
        raise ValueError("logpdf method is not supported for FreeVariable.")

    def grad_logpdf(self, _: Any) -> np.ndarray:
        """Gradient of the log-PDF with respect to *x*."""
        raise ValueError("grad_logpdf method is not supported for FreeVariable.")

    def pdf(self, _: Any) -> np.ndarray:
        """Probability density function."""
        raise ValueError("pdf method is not supported for FreeVariable.")

    def ppf(self, _: Any) -> np.ndarray:
        """Percent point function (inverse of CDF — quantiles)."""
        raise ValueError("ppf method is not supported for FreeVariable.")
