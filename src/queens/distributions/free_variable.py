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

from queens.distributions._distribution import Continuous
from queens.utils.logger_settings import log_init_args


class FreeVariable(Continuous):
    """Free variable class.

    This is not a proper distribution class. It is used for variables
    with no underlying distribution.
    """

    @log_init_args
    def __init__(self, dimension):
        """Initialize FreeVariable object.

        Args:
            dimension (int): Dimensionality of the variable
        """
        super().__init__(mean=None, covariance=None, dimension=dimension)

    def cdf(self, _):
        """Cumulative distribution function."""
        raise ValueError("cdf method is not supported for FreeVariable.")

    def draw(self, _=1):
        """Draw samples."""
        raise ValueError("draw method is not supported for FreeVariable.")

    def logpdf(self, _):
        """Log of the probability density function."""
        raise ValueError("logpdf method is not supported for FreeVariable.")

    def grad_logpdf(self, _):
        """Gradient of the log pdf with respect to *x*."""
        raise ValueError("grad_logpdf method is not supported for FreeVariable.")

    def pdf(self, _):
        """Probability density function."""
        raise ValueError("pdf method is not supported for FreeVariable.")

    def ppf(self, _):
        """Percent point function (inverse of cdf — quantiles)."""
        raise ValueError("ppf method is not supported for FreeVariable.")
