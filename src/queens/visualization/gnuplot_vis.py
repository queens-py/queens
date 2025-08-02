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
"""Gnuplot visualization."""

import logging

import numpy as np

_logger = logging.getLogger(__name__)
try:
    from gnuplotlib import plot

except FileNotFoundError:
    _logger.warning("Cannot import gnuplotlib, no terminal plots available.")

    # Gnuplot is not available on certain system
    def plot(*_args, **_kwargs):
        """Dummy function if no gnuplot is available."""


def gnuplot_gp_convergence(iter_lst, fun_value_lst):
    """Convergence plots for Gaussian Process optimization and convergence.

    Args:
        iter_lst (lst): List with iteration numbers up to now
        fun_value_lst (lst): List with values of a function
    """
    plot(
        np.array(iter_lst).reshape(1, -1),
        np.array(fun_value_lst).reshape(1, -1),
        unset="grid",
        terminal="dumb 60,30",
        _with="lines",
    )
