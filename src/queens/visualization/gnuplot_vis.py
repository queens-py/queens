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
import shutil
from functools import cache

import numpy as np

_logger = logging.getLogger(__name__)

try:
    from gnuplotlib import plot

# gnuplotlib versions before 0.47 already raise on import if gnuplot is missing
except FileNotFoundError:
    _logger.warning("Cannot import gnuplotlib, no terminal plots available.")

    # Gnuplot may not be available on some systems
    def plot(*_args, **_kwargs):
        """Dummy function if gnuplot is not available."""


@cache
def gnuplot_available():
    """Check whether the gnuplot executable is available.

    The gnuplot executable is a system dependency of gnuplotlib and is not installed on every
    system. Depending on the gnuplotlib version, a missing executable raises a `FileNotFoundError`
    either on import of gnuplotlib or on the first plot call, hence it is checked explicitly here.
    The result is cached so that the warning is only emitted once.

    Returns:
        bool: True if the gnuplot executable was found on the PATH
    """
    if shutil.which("gnuplot") is None:
        _logger.warning("Cannot find the gnuplot executable, no terminal plots available.")
        return False
    return True


def gnuplot_gp_convergence(iter_lst, fun_value_lst):
    """Convergence plots for Gaussian Process optimization and convergence.

    Args:
        iter_lst (lst): List with iteration numbers up to now
        fun_value_lst (lst): List with values of a function
    """
    if not gnuplot_available():
        return

    plot(
        np.array(iter_lst).reshape(1, -1),
        np.array(fun_value_lst).reshape(1, -1),
        unset="grid",
        terminal="dumb 60,30",
        _with="lines",
    )
