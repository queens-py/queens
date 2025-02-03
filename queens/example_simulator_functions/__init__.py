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
"""Example simulator functions.

Modules containing example simulator functions for testing and
benchmarking.
"""

from queens.example_simulator_functions.agawal09 import agawal09a
from queens.example_simulator_functions.borehole83 import borehole83_hifi, borehole83_lofi
from queens.example_simulator_functions.branin78 import branin78_hifi, branin78_lofi, branin78_medfi
from queens.example_simulator_functions.currin88 import currin88_hifi, currin88_lofi
from queens.example_simulator_functions.gardner14a import gardner14a
from queens.example_simulator_functions.ishigami90 import ishigami90
from queens.example_simulator_functions.ma09 import ma09
from queens.example_simulator_functions.oakley_ohagan04 import oakley_ohagan04
from queens.example_simulator_functions.parabola_residual import parabola_residual
from queens.example_simulator_functions.paraboloid import paraboloid
from queens.example_simulator_functions.park91a import (
    park91a_hifi,
    park91a_hifi_on_grid,
    park91a_hifi_on_grid_with_gradients,
    park91a_lofi,
    park91a_lofi_on_grid,
    park91a_lofi_on_grid_with_gradients,
)
from queens.example_simulator_functions.park91b import park91b_hifi, park91b_lofi
from queens.example_simulator_functions.perdikaris17 import perdikaris17_hifi, perdikaris17_lofi
from queens.example_simulator_functions.rosenbrock60 import (
    rosenbrock60,
    rosenbrock60_residual,
    rosenbrock60_residual_1d,
    rosenbrock60_residual_3d,
)
from queens.example_simulator_functions.sinus import sinus_test_fun
from queens.example_simulator_functions.sobol_g_function import sobol_g_function
from queens.utils.valid_options_utils import get_option

VALID_EXAMPLE_SIMULATOR_FUNCTIONS = {
    "agawal09a": agawal09a,
    "borehole83_lofi": borehole83_lofi,
    "borehole83_hifi": borehole83_hifi,
    "branin78_lofi": branin78_lofi,
    "branin78_medfi": branin78_medfi,
    "branin78_hifi": branin78_hifi,
    "currin88_lofi": currin88_lofi,
    "currin88_hifi": currin88_hifi,
    "gardner14a": gardner14a,
    "ishigami90": ishigami90,
    "ma09": ma09,
    "oakley_ohagan04": oakley_ohagan04,
    "paraboloid": paraboloid,
    "parabola_residual": parabola_residual,
    "park91a_lofi_on_grid": park91a_lofi_on_grid,
    "park91a_hifi_on_grid": park91a_hifi_on_grid,
    "park91a_hifi_on_grid_with_gradients": park91a_hifi_on_grid_with_gradients,
    "park91a_lofi_on_grid_with_gradients": park91a_lofi_on_grid_with_gradients,
    "park91a_lofi": park91a_lofi,
    "park91a_hifi": park91a_hifi,
    "park91b_lofi": park91b_lofi,
    "park91b_hifi": park91b_hifi,
    "perdikaris17_lofi": perdikaris17_lofi,
    "perdikaris17_hifi": perdikaris17_hifi,
    "rosenbrock60": rosenbrock60,
    "rosenbrock60_residual": rosenbrock60_residual,
    "rosenbrock60_residual_1d": rosenbrock60_residual_1d,
    "rosenbrock60_residual_3d": rosenbrock60_residual_3d,
    "sinus_test_fun": sinus_test_fun,
    "sobol_g_function": sobol_g_function,
    "patch_for_likelihood": lambda x: 42,
}


def example_simulator_function_by_name(function_name):
    """Get example simulator function by name.

    Args:
        function_name (str): Name of the example simulator function

    Returns:
        func: Function
    """
    return get_option(VALID_EXAMPLE_SIMULATOR_FUNCTIONS, function_name)
