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
"""Unit tests for 1-grid-iterator-rosenbrock tutorial."""

import numpy as np
from testbook import testbook


# tested jupyter notebooks should be mentioned below
@testbook(
    "tutorials/1-grid-iterator-rosenbrock/1-grid-iterator-rosenbrock.ipynb",
)
def test_result_output(tb, tmp_path):
    """Parameterized test case for Jupyter notebook output.

    The notebook is run with injected lines of code, as intended by the
    tutorial
    """
    optimal_fun = 2.986025e-11
    optimal_x = np.array([0.99999463, 0.99998915]).tolist()

    tb.inject(f"output_dir = '{tmp_path}'")
    tb.inject("experiment_name = 'grid_iterator_rosenbrock'")
    tb.inject("from queens.utils import config_directories")
    tb.inject(f"config_directories = '{tmp_path}'")
    tb.inject("from queens.global_settings import GlobalSettings")
    tb.inject(
        """global_settings = GlobalSettings(experiment_name=experiment_name,
        output_dir=output_dir,
        debug=False)"""
    )

    tb.inject(
        """def rosenbrock(x1, x2):
            a = 1
            b = 100
            f = (a - x1) **2 + b* (x2 - x1 **2)**2
            return f"""
    )

    tb.inject(
        """import numpy as np
x1 = np.linspace(-2.0, 2.0, 10)
x2 = np.linspace(-3.0, 3.0, 10)
X1, X2 = np.meshgrid(x1, x2)"""
    )

    tb.inject(
        """from queens.distributions import Uniform
from queens.parameters import Parameters
x1 = Uniform(lower_bound=-2.0, upper_bound=2.0)
x2 = Uniform(lower_bound=-3.0, upper_bound=3.0)
parameters = Parameters(x1=x1, x2=x2)"""
    )

    tb.inject(
        """grid_design = {
        "x1": {"num_grid_points": 10, "axis_type": "lin", "data_type": "FLOAT"},
        "x2": {"num_grid_points": 10, "axis_type": "lin", "data_type": "FLOAT"},
    }"""
    )

    tb.inject(
        """def visualize_grid_and_surface(X1_QUEENS,X2_QUEENS, Z_QUEENS):
              return None"""
    )

    tb.execute_cell([2, 8, 17, 21, 23, 25, 27])
    tb.inject(
        """np.testing.assert_allclose(X1, X1_QUEENS)
np.testing.assert_allclose(X2, X2_QUEENS)
np.testing.assert_allclose(Z, Z_QUEENS)"""
    )

    tb.execute_cell([31])
    tb.inject(f"np.testing.assert_allclose(optimal_fun, {optimal_fun},rtol=1e-4)")
    tb.inject(f"np.testing.assert_allclose(optimal_x, np.array({optimal_x}))")
