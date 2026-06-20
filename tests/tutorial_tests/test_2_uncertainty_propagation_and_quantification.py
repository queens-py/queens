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
"""Integration tests for 1-grid-iterator-rosenbrock tutorial."""

import numpy as np
from testbook import testbook

from test_utils.path import relative_path_from_root
from test_utils.tutorial_tests import inject_mock_base_dir, markers_for_notebook

NOTEBOOK_PATH = relative_path_from_root(
    "tutorials/2_uncertainty_propagation_and_quantification.ipynb"
)

pytestmark = markers_for_notebook(NOTEBOOK_PATH)


@testbook(
    NOTEBOOK_PATH,
    timeout=-1,
)
def test_output_tutorial_2(tb, tmp_path):
    """Parameterized test case for tutorial 2.

    The tutorial is on Uncertainty Propagation and Quantification. The
    notebook is run with injected lines of codes for testing that the
    final results are as expected.
    """
    mu = np.array([0.006638095218949362, 0.0063343892089076466, 0.0021987605285805315]).tolist()
    x = np.array([0.203125, 0.5, 0.796875]).tolist()

    # inject testing cells
    tb.inject(
        f"""for i, samples_at_location in enumerate(monte_carlo_sample_outputs.T):
    np.testing.assert_allclose(mesh.p.T[index[i]][0], np.array({x}[i]))
    np.testing.assert_allclose(mean[index[i]], np.array({mu}[i]))""",
        after=20,
        run=False,
    )
    inject_mock_base_dir(tb, tmp_path)

    tb.execute()
