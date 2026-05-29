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
"""Test the readme QUEENS example."""

import os
import subprocess
import sys

from test_utils.get_queens_example_from_readme import get_queens_example_from_readme


def test_readme_example(tmp_path):
    """Test if the example in the readme runs."""
    # Disable plotting in the script
    example_source = "import matplotlib"
    example_source += "\nmatplotlib.use('Agg')\n"

    # Get the source of the example
    example_source += get_queens_example_from_readme(tmp_path)

    # Create script
    script_path = tmp_path / "script.py"
    script_path.write_text(example_source)

    # Run the script
    environment = os.environ.copy()
    environment["MPLCONFIGDIR"] = str(tmp_path / "matplotlib")
    process = subprocess.run(
        [sys.executable, str(script_path)],
        check=False,
        capture_output=True,
        env=environment,
        text=True,
    )

    assert process.returncode == 0, (
        f"README example failed with exit code {process.returncode}.\n"
        f"stdout:\n{process.stdout}\n"
        f"stderr:\n{process.stderr}"
    )
