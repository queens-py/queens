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
"""Utility methods used by the unit tests for jupyter notebooks."""


def inject_mock_path(tb, tmp_path):
    """Inject a mock temporary path for testing notebooks.

    Args:
        tb (testbook): testbook object for inserting code into the notebook
        tmp_path: pytest tmp_path fixture for experiment directory

    Returns:
        None
    """
    tb.inject(
        f"""
        from unittest.mock import MagicMock
        from pathlib import Path
        import queens.utils.config_directories
        mock_base_dir = Path('{tmp_path}')
        queens.utils.config_directories.base_directory = MagicMock(return_value=mock_base_dir)
        """,
        before=0,
    )
