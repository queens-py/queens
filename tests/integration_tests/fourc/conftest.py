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
"""Fixtures for the 4C integration tests."""

import pytest


@pytest.fixture(name="setup_symbolic_links_fourc", autouse=True)
def fixture_setup_symbolic_links_fourc(fourc_link):
    """Set-up of 4C symbolic links.

    Args:
        fourc_link (Path): Symbolic link to 4C executable.
    """
    # check if symbolic links are existent
    try:
        # check if existing link to fourc works and points to a valid file
        if not fourc_link.resolve().exists():
            raise FileNotFoundError(
                f"The following link seems to be dead: {fourc_link}\n"
                f"It points to (non-existing): {fourc_link.resolve()}\n"
            )
    except FileNotFoundError as error:
        raise FileNotFoundError(
            "Please make sure to make the missing executable available under the given "
            "path OR\n"
            "make sure the symbolic link in the config directory points to the build directory of "
            "4C! \n"
            "You can create the necessary symbolic link on Linux via:\n"
            "-------------------------------------------------------------------------\n"
            "ln -s <path-to-4C-build-directory> <queens-base-dir>/config/4C_build\n"
            "-------------------------------------------------------------------------\n"
        ) from error
