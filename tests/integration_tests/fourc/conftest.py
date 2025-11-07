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
def fixture_setup_symbolic_links_fourc(fourc_link_paths):
    """Set-up of 4C symbolic links.

    Args:
        fourc_link_paths (Path): Symbolic links to 4C executables.
    """
    (
        fourc,
        post_ensight,
        post_processor,
    ) = fourc_link_paths

    # check if symbolic links are existent
    try:
        # check if existing link to fourc works and points to a valid file
        if not fourc.resolve().exists():
            raise FileNotFoundError(
                f"The following link seems to be dead: {fourc}\n"
                f"It points to (non-existing): {fourc.resolve()}\n"
            )
        # check if existing link to post_ensight works and points to a valid file
        if not post_ensight.resolve().exists():
            raise FileNotFoundError(
                f"The following link seems to be dead: {post_ensight}\n"
                f"It points to: {post_ensight.resolve()}\n"
            )
        # check if existing link to post_processor works and points to a valid file
        if not post_processor.resolve().exists():
            raise FileNotFoundError(
                f"The following link seems to be dead: {post_processor}\n"
                f"It points to: {post_processor.resolve()}\n"
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
