#
# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (c) 2024, QUEENS contributors.
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
"""Pytest configuration for 4C integration tests."""

import pytest


@pytest.fixture(name="setup_symbolic_links_fourc", autouse=True)
def fixture_setup_symbolic_links_fourc(fourc_link_paths, fourc_build_paths_for_gitlab_runner):
    """Set-up of 4C symbolic links.

    Args:
        fourc_link_paths (Path): destination for symbolic links to executables
        fourc_build_paths_for_gitlab_runner (Path): Several paths that are needed to build symbolic
                                                links to executables
    """
    (
        dst_fourc,
        dst_post_ensight,
        dst_post_processor,
    ) = fourc_link_paths

    (
        fourc,
        post_ensight,
        post_processor,
    ) = fourc_build_paths_for_gitlab_runner
    # check if symbolic links are existent
    try:
        # create link to default 4C executable location if no link is available
        if not dst_fourc.is_symlink():
            if not fourc.is_file():
                raise FileNotFoundError(
                    f"Failed to create link to default 4C location.\n"
                    f"No 4C found under default location:\n"
                    f"\t{fourc}\n"
                )
            dst_fourc.symlink_to(fourc)
        # create link to default post_ensight location if no link is available
        if not dst_post_ensight.is_symlink():
            if not post_ensight.is_file():
                raise FileNotFoundError(
                    f"Failed to create link to default post_ensight location.\n"
                    f"No post_ensight found under default location:\n"
                    f"\t{post_ensight}\n"
                )
            dst_post_ensight.symlink_to(post_ensight)
        # create link to default post_processor location if no link is available
        if not dst_post_processor.is_symlink():
            if not post_processor.is_file():
                raise FileNotFoundError(
                    f"Failed to create link to default post_processor location.\n"
                    f"No post_processor found under default location:\n"
                    f"\t{post_processor}\n"
                )
            dst_post_processor.symlink_to(post_processor)

        # check if existing link to fourc works and points to a valid file
        if not dst_fourc.resolve().exists():
            raise FileNotFoundError(
                f"The following link seems to be dead: {dst_fourc}\n"
                f"It points to (non-existing): {dst_fourc.resolve()}\n"
            )
        # check if existing link to post_ensight works and points to a valid file
        if not dst_post_ensight.resolve().exists():
            raise FileNotFoundError(
                f"The following link seems to be dead: {dst_post_ensight}\n"
                f"It points to: {dst_post_ensight.resolve()}\n"
            )
        # check if existing link to post_processor works and points to a valid file
        if not dst_post_processor.resolve().exists():
            raise FileNotFoundError(
                f"The following link seems to be dead: {dst_post_processor}\n"
                f"It points to: {dst_post_processor.resolve()}\n"
            )
    except FileNotFoundError as error:
        raise FileNotFoundError(
            "Please make sure to make the missing executable available under the given "
            "path OR\n"
            "make sure the symbolic link under the config directory points to an "
            "existing file! \n"
            "You can create the necessary symbolic links on Linux via:\n"
            "-------------------------------------------------------------------------\n"
            "ln -s <path/to/fourc> <QUEENS_BaseDir>/config/4C\n"
            "ln -s <path/to/post_ensight> <QUEENS_BaseDir>/config/post_ensight\n"
            "ln -s <path/to/post_processor> <QUEENS_BaseDir>/config/post_processor\n"
            "-------------------------------------------------------------------------\n"
            "...and similar for the other links."
        ) from error
