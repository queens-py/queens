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
"""Test copying with rsync."""

import filecmp

import pytest

from queens.utils.rsync import rsync


@pytest.fixture(name="_create_source_files")
def fixture_create_source_files(source_path, files_to_copy):
    """Create source files."""
    for file in files_to_copy:
        (source_path / file).write_text(file)


@pytest.fixture(name="source_path")
def fixture_source_path(tmp_path):
    """Source path."""
    source_path = tmp_path / "source"
    source_path.mkdir()
    return source_path


@pytest.fixture(name="destination_path")
def fixture_destination_path(tmp_path):
    """Destination path."""
    destination_path = tmp_path / "destination"
    return destination_path


def test_single_file(_create_source_files, source_path, destination_path, files_to_copy):
    """Test copy a single file."""
    source_file = source_path / files_to_copy[0]
    rsync(source_file, destination_path)
    assert filecmp.cmp(destination_path / files_to_copy[0], source_file)


def test_multiple_files(_create_source_files, source_path, destination_path, files_to_copy):
    """Test copy a list of files."""
    source_files = [source_path / file for file in files_to_copy]
    rsync(source_files, destination_path)
    match, mismatch, errors = filecmp.cmpfiles(destination_path, source_path, common=files_to_copy)
    assert len(match) == len(files_to_copy)  # all files are copied
    assert not mismatch  # no mismatches
    assert not errors  # no errors


def test_directory(_create_source_files, source_path, destination_path, files_to_copy):
    """Test copy a directory."""
    rsync(source_path, destination_path)
    match, mismatch, errors = filecmp.cmpfiles(
        destination_path / source_path.name, source_path, common=files_to_copy
    )
    assert len(match) == len(files_to_copy)  # all files are copied
    assert not mismatch  # no mismatches
    assert not errors  # no errors
