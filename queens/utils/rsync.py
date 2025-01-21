#
# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (c) 2025, QUEENS contributors.
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
"""Rsync utils."""

import logging
from pathlib import Path

from queens.utils.path_utils import is_empty
from queens.utils.run_subprocess import run_subprocess

_logger = logging.getLogger(__name__)


def assemble_rsync_command(
    source,
    destination,
    archive=False,
    exclude=None,
    filters=None,
    verbose=True,
    rsh=None,
    host=None,
    rsync_options=None,
):
    """Assemble rsync command.

    Args:
        source (str, Path, list): paths to copy
        destination (str, Path): destination relative to host
        archive (bool): use the archive option
        exclude (str, list): options to exclude
        filters (str): filters for rsync
        verbose (bool): true for verbose
        rsh (str): remote ssh command
        host (str): host where to copy the files to
        rsync_options (list): additional rsync options

    Returns:
        str command to run rsync
    """

    def listify(obj):
        if isinstance(obj, (str, Path)):
            return [obj]
        return obj

    options = []
    if archive:
        options.append("--archive")
    if verbose:
        options.append("--verbose")
    if filters:
        options.append(f"--filter='{filters}'")
    if exclude:
        for e in listify(exclude):
            options.append(f"--exclude='{e}'")
    if rsync_options:
        options.extend(listify(rsync_options))
    if rsh:
        options.append(f"--rsh='{rsh}'")
    source = listify(source)
    if host:
        destination = f"{host}:{destination}"

    options = " ".join([str(option) for option in options])
    source = " ".join([str(file) for file in source])
    command = f"rsync {options} {source} {destination}/"
    return command


def rsync(
    source,
    destination,
    archive=True,
    exclude=None,
    filters=None,
    verbose=True,
    rsh=None,
    host=None,
    rsync_options=None,
):
    """Run rsync command.

    Args:
        source (str, Path, list): paths to copy
        destination (str, Path): destination relative to host
        archive (bool): use the archive option
        exclude (str, list): options to exclude
        filters (str): filters for rsync
        verbose (bool): true for verbose
        rsh (str): remote ssh command
        host (str): host where to copy the files to
        rsync_options (list): additional rsync options
    """
    if not is_empty(source):
        command = assemble_rsync_command(
            source=source,
            destination=destination,
            archive=archive,
            exclude=exclude,
            filters=filters,
            verbose=verbose,
            rsh=rsh,
            host=host,
            rsync_options=rsync_options,
        )

        run_subprocess(command)
    else:
        _logger.debug("List of source files was empty. Did not copy anything.")
