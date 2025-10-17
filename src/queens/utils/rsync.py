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
"""Rsync utils."""

import logging
from pathlib import Path
from typing import Any, Sequence

from queens.utils.path import is_empty
from queens.utils.run_subprocess import run_subprocess

_logger = logging.getLogger(__name__)


def assemble_rsync_command(
    source: str | Path | Sequence,
    destination: Path | str,
    archive: bool = False,
    exclude: str | Sequence | None = None,
    filters: str | None = None,
    verbose: bool = True,
    rsh: str | None = None,
    host: str | None = None,
    rsync_options: Sequence | None = None,
) -> str:
    """Assemble rsync command.

    Args:
        source: Paths to copy
        destination: Destination relative to host
        archive: Use the archive option
        exclude: Options to exclude
        filters: Filters for rsync
        verbose: True for verbose
        rsh: Remote ssh command
        host: Host to which to copy the files
        rsync_options: Additional rsync options

    Returns:
        Command to run rsync
    """

    def listify(obj: Any) -> Any:
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
    if host:
        destination = f"{host}:{destination}"

    options_string = " ".join([str(option) for option in options])
    source_string = " ".join([str(file) for file in listify(source)])
    command = f"rsync {options_string} {source_string} {destination}/"
    return command


def rsync(
    source: str | Path | Sequence,
    destination: str | Path,
    archive: bool = True,
    exclude: str | Sequence | None = None,
    filters: str | None = None,
    verbose: bool = True,
    rsh: str | None = None,
    host: str | None = None,
    rsync_options: Sequence | None = None,
) -> None:
    """Run rsync command.

    Args:
        source: Paths to copy
        destination: Destination relative to host
        archive: Use the archive option
        exclude: Options to exclude
        filters: Filters for rsync
        verbose: True for verbose
        rsh: Remote ssh command
        host: Host where to copy the files to
        rsync_options: Additional rsync options
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
