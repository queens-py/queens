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
"""Wrapped functions of subprocess stdlib module."""

import logging
import subprocess

from queens.utils.exceptions import SubprocessError

_logger = logging.getLogger(__name__)

# Currently allowed errors that might appear but have no effect on subprocesses
_ALLOWED_ERRORS = ["Invalid MIT-MAGIC-COOKIE-1 key", "No protocol specified"]


def run_subprocess(
    command: str,
    raise_error_on_subprocess_failure: bool = True,
    additional_error_message: str | None = None,
    allowed_errors: list[str] | None = None,
) -> tuple[int, int, str, str]:
    """Run a system command outside of the Python script.

    Args:
        command: Command that will be run in subprocess
        raise_error_on_subprocess_failure: Raise or warn error defaults to True
        additional_error_message: Additional error message to be displayed
        allowed_errors: List of strings to be removed from the error message

    Returns:
        Code for success of subprocess
        Unique process ID that was assigned to the subprocess on computing machine
        Standard output content
        Standard error content
    """
    process = start_subprocess(command)

    stdout, stderr = process.communicate()
    process_id = process.pid
    process_returncode = process.returncode

    _raise_or_warn_error(
        command=command,
        stdout=stdout,
        stderr=stderr,
        raise_error_on_subprocess_failure=raise_error_on_subprocess_failure,
        additional_error_message=additional_error_message,
        allowed_errors=allowed_errors,
    )
    return process_returncode, process_id, stdout, stderr


def start_subprocess(command: str) -> subprocess.Popen:
    """Start subprocess.

    Args:
        command: Command that will be run in subprocess

    Returns:
        Subprocess object
    """
    process = subprocess.Popen(  # pylint: disable=consider-using-with
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=True,
        universal_newlines=True,
    )
    return process


def _raise_or_warn_error(
    command: str,
    stdout: str,
    stderr: str,
    raise_error_on_subprocess_failure: bool,
    additional_error_message: str | None,
    allowed_errors: list[str] | None,
) -> None:
    """Raise or warn eventual exception if subprocess fails.

    Args:
        command: Command string
        stdout: Command output
        stderr: Error of the output
        raise_error_on_subprocess_failure: Raise or warn error defaults to True
        additional_error_message: Additional error message to be displayed
        allowed_errors: List of strings to be removed from the error message
    """
    # Check for allowed error messages and remove them
    if allowed_errors is None:
        allowed_errors = []

    stderr = _remove_allowed_errors(stderr, allowed_errors)
    if stderr:
        subprocess_error = SubprocessError.construct_error_from_command(
            command, stdout, stderr, additional_error_message
        )
        if raise_error_on_subprocess_failure:
            raise subprocess_error
        _logger.warning(str(subprocess_error))


def _remove_allowed_errors(stderr: str, allowed_errors: list[str]) -> str:
    """Remove allowed error messages from error output.

    Args:
        stderr: Error message
        allowed_errors: Allowed error messages

    Returns:
        Error message without allowed errors
    """
    # Add known exceptions
    allowed_errors.extend(_ALLOWED_ERRORS)
    # Remove the allowed error messages from stderr
    for error_message in allowed_errors:
        stderr = stderr.replace(error_message, "")

    # Remove trailing spaces, tabs and newlines and check if an error message remains
    if "".join(stderr.split()) == "":
        stderr = ""

    return stderr
