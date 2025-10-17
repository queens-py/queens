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
"""Custom exceptions."""

from __future__ import annotations


class QueensException(Exception):
    """QUEENS exception."""


class FileTypeError(QueensException):
    """Exception for wrong file types."""


class CLIError(QueensException):
    """QUEENS exception for CLI input."""


class InvalidOptionError(QueensException):
    """Custom error class for invalid options during QUEENS runs."""

    @classmethod
    def construct_error_from_options(
        cls, valid_options: dict | list, desired_option: str, additional_message: str = ""
    ) -> InvalidOptionError:
        """Construct invalid option error from the valid and desired options.

        Args:
            valid_options: List of valid option keys
            desired_option: Key of desired option
            additional_message: Additional message to pass (default is None)

        Returns:
            InvalidOptionError
        """
        message = "\n" + additional_message
        message += f"\nInvalid option(s) '{desired_option}'. Valid options are:\n" + ", ".join(
            sorted(valid_options)
        )
        return cls(message)


class SubprocessError(QueensException):
    """Custom error class for the QUEENS subprocess wrapper."""

    @classmethod
    def construct_error_from_command(
        cls,
        command: str,
        command_output: str,
        error_message: str,
        additional_message: str | None = "",
    ) -> SubprocessError:
        """Construct a Subprocess error from a command and its outputs.

        Args:
            command: Command used that raised the error
            command_output: Command output
            error_message: Error message of the command
            additional_message: Additional message to pass

        Returns:
            SubprocessError
        """
        message = "\n\nQUEENS' subprocess wrapper caught the following error:\n"
        message += error_message
        message += "\n\n\nwith commandline output:\n"
        message += str(command_output)
        message += "\n\n\nwhile executing the command:\n" + command
        if additional_message:
            message += "\n\n" + additional_message
        return cls(message)
