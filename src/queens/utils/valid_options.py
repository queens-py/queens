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
"""Helper functions for valid options and switch analogy."""

from typing import Any

from queens.utils.exceptions import InvalidOptionError


def get_option(options_dict: dict[str, Any], desired_option: str, error_message: str = "") -> Any:
    """Get option *desired_option* from *options_dict*.

    The *options_dict* consists of the keys and their values. Note that the value can also be
    functions. In case the option is not found an error is raised.

    Args:
        options_dict: Dictionary with valid options and their value
        desired_option: Desired method key
        error_message: Custom error message to be used if the *desired_option* is not found.

    Returns:
        Value of the desired option
    """
    check_if_valid_options(list(options_dict.keys()), desired_option, error_message)
    return options_dict[desired_option]


def check_if_valid_options(
    valid_options: list | dict,
    desired_options: str | dict[str, int] | list[str],
    error_message: str = "",
) -> None:
    """Check if the desired option(s) is/are in valid_options.

    Args:
        valid_options: List of valid option keys or dict with valid options as keys
        desired_options: Key(s) of desired options
        error_message: Error message in case the desired option can not be found

    Raises:
        InvalidOptionError: If any of the desired options is in invalid options
    """
    desired_options_set = set(desired_options)
    if isinstance(desired_options, str):
        desired_options_set = {desired_options}

    # Set of options that are not valid
    invalid_options = (desired_options_set ^ set(valid_options)) - set(valid_options)

    if invalid_options:
        raise InvalidOptionError.construct_error_from_options(
            valid_options, ", ".join(desired_options_set), error_message
        )
