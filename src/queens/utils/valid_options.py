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

from queens.utils.exceptions import InvalidOptionError


def get_option(options_dict, desired_option, error_message=""):
    """Get option *desired_option* from *options_dict*.

    The *options_dict* consists of the keys and their values. Note that the value can also be
    functions. In case the option is not found an error is raised.

    Args:
        options_dict (dict): Dictionary with valid options and their value
        desired_option (str): Desired method key
        error_message (str, optional): Custom error message to be used if the *desired_option* is
                                       not found. Defaults to an empty string.

    Returns:
        Value of the *desired_option*
    """
    check_if_valid_options(list(options_dict.keys()), desired_option, error_message)
    return options_dict[desired_option]


def check_if_valid_options(valid_options, desired_options, error_message=""):
    """Check if the desired option(s) is/are in valid_options.

    Raises InvalidOptionError if invalid options are present.

    Args:
        valid_options (lst,dict): List of valid option keys or dict with valid options as keys
        desired_options (str, lst(str), dict): Key(s) of desired options
        error_message (str, optional): Error message in case the desired option can not be found
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
