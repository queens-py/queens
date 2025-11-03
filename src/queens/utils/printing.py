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
"""Print utils."""

DEFAULT_OUTPUT_WIDTH = 101


def get_str_table(name: str, print_dict: dict, use_repr: bool = False) -> str:
    """Function to get table to be used in *__str__* methods.

    Args:
        name: Object name
        print_dict: Dict containing labels and values to print
        use_repr: If true, use repr() function to obtain string representations of objects

    Returns:
        Table to print
    """
    column_name = [str(k) for k in print_dict.keys()]
    str_fun = repr if use_repr else str
    column_value = [str_fun(v).replace("\n", " ") for v in print_dict.values()]
    column_width_name = max(len(s) for s in column_name)
    column_width_value = max(len(s) for s in column_value)

    data_template = f"{{:<{column_width_name}}} : {{:<{column_width_value}}}"

    # find max width and create seperators
    seperator_width = max(
        max(len(data_template.format("", "")), len(name)) + 4, DEFAULT_OUTPUT_WIDTH
    )
    line_template = f"| {{:{seperator_width-4}}} |\n"
    main_seperator_line = "+" + "-" * (seperator_width - 2) + "+\n"
    soft_separator_line = (
        "|" + "- " * ((seperator_width - 2) // 2) + "-" * (seperator_width % 2) + "|\n"
    )

    # Create table string
    string = "\n" + main_seperator_line
    string += f"| {{:^{seperator_width-4}}} |\n".format(name)
    string += soft_separator_line
    for field_name, value in zip(column_name, column_value):
        content = data_template.format(field_name, value)
        string += line_template.format(content)
    string += main_seperator_line
    return string
