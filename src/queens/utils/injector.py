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
"""Injector module.

The module supplies functions to inject parameter values into a template
text file.
"""

from pathlib import Path

from jinja2 import Environment, StrictUndefined, Undefined

from queens.utils.io import read_file


def render_template(params, template, strict=True):
    """Function to insert parameters into a template.

    Args:
        params (dict): Dict with parameters to inject
        template (str): Template file as string
        strict (bool): Raises exception if required parameters from the template are missing

    Returns:
        str: injected template
    """
    undefined = StrictUndefined if strict else Undefined

    environment = Environment(undefined=undefined).from_string(template)
    return environment.render(**params)


def inject_in_template(params, template, output_file, strict=True):
    """Function to insert parameters into file template and write to file.

    Args:
        params (dict): Dict with parameters to inject
        template (str): Template (str)
        output_file (str, Path): Name of output file with injected parameters
        strict (bool): Raises exception if mismatch between provided and required parameters
    """
    injected_template = render_template(params, template, strict)
    Path(output_file).write_text(injected_template, encoding="utf-8")


def inject(params, template_path, output_file, strict=True):
    """Function to insert parameters into file template and write to file.

    Args:
        params (dict): Dict with parameters to inject
        template_path (str, Path): Path to template
        output_file (str, Path): Name of output file with injected parameters
        strict (bool): Raises exception if mismatch between provided and required parameters
    """
    template = read_file(template_path)
    inject_in_template(params, template, output_file, strict)
