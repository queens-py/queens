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
"""Import utils."""

import importlib.util
import logging
import sys
from pathlib import Path

from queens.utils.path import check_if_path_exists
from queens.utils.valid_options import get_option

_logger = logging.getLogger(__name__)


def get_module_attribute(path_to_module, function_or_class_name):
    """Load function from python file by path.

    Args:
        path_to_module (Path | str): "Path" to file
        function_or_class_name (str): Name of the function

    Returns:
        function or class: Function or class from the module
    """
    # Set the module name
    module_path_obj = Path(path_to_module)
    module_name = module_path_obj.stem
    module_ending = module_path_obj.suffix

    # Check if file exists
    if not check_if_path_exists(module_path_obj):
        raise FileNotFoundError(f"Could not find python file {path_to_module}.")

    # Check if ending is correct
    if module_ending != ".py":
        raise ImportError(f"Python file {path_to_module} does not have a .py ending")

    # Load the module
    spec = importlib.util.spec_from_file_location(module_name, path_to_module)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules[module_name] = module

    try:
        # Check if function can be loaded
        function = getattr(module, function_or_class_name)
    except AttributeError as error:
        raise AttributeError(
            f"External python module {path_to_module} does not have an attribute called "
            f"{function_or_class_name}"
        ) from error

    _logger.debug(
        "Using now external Python method or class %s \nin the file %s.",
        function_or_class_name,
        path_to_module,
    )
    return function


def get_module_class(module_options, valid_types, module_type_specifier="type"):
    """Return module class defined in config file.

    Args:
        module_options (dict): Module options
        valid_types (dict): Dict of valid types with corresponding module paths and class names
        module_type_specifier (str): Specifier for the module type

    Returns:
        module_class (class): Class from the module
    """
    # determine which object to create
    module_type = module_options.pop(module_type_specifier)
    if module_options.get("external_python_module"):
        module_path = module_options.pop("external_python_module")
        module_class = get_module_attribute(module_path, module_type)
    else:
        module_class = get_option(valid_types, module_type)

    return module_class


class LazyLoader:
    """Lazy loader for modules that take long to load.

    Inspired from https://stackoverflow.com/a/78312617
    """

    def __init__(self, module_name):
        """Initialize the loader.

        Args:
            module_name (str): name of the module to be imported
        """
        self._module_name = module_name
        self._module = None

    def __getattr__(self, attr):
        """Get attribute.

        Args:
            attr (str): Attribute name

        Returns:
            obj: attribute
        """
        if self._module is None:
            self._module = importlib.import_module(self._module_name)

        return getattr(self._module, attr)
