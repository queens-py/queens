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
import re
import sys
from pathlib import Path
from typing import Any, Callable

from queens.utils.path import check_if_path_exists
from queens.utils.valid_options import get_option

_logger = logging.getLogger(__name__)


def get_module_attribute(path_to_module: str | Path, function_or_class_name: str) -> Callable:
    """Load function from python file by path.

    Args:
        path_to_module: Path to file
        function_or_class_name: Name of the function

    Returns:
        Function or class from the module
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
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module {module_name} from path {path_to_module}.")
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


def get_module_class(
    module_options: dict, valid_types: dict, module_type_specifier: str = "type"
) -> Any:
    """Return module class defined in config file.

    Args:
        module_options: Module options
        valid_types: Dict of valid types with corresponding module paths and class names
        module_type_specifier: Specifier for the module type

    Returns:
        Class from the module
    """
    # determine which object to create
    module_type = module_options.pop(module_type_specifier)
    if module_options.get("external_python_module"):
        module_path = module_options.pop("external_python_module")
        module_class = get_module_attribute(module_path, module_type)
    else:
        module_class = get_option(valid_types, module_type)

    return module_class


def import_class_from_class_module_map(
    name: str, class_module_map: dict, package: str | None = None
) -> Any:
    """Import class from class_module_map.

    Args:
        name: Name of the class.
        class_module_map: Class to module mapping.
        package (str, opt): Package name (only necessary if import path is relative)

    Returns:
        class (obj): Class object.
    """
    if name in class_module_map:
        module = importlib.import_module(class_module_map[name], package=package)
        return getattr(module, name)
    raise AttributeError


def extract_type_checking_imports(file_path: str) -> dict:
    """Extract imports inside TYPE_CHECKING blocks from file.

    Args:
        file_path: Path to the file

    Returns:
        A dict mapping class names to their source modules.
    """
    inside_type_checking = False

    with open(file_path, "r", encoding="utf-8") as file:
        import_text = ""

        for line in file:
            stripped_line = line.partition("#")[0]  # remove comments from line
            stripped_line = stripped_line.strip()

            # Detect start of TYPE_CHECKING block
            if stripped_line.replace(" ", "") == "ifTYPE_CHECKING:":
                inside_type_checking = True
                continue

            if inside_type_checking:
                # Detect end of block (naively assumes dedent)
                if len(line.lstrip()) == len(line):
                    inside_type_checking = False
                    continue
                import_text += stripped_line

        mapping = {}
        import_statements = import_text.split("from")[1:]
        for import_statement in import_statements:
            module, class_names = import_statement.split("import")
            module = module.strip()
            class_names_lst = class_names.split(",")
            for class_name in class_names_lst:
                mapping[re.sub(r"\W+", "", class_name)] = module

    return mapping


class LazyLoader:
    """Lazy loader for modules that take long to load.

    Inspired from https://stackoverflow.com/a/78312617
    """

    def __init__(self, module_name: str):
        """Initialize the loader.

        Args:
            module_name: name of the module to be imported
        """
        self._module_name = module_name
        self._module = None

    def __getattr__(self, attr: str) -> Any:
        """Get attribute.

        Args:
            attr: Attribute name

        Returns:
            attribute
        """
        if self._module is None:
            self._module = importlib.import_module(self._module_name)  # type: ignore[assignment]

        return getattr(self._module, attr)
