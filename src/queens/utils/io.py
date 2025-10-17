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
"""Utils for input/output handling."""

import csv
import json
import logging
import pickle
from pathlib import Path
from typing import Any, Callable

import numpy as np
import yaml

from queens.utils.exceptions import FileTypeError

_logger = logging.getLogger(__name__)


def load_pickle(file_path: Path) -> dict:
    """Load a pickle file directly from path.

    Args:
        file_path: Path to pickle-file

    Returns:
        Data in the pickle file
    """
    if not file_path.is_file():
        raise FileNotFoundError(f"File {file_path} does not exist.")
    try:
        with open(file_path, "rb") as file:
            data = pickle.load(file)
        return data
    except Exception as exception:
        raise IOError(f"Could not open the pickle file {file_path}") from exception


def print_pickled_data(file_path: Path) -> None:
    """Print a table of the data within a pickle file.

    Only goes one layer deep for dicts. This is similar to *python -m pickle file_path* but makes
    it a single command and fancy prints.

    Args:
        file_path: Path to pickle-file
    """
    data = load_pickle(file_path)
    _logger.info("\n\npickle file: %s", file_path)

    if isinstance(data, dict):
        for key, item in data.items():
            item_type = type(item)
            if isinstance(item, dict):
                string = ""
                for subkey, subitem in item.items():
                    string += str(
                        _create_single_item_string(subkey, subitem, type(subitem), separator="-")
                        + "\n"
                    )
                item = string.replace("\n", "\n    ")
            _logger.info(_create_single_item_string(key, item, item_type))
            _logger.info(" ")


def _create_single_item_string(key: str, item: Any, item_type: type, separator: str = "=") -> str:
    """Create a table for a single item.

    Args:
        key: Key of the item
        item: Item value for the key
        item_type: Type of the item value
        separator: Create separator line (default is "=")

    Returns:
        Table for this item.
    """
    string = (
        separator * 60
        + f"\nKey:  {key}\n"
        + f"Type: {item_type}\n"
        + f"Value:\n{item}\n"
        + separator * 60
    )
    return string


def load_input_file(input_file_path: Path) -> dict:
    """Load inputs from file by path.

    Args:
        input_file_path: Path to the input file

    Returns:
        Options in the input file.
    """
    if not input_file_path.is_file():
        raise FileNotFoundError(f"Input file {input_file_path} does not exist.")

    file_type = input_file_path.suffix.lower()
    loader: Callable
    if file_type == ".json":
        loader = json.load
    elif file_type in [".yml", ".yaml"]:
        loader = yaml.safe_load
    else:
        raise FileTypeError(
            f"Only json or yaml/yml files allowed, not of type '{file_type}' ({input_file_path})"
        )
    with open(input_file_path, "r", encoding="utf-8") as stream:
        try:
            options = loader(stream)
        except Exception as exception:
            raise type(exception)(
                f"Could not load file {input_file_path}. Verify the syntax."
            ) from exception
    return options


def load_result(path_to_result_file: Path) -> Any:
    """Load QUEENS results.

    Args:
        path_to_result_file: Path to results
    Returns:
        Results
    """
    path_to_result_file = Path(path_to_result_file)
    results = load_pickle(path_to_result_file)
    return results


def write_to_csv(output_file_path: Path, data: np.ndarray, delimiter: str = ",") -> None:
    """Write a simple csv file.

    Write data out to a csv-file. Nothing fancy, at the moment,
    only now header line or index column is supported just pure data.

    Args:
        output_file_path: Path to the file the data should be written to
        data: Data that should be written to the csv file.
        delimiter: Delimiter to separate individual data. Defaults to comma delimiter.
    """
    # Write data to new file
    with open(output_file_path, "w", newline="", encoding="utf-8") as out_file:
        writer = csv.writer(out_file, delimiter=delimiter)
        # write only new data
        for row in data:
            writer.writerow(row)


def read_file(file_path: Path | str) -> str:
    """Function to read in a file.

    Args:
        file_path: Path to file

    Returns:
        Read-in file
    """
    file = Path(file_path).read_text(encoding="utf-8")
    return file


def to_dict_with_standard_types(obj: Any) -> Any:
    """Convert dictionaries to dictionaries with python standard types only.

    Args:
        obj: Dictionary to convert

    Returns:
        Dictionary with standard types
    """
    match obj:
        case Path():
            return str(obj)
        case tuple():
            return [to_dict_with_standard_types(value) for value in obj]
        case list():
            return [to_dict_with_standard_types(value) for value in obj]
        case dict():
            for key, value in obj.items():
                obj[key] = to_dict_with_standard_types(value)
            return obj
        case _ if hasattr(obj, "tolist"):
            return obj.tolist()
        case _:
            return obj
