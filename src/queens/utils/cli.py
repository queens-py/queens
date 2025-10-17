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
"""Command Line Interface utils collection."""

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Callable, Sequence

from queens.utils import ascii_art
from queens.utils.exceptions import CLIError
from queens.utils.injector import inject
from queens.utils.io import print_pickled_data
from queens.utils.logger_settings import reset_logging, setup_cli_logging
from queens.utils.metadata import write_metadata_to_csv
from queens.utils.path import PATH_TO_ROOT
from queens.utils.printing import get_str_table
from queens.utils.run_subprocess import run_subprocess

_logger = logging.getLogger(__name__)


def cli_logging(func: Callable) -> Callable:
    """Decorator to create logger for CLI function.

    Args:
        func: Function that is to be decorated
    """

    def decorated_function(*args: Any, **kwargs: Any) -> Any:
        setup_cli_logging()
        results = func(*args, **kwargs)
        reset_logging()

        # For CLI commands there should be no results, but just in case
        return results

    return decorated_function


@cli_logging
def inject_template_cli() -> None:
    """Use the injector of QUEENS."""
    ascii_art.print_crown(80)
    ascii_art.print_banner("Injector", 80)
    parser = argparse.ArgumentParser(
        description="QUEENS injection CLI for Jinja2 templates. The parameters to be injected can "
        "be supplied by adding additional '--<name> <value>' arguments. All occurrences of <name> "
        "will be replaced with <value> in the template. Below, only two examples are shown, but an "
        "arbitrary number of parameters (name-value pairs) can be added."
    )
    parser.add_argument(
        "--template",
        type=str,
        required=True,
        help="Jinja2 template to be injected.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Output path for the injected template.",
    )

    # These two are dummy arguments to indicate how to use this CLI
    parser.add_argument(
        "--name_1",
        type=str,
        default=None,
        metavar="value_1",
        help="Example name-value pair: inject a parameter called <name_1> with the value <value_1>",
    )
    parser.add_argument(
        "--name_2",
        type=str,
        default=None,
        metavar="value_2",
        help="Example name-value pair: inject a parameter called <name_2> with the value <value_2>",
    )

    path_arguments, parameter_arguments = parser.parse_known_args()

    template_path = Path(path_arguments.template)
    if path_arguments.output_path is None:
        output_path = template_path.with_name(
            template_path.stem + "_injected" + template_path.suffix
        )
    else:
        output_path = Path(path_arguments.output_path)

    _logger.info("Template: %s", template_path.resolve())
    _logger.info("Output path: %s", template_path.resolve())
    _logger.info(" ")

    # Get injection parameters
    injection_parser = argparse.ArgumentParser()

    # Add input parameters to inject
    for arg in parameter_arguments:
        if arg.find("--") > -1:
            injection_parser.add_argument(arg)

    # Create the dictionary
    injection_dict = vars(injection_parser.parse_args(parameter_arguments))
    _logger.info(get_str_table("Injection parameters", injection_dict))
    inject(injection_dict, template_path, output_path)

    _logger.info("Injection done, created file %s", output_path)


@cli_logging
def print_pickle_data_cli() -> None:
    """Print pickle data wrapper."""
    ascii_art.print_crown(60)
    ascii_art.print_banner("QUEENS", 60)
    args = sys.argv[1:]
    if len(args) == 0:
        _logger.info("No pickle file was provided!")
    else:
        file_path = args[0]
        print_pickled_data(Path(file_path))


@cli_logging
def gather_metadata_and_write_to_csv() -> None:
    """Gather metadata and write them to csv."""
    ascii_art.print_crown(60)
    ascii_art.print_banner("QUEENS", 60)

    parser = argparse.ArgumentParser(
        description="QUEENS cli util to create csv file for experiment simulation metadata."
    )
    parser.add_argument(
        "--experiment_dir",
        type=str,
        help="Experiment dir to simulation folders",
    )
    parser.add_argument(
        "--csv_path", type=str, help="Path to export metadata csv file", default=None
    )

    args = sys.argv[1:]
    parsed_args = parser.parse_args(args)
    _logger.info("Gathering metadata and exporting to csv.")
    write_metadata_to_csv(
        experiment_dir=parsed_args.experiment_dir,
        csv_path=parsed_args.csv_path,
    )
    _logger.info("Done.")


def build_html_coverage_report() -> None:
    """Build html coverage report."""
    _logger.info("Build html coverage report...")

    pytest_command_string = (
        'pytest -m "unit_tests or integration_tests or integration_tests_fourc" '
        "--cov --cov-report=html:html_coverage_report"
    )
    command_list = ["cd", str(PATH_TO_ROOT), "&&", pytest_command_string]
    command_string = " ".join(command_list)
    run_subprocess(command_string)


def remove_html_coverage_report() -> None:
    """Remove html coverage report files."""
    _logger.info("Remove html coverage report...")

    pytest_command_string = "rm -r html_coverage_report/; rm .coverage*"
    command_list = ["cd", str(PATH_TO_ROOT), "&&", pytest_command_string]
    command_string = " ".join(command_list)
    run_subprocess(command_string)


def str_to_bool(value: str) -> bool:
    """Convert string to boolean for cli commands.

    Args:
        value: String to convert to a bool

    Returns:
        Bool of the string
    """
    if isinstance(value, bool):
        return value

    false_options = ("false", "f", "0", "no", "n")
    true_options = ("true", "t", "1", "yes", "y")
    if value.lower() in false_options:
        return False
    if value.lower() in true_options:
        return True
    raise CLIError(
        f"{value} is not a valid boolean value. Valid options are:\n"
        f"{', '.join(list(true_options+false_options))}"
    )


def get_cli_options(args: Sequence[str]) -> tuple[Path, Path, bool]:
    """Get input file path, output directory and debug from args.

    Args:
        args: cli arguments

    Returns:
        Path object to input file
        Path object to the output directory
        True if debug mode is to be used
    """
    parser = argparse.ArgumentParser(description="QUEENS")
    parser.add_argument(
        "--input", type=str, default=None, help="Input file in .json or .yaml/yml format."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory to write results to. The directory has to be created by the user!",
    )
    parser.add_argument("--debug", type=str_to_bool, default=False, help="Debug mode yes/no.")

    parsed_args = parser.parse_args(args)

    if parsed_args.input is None:
        raise CLIError("No input file was provided with option --input.")

    if parsed_args.output_dir is None:
        raise CLIError("No output directory was provided with option --output_dir.")

    debug = parsed_args.debug
    output_dir = Path(parsed_args.output_dir)
    input_file = Path(parsed_args.input)

    return input_file, output_dir, debug


@cli_logging
def print_greeting_message() -> None:
    """Print a greeting message and how to use QUEENS."""
    ascii_art.print_banner_and_description()
    ascii_art.print_centered_multiline("Welcome to the royal family!")
    _logger.info("\nTo use QUEENS run:\n")
    _logger.info("queens --input <inputfile> --output_dir <output_dir>\n")
    _logger.info("or\n")
    _logger.info("python -m queens.main --input <inputfile> --output_dir <output_dir>\n")
    _logger.info("or\n")
    _logger.info(
        "python path_to_queens/queens/main.py --input <inputfile> --output_dir <output_dir>\n"
    )
