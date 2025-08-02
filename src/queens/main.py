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
"""QUEENS main.

Main module of QUEENS containing the high-level control routine for
input file workflow.
"""

import logging
import sys
import time

from queens.global_settings import GlobalSettings
from queens.utils.cli import get_cli_options, print_greeting_message
from queens.utils.from_config_create import from_config_create_iterator
from queens.utils.io import load_input_file

_logger = logging.getLogger(__name__)


def run(input_file, output_dir, debug=False):
    """Do a QUEENS run.

    Args:
        input_file (Path): Path object to the input file
        output_dir (Path): Path object to the output directory
        debug (bool): True if debug mode is to be used
    """
    start_time_input = time.time()

    # read input and create config
    config = load_input_file(input_file)

    experiment_name = config.pop("experiment_name")

    with GlobalSettings(
        experiment_name=experiment_name,
        output_dir=output_dir,
        debug=debug,
    ) as global_settings:

        # Inform user on new script approach
        # Do not remove this from inside the context, otherwise logging will be missing.
        _logger.warning(
            "You started QUEENS from an input file. Consider using QUEENS as a library within a "
            "Python script. You can generate a Python script from this input file using"
            "\nqueens-input-to-script --input %s --output_dir %s --script_path queens_script.py\n",
            input_file,
            output_dir,
        )

        # Create iterator
        my_iterator = from_config_create_iterator(config, global_settings)

        end_time_input = time.time()

        _logger.info("")
        _logger.info("Time for INPUT: %s s", end_time_input - start_time_input)
        _logger.info("")

        # Perform analysis
        run_iterator(my_iterator, global_settings)


def run_iterator(iterator, global_settings):
    """Run the main queens iterator.

    Args:
        iterator (Iterator): Main queens iterator
        global_settings (GlobalSettings): settings of the QUEENS experiment including its name
                                  and the output directory
    """
    global_settings.print_git_information()

    start_time_calc = time.time()

    _logger.info(
        "%s for experiment: %s", iterator.__class__.__name__, global_settings.experiment_name
    )
    _logger.info("")
    _logger.info("Starting Analysis...")
    _logger.info("")

    try:
        iterator.run()
    except Exception as exception:
        _logger.exception(exception)
        global_settings.__exit__(None, None, None)
        # TODO: Write iterator in pickle file # pylint: disable=fixme
        raise exception

    end_time_calc = time.time()
    _logger.info("")
    _logger.info("Time for CALCULATION: %s s", end_time_calc - start_time_calc)
    _logger.info("")


def main():
    """Main function."""
    # The first argument is the file name
    args = sys.argv[1:]

    if len(args) > 0:
        # Do QUEENS run
        input_file_path, output_dir, debug = get_cli_options(args)
        run(input_file_path, output_dir, debug)
    else:
        # Print some infos
        print_greeting_message()


if __name__ == "__main__":
    main()
