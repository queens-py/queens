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
"""Data processor class for txt data extraction."""

import logging
import re
from pathlib import Path

from queens.data_processors._data_processor import DataProcessor
from queens.utils.logger_settings import log_init_args

_logger = logging.getLogger(__name__)


class TxtFile(DataProcessor):
    """Class for extracting data from txt files.

    Provides basic functionality for extracting data from txt files,
    however the final implementation is up to the user.

    Attributes:
        The implementation of the filter_and_manipulate_raw_data method is up to the user.

    Throws:
        MemoryError: We throw a conservative MemoryError if the txt file is larger than 200 MB.
        This is due to the current design, which loads the entire content of the
        .txt file into memory.

    Potential Improvement:
        Use a generator for reading the content of the file in chunks.
        This however requires a more advanced logic with the possibility to nest functions
        calls in a flexible way.
    """

    @log_init_args
    def __init__(
        self,
        file_name_identifier=None,
        file_options_dict=None,
        files_to_be_deleted_regex_lst=None,
        remove_logger_prefix_from_raw_data=True,
        logger_prefix=r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3} - "
        r"queens\.drivers\.driver_\d* - INFO -",
        max_file_size_in_mega_byte=200,
    ):
        """Instantiate data processor class for txt data.

        Args:
            file_name_identifier (str):             Identifier of file name.
                                                    The file prefix can contain a regex expression
                                                    and subdirectories.
            file_options_dict (dict):               Dictionary with read-in options for the file:
            files_to_be_deleted_regex_lst (lst):    List with paths to files that should be deleted.
                                                    The paths can contain regex expressions.
            remove_logger_prefix_from_raw_data(bool):   Defaults to True. Removes the logger_prefix
                                                        from the raw_data during the reading of
                                                        the file.
            logger_prefix (str):                    A string or regular expressions that precedes
                                                    each line of the queens log file.
            max_file_size_in_mega_byte (int):       Upper limit of the file size to be read into
                                                    memory in megabyte (MB). See comment above on
                                                    Potential Improvement.
        """
        super().__init__(
            file_name_identifier=file_name_identifier,
            file_options_dict=file_options_dict,
            files_to_be_deleted_regex_lst=files_to_be_deleted_regex_lst,
        )
        self.remove_logger_prefix_from_raw_data = remove_logger_prefix_from_raw_data
        self.logger_prefix = logger_prefix
        self.max_file_size_in_mega_byte = max_file_size_in_mega_byte

    def get_raw_data_from_file(self, file_path):
        """Load the text file into memory.

        Args:
            file_path (str): Actual path to the file of interest.

        Returns:
            raw_data (lst): A list of strings read in from file_path.
        """
        raw_data = []
        try:
            self._check_file_size(file_path)
            with open(file_path, "r", encoding="utf-8") as file:
                if self.remove_logger_prefix_from_raw_data:
                    for line in file:
                        match = re.search(self.logger_prefix, line)
                        extracted_part = line[match.end() :]
                        extracted_part = extracted_part.lstrip().rstrip()
                        raw_data.append(extracted_part)
                else:
                    raw_data = file.readlines()
                return raw_data
        except IOError as error:
            # pylint: disable=duplicate-code
            _logger.warning(
                "Could not read the file: %s. The following IOError was raised: %s. "
                "Skipping the file and continuing.",
                file_path,
                error,
            )
            return None

    def _check_file_size(self, file_path):
        """Check the file size of the input file.

        Args:
            file_path (str): Path to the input file.

        Throws:
            Memory error if the file size is larger than max_file_size_in_mega_byte.
        """
        path = Path(file_path)
        # Get the size of the file in bytes
        file_size = path.stat().st_size
        if file_size > self.max_file_size_in_mega_byte * 1024 * 1024:
            raise MemoryError(
                f"Maximum allowed file size of {self.max_file_size_in_mega_byte} MB "
                f"exceeded for file %s.",
                file_path,
            )

    def _extract_section_from_raw_data(self, raw_data, marker_type, regex_start="", regex_end=""):
        """Divide the raw data into sections.

        Extracts a section of text from the given raw data based on regular
        expressions. Calls the appropriate subroutine depending on the
        marker_type.

        Args:
            raw_data (str):                 The raw data from which a section is
                                            to be extracted.
            marker_type (str):              The type of marker indicating how the
                                            section should be extracted.
            regex_start (str, optional):    The regular expression pattern
                                            indicating the start of the section.
                                            Required when marker_type is 'start'
                                            or 'start_end'.
            regex_end (str, optional):      The regular expression pattern indicating
                                            the end of the section.
                                            Required when marker_type is 'end' or 'start_end'.

        Returns:
            raw_section_data (lst):        A dictionary where the key is an incremental
                                            counter and the corresponding value is the
                                            extracted section of text (str).
        """
        if marker_type == "start":
            if regex_start == "":
                raise ValueError("regex_start must be set when marker_type is 'start'")
            raw_section_data = self._extract_section_with_start_marker(raw_data, regex_start)

        elif marker_type == "end":
            if regex_end == "":
                raise ValueError("regex_end must be set when marker_type is 'end'")
            raw_section_data = self._extract_section_with_end_marker(raw_data, regex_end)

        elif marker_type == "start_end":
            if regex_end == "" or regex_start == "":
                raise ValueError(
                    "regex_start and regex_end must be set when marker_type is 'start_end'"
                )
            raw_section_data = self._extract_section_with_start_and_end_marker(
                raw_data, regex_start, regex_end
            )
        else:
            raise ValueError(f"Unrecognised marker_type: '{marker_type}'")

        return raw_section_data

    @staticmethod
    def _extract_section_with_start_marker(raw_data, regex):
        """Subroutine for a marked start of the section.

        Extracts a section of text, where the start of the section is marked by a regex.

        Args:
            raw_data (str): The raw data from which a section is to be extracted.
            regex (str):    The regular expression pattern indicating the start of the section.

        Returns:
            raw_section_data (lst):    A list where each element is the extracted section
                                        of text (list of strings).
        """
        raw_section_data = []
        current_section = []
        initial = True
        for line in raw_data:
            if re.search(regex, line):
                if not initial:
                    raw_section_data.append(current_section)
                initial = False
                current_section = [line]
            elif current_section:
                current_section.append(line)

        # Append the last section
        raw_section_data.append(current_section)
        return raw_section_data

    @staticmethod
    def _extract_section_with_end_marker(raw_data, regex):
        """Subroutine for a marked end of the section.

        Extracts a section of text, where the end of the section is marked by a regex.

        Args:
            raw_data (str): The raw data from which a section is to be extracted.
            regex (str):    The regular expression pattern indicating the end of the section.


        Returns:
            raw_section_data (lst):    A list where each element is the extracted section
                                        of text (list of strings).
        """
        raw_section_data = []
        current_section = []

        for line in raw_data:
            current_section.append(line)

            if re.search(regex, line):
                raw_section_data.append(current_section)
                current_section = []

        # Append the last section
        raw_section_data.append(current_section)
        return raw_section_data

    @staticmethod
    def _extract_section_with_start_and_end_marker(raw_data, regex_start, regex_end):
        """Subroutine for a marked start and end of the section.

        Extracts a section of text, where the start and the end of the section is marked
        by a regex each.

        Args:
            raw_data (str):     The raw data from which a section is to be extracted.
            regex_start (str):  The regular expression pattern indicating the
                                start of the section.
            regex_end (str):    The regular expression pattern indicating the end
                                of the section.


        Returns:
            raw_section_data (lst):    A list where each element is the extracted section
                                        of text (list of strings).
        """
        raw_section_data = []
        current_section = []
        for line in raw_data:
            if re.search(regex_start, line):
                current_section = [line]
            elif current_section and re.search(regex_end, line):
                current_section.append(line)
                raw_section_data.append(current_section)
                current_section = []
            elif current_section:
                current_section.append(line)
        return raw_section_data

    @staticmethod
    def _extract_lines_with_regex(section, regex):
        """Extracts lines from a section.

        Find and extract lines in the given section that match the specified
        regular expression.

        Args:
            section (lst):     List of lines (str) to search through.
            regex (str):        Regular expression pattern to match lines.

        Returns:
            matches (lst):     A list containing tuples of matches
                                (line number, line content).
        """
        matches = []

        # Iterate over each line in the section
        for line_number, line in enumerate(section, start=1):
            # Check if the line matches the specified regex
            if re.search(regex, line):
                # Add the line to the result dictionary with adjusted line number
                matches.append((line_number, line))

        return matches

    @staticmethod
    def _extract_quantities_from_line(line, regexp):
        """Extract quantities from a given line using a regular expression.

        Args:
            line (str):     The input string from which quantities are to be extracted.
            regexp (str):   The regular expression pattern used to identify
                            quantities in the line.

        Returns:
            matches(lst):   A list containing matched quantities found in the line.
        """
        matches = re.findall(regexp, line)
        return matches
