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
"""Module for handling external geometry objects."""

import abc
import logging

_logger = logging.getLogger(__name__)


class ExternalGeometry(metaclass=abc.ABCMeta):
    """Abstract base class to read in *external_geometry_obj* formats.

    The class enables, e.g., *external_geometry_obj* based construction
    of random fields or post processing routines.

    Returns:
        geometry_obj (obj): Instance of the ExternalGeometry class
    """

    def __init__(self):
        """Initialize ExternalGeometry."""

    def main_run(self):
        """Main routine of *external_geometry_obj* object."""
        _logger.info("Start reading external geometry from file...")
        self.organize_sections()
        self.read_external_data()
        self.finish_and_clean()
        _logger.info("Finished reading external geometry from file!")

    @abc.abstractmethod
    def read_external_data(self):
        """Method that reads in external files.

        Method that reads in external files containing an
        *external_geometry_obj* definition.
        """

    @abc.abstractmethod
    def organize_sections(self):
        """Organizes (geometric) sections.

        Organizes (geometric) sections in external file to read in
        geometric data efficiently.
        """

    @abc.abstractmethod
    def finish_and_clean(self):
        """Finishing, postprocessing and cleaning."""
