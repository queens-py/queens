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
"""Data Processor.

Modules for extracting and processing data from simulation output files.
"""

from queens.data_processors.csv_file import CsvFile
from queens.data_processors.ensight_file import EnsightFile
from queens.data_processors.numpy_file import NumpyFile
from queens.data_processors.pvd_file import PvdFile
from queens.data_processors.txt_file import TxtFile

VALID_TYPES = {
    "csv": CsvFile,
    "ensight": EnsightFile,
    "numpy": NumpyFile,
    "pvd": PvdFile,
    "txt": TxtFile,
}
