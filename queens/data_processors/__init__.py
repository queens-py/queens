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

from queens.data_processors.csv import Csv
from queens.data_processors.ensight import Ensight
from queens.data_processors.ensight_interface import EnsightInterfaceDiscrepancy
from queens.data_processors.numpy import Numpy
from queens.data_processors.pvd import Pvd
from queens.data_processors.txt import Txt

VALID_TYPES = {
    "csv": Csv,
    "ensight": Ensight,
    "ensight_interface_discrepancy": EnsightInterfaceDiscrepancy,
    "numpy": Numpy,
    "pvd": Pvd,
    "txt": Txt,
}
