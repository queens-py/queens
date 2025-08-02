# -*- coding: utf-8 -*-
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
"""Drivers.

Modules for calling external simulation software.
"""

from queens.drivers.fourc import Fourc
from queens.drivers.function import Function
from queens.drivers.jobscript import Jobscript
from queens.drivers.mpi import Mpi

VALID_TYPES = {
    "fourc": Fourc,
    "mpi": Mpi,
    "jobscript": Jobscript,
    "function": Function,
}
