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
from typing import TYPE_CHECKING

from queens.utils.imports import extract_type_checking_imports, import_class_from_class_module_map

if TYPE_CHECKING:
    from queens.drivers.jobscript import Jobscript
    from queens.drivers.mpi import Mpi


class_module_map = extract_type_checking_imports(__file__)


def __getattr__(name):
    return import_class_from_class_module_map(name, class_module_map, __name__)
