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
"""Random fields.

Modules for random fields.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from queens.utils.imports import extract_type_checking_imports, import_class_from_class_module_map

if TYPE_CHECKING:
    from queens.parameters.random_fields._random_field import RandomField
    from queens.parameters.random_fields.fourier import Fourier
    from queens.parameters.random_fields.karhunen_loeve import KarhunenLoeve
    from queens.parameters.random_fields.piece_wise import PieceWise


class_module_map = extract_type_checking_imports(__file__)


def __getattr__(name: str) -> type[RandomField]:
    return import_class_from_class_module_map(name, class_module_map, __name__)
