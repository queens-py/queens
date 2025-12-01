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
"""Utilities for type hinting."""

from typing import Literal, TypeAlias

import numpy as np

# pylint: disable=invalid-name

ArrayN: TypeAlias = np.ndarray[tuple[int], np.dtype[np.floating]]
Array1xN: TypeAlias = np.ndarray[tuple[Literal[1], int], np.dtype[np.floating]]
ArrayNx1: TypeAlias = np.ndarray[tuple[int, Literal[1]], np.dtype[np.floating]]
ArrayNxM: TypeAlias = np.ndarray[tuple[int, int], np.dtype[np.floating]]

Array1D: TypeAlias = ArrayN | Array1xN | ArrayNx1
