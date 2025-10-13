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
"""Numpy array utils."""

import numpy as np


def at_least_2d(arr: np.ndarray) -> np.ndarray:
    """View input array as array with at least two dimensions.

    Args:
        arr: Input array

    Returns:
        View of input array with at least two dimensions
    """
    if arr.ndim == 0:
        return arr.reshape((1, 1))
    if arr.ndim == 1:
        return arr[:, np.newaxis]
    return arr


def at_least_3d(arr: np.ndarray) -> np.ndarray:
    """View input array as array with at least three dimensions.

    Args:
        arr: Input array

    Returns:
        View of input array with at least three dimensions
    """
    if arr.ndim == 0:
        return arr.reshape((1, 1, 1))
    if arr.ndim == 1:
        return arr[:, np.newaxis, np.newaxis]
    if arr.ndim == 2:
        return arr[:, :, np.newaxis]
    return arr


def extract_block_diag(array: np.ndarray, block_size: int) -> np.ndarray:
    """Extract block diagonals of square 2D Array.

    Args:
        array: Square 2D array
        block_size: Block size

    Returns:
        3D Array containing block diagonals
    """
    n_blocks = array.shape[0] // block_size

    new_shape = (n_blocks, block_size, block_size)
    new_strides = (
        block_size * array.strides[0] + block_size * array.strides[1],
        array.strides[0],
        array.strides[1],
    )

    return np.lib.stride_tricks.as_strided(array, new_shape, new_strides)
