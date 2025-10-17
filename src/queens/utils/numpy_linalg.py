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
"""Numpy linear algebra utils."""

import logging

import numpy as np

_logger = logging.getLogger(__name__)


def safe_cholesky(matrix: np.ndarray, jitter_start_value: np.generic | float = 1e-10) -> np.ndarray:
    """Numerically stable Cholesky decomposition.

    Compute the Cholesky decomposition of a matrix. Numeric stability is increased by
    sequentially adding a small term to the diagonal of the matrix.

    Args:
        matrix: Matrix to be decomposed
        jitter_start_value: Starting value to be added to the diagonal

    Returns:
        Lower-triangular Cholesky factor of matrix
    """
    try:
        low_cholesky = np.linalg.cholesky(matrix)
        return low_cholesky
    except np.linalg.LinAlgError as linalg_error:
        for i in range(5):
            jitter = jitter_start_value * 10**i
            matrix_ = matrix + np.eye(matrix.shape[0]) * jitter
            _logger.warning(
                "Added %.2e to diagonal of matrix for numerical stability "
                "of cholesky decompostition",
                jitter,
            )
            try:
                low_cholesky = np.linalg.cholesky(matrix_)
                return low_cholesky
            except np.linalg.LinAlgError:
                continue
        raise np.linalg.LinAlgError(
            "Cholesky decomposition failed due to ill-conditioning!"
        ) from linalg_error


def add_nugget_to_diagonal(matrix: np.ndarray, nugget_value: np.generic | float) -> np.ndarray:
    """Add a small value to diagonal of matrix.

    The nugget value is only added to diagonal entries that are smaller than the nugget value.

    Args:
        matrix: Matrix
        nugget_value: Small nugget value to be added

    Returns:
        Manipulated matrix
    """
    nugget_diag = np.where(np.diag(matrix) < nugget_value, nugget_value, 0)
    matrix += np.diag(nugget_diag)
    return matrix
