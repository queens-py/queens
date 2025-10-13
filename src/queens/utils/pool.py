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
"""Pool utils."""

import logging

from pathos.multiprocessing import ProcessingPool as Pool
from pathos.multiprocessing import ProcessPool

_logger = logging.getLogger(__name__)


def create_pool(number_of_workers: int) -> ProcessPool | None:
    """Create pathos Pool from number of workers.

    Args:
        number_of_workers: Number of parallel evaluations

    Returns:
        Pathos multiprocessing pool
    """
    if isinstance(number_of_workers, int) and number_of_workers > 1:
        _logger.info(
            "Activating parallel evaluation of samples with %s workers.\n", number_of_workers
        )
        pool = Pool(processes=number_of_workers)
    else:
        pool = None
    return pool
