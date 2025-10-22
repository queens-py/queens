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
"""Discrete uniform distribution."""

import numpy as np

from queens.distributions.particle import Particle
from queens.utils.logger_settings import log_init_args


class UniformDiscrete(Particle):
    """Discrete uniform distribution."""

    @log_init_args
    def __init__(self, sample_space: np.ndarray) -> None:
        """Initialize discrete uniform distribution.

        Args:
            sample_space: Samples, i.e., possible outcomes of sampling the distribution
        """
        probabilities = np.ones(len(sample_space)) / len(sample_space)
        super().check_duplicates_in_sample_space(sample_space)
        super().__init__(probabilities, sample_space)
