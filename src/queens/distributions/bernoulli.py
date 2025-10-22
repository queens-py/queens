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
"""Bernoulli distribution."""

import numpy as np

from queens.distributions.particle import Particle
from queens.utils.logger_settings import log_init_args


class Bernoulli(Particle):
    """Bernoulli distribution."""

    @log_init_args
    def __init__(self, success_probability: float) -> None:
        """Initialize Bernoulli distribution.

        Args:
            success_probability: Probability of sampling 1
        """
        if success_probability <= 0 or success_probability >= 1:
            raise ValueError(
                "The success probability has to be 0<success_probability<1. You specified "
                f"success_probability={success_probability}"
            )
        sample_space = np.array([0, 1]).reshape(-1, 1)
        probabilities = np.array([1 - success_probability, success_probability])
        super().check_duplicates_in_sample_space(sample_space)
        super().__init__(probabilities, sample_space)

        self.success_probability = success_probability
