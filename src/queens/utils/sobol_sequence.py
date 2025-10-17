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
"""Collection of utility functions and classes for Sobol sequences."""

import numpy as np
from scipy.stats.qmc import Sobol

from queens.parameters.parameters import Parameters


def sample_sobol_sequence(
    dimension: int,
    number_of_samples: int,
    parameters: Parameters,
    randomize: bool = False,
    seed: int | np.random.Generator | None = None,
) -> np.ndarray:
    """Generate samples from Sobol sequence.

    Args:
        dimension: Dimensionality of the sequence. Max dimensionality is 21201.
        number_of_samples: Number of samples to generate in the parameter space
        parameters: Parameters object defining the true distribution of the samples
        randomize: If True, use LMS+shift scrambling, i.e. randomize the sequence. Otherwise, no
            scrambling is done.
        seed: If `seed` is an int or None, a new `numpy.random.Generator` is created using
            ``np.random.default_rng(seed)``. If `seed` is already a ``Generator`` instance, then the
            provided instance is used.

    Returns:
        Sobol sequence quasi Monte Carlo samples for the parameter distribution
    """
    # create uniform quasi Monte Carlo (qmc) samples
    sobol_engine = Sobol(d=dimension, scramble=randomize, seed=seed)

    qmc_samples = sobol_engine.random(n=number_of_samples)
    # scale and transform samples according to the inverse cdf such that they follow their "true"
    # distributions
    samples = parameters.inverse_cdf_transform(qmc_samples)

    return samples
