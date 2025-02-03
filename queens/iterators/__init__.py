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
"""Iterators.

Modules for parameter studies, uncertainty quantification, sensitivity
analysis, Bayesian inverse analysis, and optimization.
"""

from queens.iterators.black_box_variational_bayes import BBVIIterator
from queens.iterators.bmfia_iterator import BMFIAIterator
from queens.iterators.bmfmc_iterator import BMFMCIterator
from queens.iterators.classification import ClassificationIterator
from queens.iterators.data_iterator import DataIterator
from queens.iterators.elementary_effects_iterator import ElementaryEffectsIterator
from queens.iterators.grid_iterator import GridIterator
from queens.iterators.hmc_iterator import HMCIterator
from queens.iterators.lhs_iterator import LHSIterator
from queens.iterators.lm_iterator import LMIterator
from queens.iterators.metropolis_hastings_iterator import MetropolisHastingsIterator
from queens.iterators.metropolis_hastings_pymc_iterator import MetropolisHastingsPyMCIterator
from queens.iterators.monte_carlo_iterator import MonteCarloIterator
from queens.iterators.nuts_iterator import NUTSIterator
from queens.iterators.optimization_iterator import OptimizationIterator
from queens.iterators.points_iterator import PointsIterator
from queens.iterators.polynomial_chaos_iterator import PolynomialChaosIterator
from queens.iterators.reparameteriztion_based_variational_inference import RPVIIterator
from queens.iterators.sequential_monte_carlo_chopin import SequentialMonteCarloChopinIterator
from queens.iterators.sequential_monte_carlo_iterator import SequentialMonteCarloIterator
from queens.iterators.sobol_index_gp_uncertainty_iterator import SobolIndexGPUncertaintyIterator
from queens.iterators.sobol_index_iterator import SobolIndexIterator
from queens.iterators.sobol_sequence_iterator import SobolSequenceIterator

VALID_TYPES = {
    "hmc": HMCIterator,
    "lhs": LHSIterator,
    "metropolis_hastings": MetropolisHastingsIterator,
    "metropolis_hastings_pymc": MetropolisHastingsPyMCIterator,
    "monte_carlo": MonteCarloIterator,
    "nuts": NUTSIterator,
    "optimization": OptimizationIterator,
    "read_data_from_file": DataIterator,
    "elementary_effects": ElementaryEffectsIterator,
    "polynomial_chaos": PolynomialChaosIterator,
    "sobol_indices": SobolIndexIterator,
    "sobol_indices_gp_uncertainty": SobolIndexGPUncertaintyIterator,
    "smc": SequentialMonteCarloIterator,
    "smc_chopin": SequentialMonteCarloChopinIterator,
    "sobol_sequence": SobolSequenceIterator,
    "points": PointsIterator,
    "bmfmc": BMFMCIterator,
    "grid": GridIterator,
    "lm": LMIterator,
    "bbvi": BBVIIterator,
    "bmfia": BMFIAIterator,
    "rpvi": RPVIIterator,
    "classification": ClassificationIterator,
}
