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

from queens.iterators.bbvi import BBVI
from queens.iterators.bmfia import BMFIA
from queens.iterators.bmfmc import BMFMC
from queens.iterators.classification import ClassificationIterator
from queens.iterators.data import Data
from queens.iterators.elementary_effects import ElementaryEffects
from queens.iterators.grid import Grid
from queens.iterators.hamiltonian_monte_carlo import HamiltonianMonteCarlo
from queens.iterators.latin_hypercube_sampling import LatinHypercubeSampling
from queens.iterators.least_squares import LeastSquares
from queens.iterators.metropolis_hastings import MetropolisHastings
from queens.iterators.metropolis_hastings_pymc import MetropolisHastingsPyMC
from queens.iterators.monte_carlo import MonteCarlo
from queens.iterators.nuts import NUTS
from queens.iterators.optimization import Optimization
from queens.iterators.points import Points
from queens.iterators.polynomial_chaos import PolynomialChaos
from queens.iterators.reinforcement_learning import ReinforcementLearning
from queens.iterators.reparameteriztion_based_variational import RPVI
from queens.iterators.sequential_monte_carlo import SequentialMonteCarlo
from queens.iterators.sequential_monte_carlo_chopin import SequentialMonteCarloChopin
from queens.iterators.sobol_index import SobolIndex
from queens.iterators.sobol_index_gp_uncertainty import SobolIndexGPUncertainty
from queens.iterators.sobol_sequence import SobolSequence

VALID_TYPES = {
    "hmc": HamiltonianMonteCarlo,
    "lhs": LatinHypercubeSampling,
    "metropolis_hastings": MetropolisHastings,
    "metropolis_hastings_pymc": MetropolisHastingsPyMC,
    "monte_carlo": MonteCarlo,
    "nuts": NUTS,
    "optimization": Optimization,
    "read_data_from_file": Data,
    "elementary_effects": ElementaryEffects,
    "polynomial_chaos": PolynomialChaos,
    "sobol_indices": SobolIndex,
    "sobol_indices_gp_uncertainty": SobolIndexGPUncertainty,
    "smc": SequentialMonteCarlo,
    "smc_chopin": SequentialMonteCarloChopin,
    "sobol_sequence": SobolSequence,
    "points": Points,
    "bmfmc": BMFMC,
    "grid": Grid,
    "least_squares": LeastSquares,
    "bbvi": BBVI,
    "bmfia": BMFIA,
    "rpvi": RPVI,
    "classification": ClassificationIterator,
    "reinforcement_learning": ReinforcementLearning,
}
