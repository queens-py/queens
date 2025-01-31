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
"""Distributions.

Sample from and compute probabilities for a variety of probability
distributions.
"""

from queens.distributions.bernoulli import BernoulliDistribution
from queens.distributions.beta import BetaDistribution
from queens.distributions.categorical import CategoricalDistribution
from queens.distributions.exponential import ExponentialDistribution
from queens.distributions.free import FreeVariable
from queens.distributions.lognormal import LogNormalDistribution
from queens.distributions.mean_field_normal import MeanFieldNormalDistribution
from queens.distributions.multinomial import MultinomialDistribution
from queens.distributions.normal import NormalDistribution
from queens.distributions.particles import ParticleDiscreteDistribution
from queens.distributions.uniform import UniformDistribution
from queens.distributions.uniform_discrete import UniformDiscreteDistribution

VALID_TYPES = {
    "normal": NormalDistribution,
    "mean_field_normal": MeanFieldNormalDistribution,
    "uniform": UniformDistribution,
    "lognormal": LogNormalDistribution,
    "beta": BetaDistribution,
    "exponential": ExponentialDistribution,
    "free": FreeVariable,
    "categorical": CategoricalDistribution,
    "bernoulli": BernoulliDistribution,
    "multinomial": MultinomialDistribution,
    "particles": ParticleDiscreteDistribution,
    "uniform_discrete": UniformDiscreteDistribution,
}
