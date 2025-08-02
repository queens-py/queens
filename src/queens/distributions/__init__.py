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

Modules for probability distributions.
"""

from queens.distributions.bernoulli import Bernoulli
from queens.distributions.beta import Beta
from queens.distributions.categorical import Categorical
from queens.distributions.exponential import Exponential
from queens.distributions.free_variable import FreeVariable
from queens.distributions.lognormal import LogNormal
from queens.distributions.mean_field_normal import MeanFieldNormal
from queens.distributions.multinomial import Multinomial
from queens.distributions.normal import Normal
from queens.distributions.particle import Particle
from queens.distributions.uniform import Uniform
from queens.distributions.uniform_discrete import UniformDiscrete

VALID_TYPES = {
    "normal": Normal,
    "mean_field_normal": MeanFieldNormal,
    "uniform": Uniform,
    "lognormal": LogNormal,
    "beta": Beta,
    "exponential": Exponential,
    "free": FreeVariable,
    "categorical": Categorical,
    "bernoulli": Bernoulli,
    "multinomial": Multinomial,
    "particles": Particle,
    "uniform_discrete": UniformDiscrete,
}
