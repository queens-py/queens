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
from __future__ import annotations

from typing import TYPE_CHECKING

from queens.utils.imports import extract_type_checking_imports, import_class_from_class_module_map

if TYPE_CHECKING:
    from queens.distributions._distribution import Continuous, Discrete, Distribution
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


class_module_map = extract_type_checking_imports(__file__)


def __getattr__(name: str) -> Distribution:
    return import_class_from_class_module_map(name, class_module_map, __name__)
