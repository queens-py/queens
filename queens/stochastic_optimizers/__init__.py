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
"""Stochastic optimizers.

Modules containing stochastic optimizers.
"""

from queens.stochastic_optimizers.adam import Adam
from queens.stochastic_optimizers.adamax import Adamax
from queens.stochastic_optimizers.rms_prop import RMSprop
from queens.stochastic_optimizers.sgd import SGD

VALID_TYPES = {"adam": Adam, "adamax": Adamax, "rms_prop": RMSprop, "sgd": SGD}
