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
"""2D paraboloid."""


def paraboloid(x1, x2):
    """A paraboloid.

    See  https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html

    Args:
        x1 (float):  Input parameter 1
        x2 (float):  Input parameter 2

    Returns:
        float: Value of the function
    """
    a = 1.0
    b = 2.5
    return (x1 - a) ** 2 + (x2 - b) ** 2
