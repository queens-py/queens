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
from skfem import Basis, BilinearForm, ElementTriP1, LinearForm, MeshTri, enforce, solve
from skfem.helpers import dot, grad

mesh = MeshTri().refined(6)


def poisson_pde(source_x, source_y, source_term):

    # Set discretization
    e = ElementTriP1()
    basis = Basis(mesh, e)

    @BilinearForm
    def laplace(u, v, _):
        return dot(grad(u), grad(v))

    @LinearForm
    def rhs(v, w):
        # Source term
        return source_term(w.x[0], w.x[1], source_x, source_y) * v

    # Stiffness matrix
    A = laplace.assemble(basis)

    # Right-hand side
    b = rhs.assemble(basis)

    # Enforce Dirichlet boundary conditions
    A, b = enforce(A, b, D=mesh.boundary_nodes())

    # Solve
    solution = solve(A, b)

    return solution


def plot_to_axis(field, ax):
    mesh.plot(field, ax=ax)
    ax.axis("equal")
    ax.set_aspect("equal", "box")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
