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
"""Plotting functions for tutorial 4."""
import matplotlib.pyplot as plt
import mpl_toolkits
import numpy as np
import pyvista as pv
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d.axes3d import Axes3D


def plot_field(file_path: str, field: np.ndarray, ax: Axes3D, color_bar_title: str) -> None:
    """Plot cell-wise field values on the surface of a mesh.

    Args:
        file_path: Path to the mesh file readable by PyVista.
        field: Cell data array of shape ``(num_cells,)``.
        ax: 3D Matplotlib axis to plot into.
        color_bar_title: Label shown next to the color bar.
    """
    mesh = pv.read(file_path)[0][0]
    mesh.cell_data["field"] = field
    cell_values = field

    faces = mesh.extract_surface().faces.reshape((-1, 5))[:, 1:]
    vertices = mesh.points

    cmap = plt.get_cmap("viridis")
    norm = plt.Normalize(vmin=cell_values.min(), vmax=cell_values.max())
    colors = [cmap(norm(val)) for val in cell_values]

    poly3d = [[vertices[idx] for idx in face] for face in faces]
    collection = Poly3DCollection(poly3d, facecolors=colors, edgecolor="k", linewidths=0.2)
    ax.add_collection3d(collection)

    ax.auto_scale_xyz(vertices[:, 0], vertices[:, 1], vertices[:, 2])
    ax.set_axis_off()

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, fraction=0.02, pad=0.04, label=color_bar_title)

    ax.view_init(elev=90, azim=-90)
