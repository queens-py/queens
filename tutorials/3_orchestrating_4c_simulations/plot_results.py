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
"""Plotting functions for tutorial 3."""
import os

os.environ["PYVISTA_OFF_SCREEN"] = "true"
os.environ["VTK_DEFAULT_RENDER_WINDOW_OFFSCREEN"] = "1"
os.environ["PYVISTA_JUPYTER_BACKEND"] = "static"

import pyvista as pv

fe_mesh = pv.read("beam_coarse.exo")


def plot_results(
    result_file: str,
    plotter: pv.Plotter,
    color_bar_title: str = "zz-component of Cauchy stress tensor\n",
) -> None:
    """Plots results.

    Args:
        result_file (str): Path to pvtu file.
        plotter (pyvista Plotter): pyvista plotter.
        color_bar_title (string): title of the colorbar.
    """
    outputs = pv.read(result_file).warp_by_vector("displacement")
    plotter.add_mesh(fe_mesh.copy(), style="wireframe", color="blue")
    outputs["cauchy_zz"] = outputs["element_cauchy_stresses_xyz"][:, 2]
    plotter.add_mesh(
        outputs,
        scalars="cauchy_zz",
        scalar_bar_args={
            "title": color_bar_title,
            "title_font_size": 15,
            "label_font_size": 15,
        },
    )
    plotter.add_axes(line_width=5)  # type: ignore[call-arg]
    plotter.camera_position = [
        (1.1321899035097993, -6.851600196807601, 2.7649096132703574),
        (0.0, 0.0, 0.2749999999999999),
        (-0.97637930372977, -0.08995062285804697, 0.19644933366041056),
    ]
