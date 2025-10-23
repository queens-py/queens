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
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def visualize_grid_and_surface(X1, X2, Z, min_point=None):
    """Visualize grid and surface with optional minimum point.

    Parameters
    ----------
    X1, X2 : 2D arrays
        Meshgrid arrays for x and y.
    Z : 2D array
        Function values f(X1, X2).
    min_point : tuple (x, y, z) or None
        If provided, highlights this point with a big red cross
        in both plots.
    """
    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "scene"}, {"type": "scene"}]],
        subplot_titles=("3D Scatter colored by z", "Surface + grid points (z=0 plane)"),
    )

    # --- Left: 3D scatter (share colorscale, no colorbar) ---
    fig.add_trace(
        go.Scatter3d(
            x=X1.flatten(),
            y=X2.flatten(),
            z=Z.flatten(),
            mode="markers",
            marker=dict(
                size=3,
                color=Z.flatten(),
                colorscale="Viridis",
                showscale=False,  # no duplicate colorbar
            ),
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    # --- Right: Surface (with single colorbar) ---
    fig.add_trace(
        go.Surface(
            x=X1,
            y=X2,
            z=Z,
            colorscale="Viridis",
            colorbar=dict(
                title="f(x1,x2)",
                x=1.05,  # push colorbar outside plot area
                len=0.75,  # shorten to avoid overlapping legend
            ),
        ),
        row=1,
        col=2,
    )

    # --- Right: black grid points on x-y plane (z=0) ---
    fig.add_trace(
        go.Scatter3d(
            x=X1.flatten(),
            y=X2.flatten(),
            z=np.zeros_like(Z).flatten(),
            mode="markers",
            marker=dict(size=2, color="black"),
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    # --- Highlight user-provided minimum point ---
    if min_point is not None:
        min_x, min_y, min_z = min_point
        for col in [1, 2]:
            fig.add_trace(
                go.Scatter3d(
                    x=[min_x],
                    y=[min_y],
                    z=[min_z],
                    mode="markers",
                    marker=dict(size=5, color="red"),
                    name="Minimum",
                ),
                row=1,
                col=col,
            )

    # Layout / axes
    fig.update_layout(
        title="3D Scatter vs. Surface with Grid Projection",
        height=500,
        legend=dict(
            x=0.95, y=0.9, bgcolor="rgba(255,255,255,0.6)"  # move legend away from colorbar
        ),
        scene=dict(xaxis_title="x1", yaxis_title="x2", zaxis_title="f(x1,x2)", aspectmode="cube"),
        scene2=dict(xaxis_title="x1", yaxis_title="x2", zaxis_title="f(x1,x2)", aspectmode="cube"),
    )

    fig.show()
