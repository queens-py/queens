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
"""Plotting functions for the Gaussian Neural Network."""

from pathlib import Path

import matplotlib.pyplot as plt


def plot_loss(history, loss_plot_path):
    """Plot the loss function over the training epochs.

    Args:
        history (obj): Tensorflow history object of the training routine
        loss_plot_path (str): Path to save the loss plot
    """
    _, ax = plt.subplots()
    ax.plot(history.history["loss"])
    ax.set_ylabel("-log lik.")
    ax.set_xlabel("# epochs")
    plt.savefig(Path(loss_plot_path, "loss_plot.jpg"), dpi=300)
