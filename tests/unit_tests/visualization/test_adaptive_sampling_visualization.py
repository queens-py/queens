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
"""Unit tests for adaptive sampling visualization."""

import numpy as np
import pytest

from queens.distributions.uniform import Uniform
from queens.parameters.parameters import Parameters
from queens.visualization.adaptive_sampling_visualization import AdaptiveSamplingVisualization


def _parameters(num_parameters):
    """Create scalar parameters."""
    return Parameters(
        **{f"x{i}": Uniform(lower_bound=-3, upper_bound=3) for i in range(num_parameters)}
    )


def _adaptive_sampling_results(num_parameters):
    """Create adaptive sampling-like results of one iteration."""
    generator = np.random.default_rng(41)
    particles = generator.normal(size=(30, num_parameters))
    x_train = generator.normal(size=(8, num_parameters))
    return {
        "particles": [particles],
        "weights": [np.full(particles.shape[0], 1.0 / particles.shape[0])],
        "x_train": [x_train],
        "x_train_new": [generator.normal(size=(3, num_parameters))],
        "x_train_failed": [generator.normal(size=(2, num_parameters))],
        "y_train": [generator.normal(size=(x_train.shape[0], 1))],
    }


def _plot(results, num_parameters, global_settings, **kwargs):
    """Plot with coarse grids for fast tests."""
    visualization = AdaptiveSamplingVisualization(
        kde_grid_size=8, kde_num_points=20, contour_levels=4, **kwargs
    )
    visualization.plot(
        results,
        iteration=0,
        parameters=_parameters(num_parameters),
        global_settings=global_settings,
    )


def _plot_file(global_settings, output_format="svg"):
    """Return the path of the plot of iteration 0."""
    return global_settings.result_file(suffix="_posterior_iteration_0", extension=output_format)


@pytest.mark.parametrize("output_format", ["svg", "png"])
@pytest.mark.parametrize("num_parameters", [1, 2, 3])
@pytest.mark.parametrize("layout", ["pair_grid", "1d_marginals"])
def test_plot_saves_posterior(global_settings, layout, num_parameters, output_format):
    """Smoke test that the posterior plot is saved."""
    results = _adaptive_sampling_results(num_parameters)

    _plot(results, num_parameters, global_settings, layout=layout, output_format=output_format)

    # A failing plot is only logged, so check that the file was written.
    assert _plot_file(global_settings, output_format).exists()


def test_plot_saves_posterior_of_constant_particles(global_settings):
    """Smoke test that (nearly) constant particles do not prevent the plot."""
    results = _adaptive_sampling_results(num_parameters=2)
    results["particles"][0][:, 1] *= 1e-6

    _plot(results, 2, global_settings)

    assert _plot_file(global_settings).exists()


def test_failing_plot_logs_warning(global_settings, mocker, caplog):
    """Test that a failing plot is logged instead of raised."""
    mocker.patch.object(
        AdaptiveSamplingVisualization,
        "posterior_figure",
        side_effect=RuntimeError("plot failed"),
    )

    with caplog.at_level("WARNING"):
        _plot(_adaptive_sampling_results(num_parameters=2), 2, global_settings)

    assert "Plotting iteration 0 failed" in caplog.text
    assert "plot failed" in caplog.text
