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
"""Collection of plotting capabilities for probability distributions."""

import numpy as np
import plotly
import plotly.graph_objs as go


def plot_pdf(pdf_estimate: dict, support_points: np.ndarray, bayes: bool = False) -> None:
    """Create pdf plot based on passed data.

    Args:
        pdf_estimate: Estimate of pdf at supporting points
        support_points: Supporting points
        bayes: Do we want to plot confidence intervals
    """
    mean_pdf = go.Scatter(
        x=support_points, y=pdf_estimate["mean"], mode="markers+lines", name="Mean"
    )

    data = [mean_pdf]

    if bayes:
        q5_pdf = go.Scatter(
            x=support_points, y=pdf_estimate["q5"], mode="markers+lines", name="5% quantile"
        )
        data.append(q5_pdf)

        q95_pdf = go.Scatter(
            x=support_points, y=pdf_estimate["q95"], mode="markers+lines", name="95% quantile"
        )
        data.append(q95_pdf)

        median_pdf = go.Scatter(
            x=support_points, y=pdf_estimate["median"], mode="markers+lines", name="median"
        )
        data.append(median_pdf)

    layout = {
        "title": "Probability Density Function",
        "xaxis": {"title": "QOI"},
        "yaxis": {"title": "PDF"},
    }

    fig = go.Figure(data=data, layout=layout)
    plotly.offline.plot(fig, filename="PDF.html", auto_open=True)


def plot_cdf(cdf_estimate: dict, support_points: np.ndarray, bayes: bool = False) -> None:
    """Create cdf plot based on passed data.

    Args:
        cdf_estimate: Estimate of cdf at supporting points
        support_points: Supporting points
        bayes: Do we want to plot confidence intervals
    """
    # Create a trace
    mean_cdf = go.Scatter(
        x=support_points, y=cdf_estimate["mean"], mode="markers+lines", name="Mean"
    )
    data = [mean_cdf]

    if bayes:
        q5_cdf = go.Scatter(
            x=support_points, y=cdf_estimate["q5"], mode="markers+lines", name="5% quantile"
        )
        data.append(q5_cdf)

        q95_cdf = go.Scatter(
            x=support_points, y=cdf_estimate["q95"], mode="markers+lines", name="95% quantile"
        )
        data.append(q95_cdf)

        median_cdf = go.Scatter(
            x=support_points, y=cdf_estimate["median"], mode="markers+lines", name="median"
        )
        data.append(median_cdf)

        # in case we want to plot posterior samples
        #  my_post_samples = cdf_estimate["post_samples"]
        # for i in range(my_post_samples.shape[0]):
        #     sample_cdf = go.Scatter(
        #         x=support_points,
        #         y=my_post_samples[i, :],
        #         mode='markers+lines',
        #         name='sample'
        #     )
        #     data.append(sample_cdf)

    layout = {
        "title": "Cumulative Density Function",
        "xaxis": {"title": "QOI"},
        "yaxis": {"title": "CDF"},
    }

    fig = go.Figure(data=data, layout=layout)
    plotly.offline.plot(fig, filename="CDF.html", auto_open=True)


def plot_icdf(icdf_estimate: dict, bayes: bool = False) -> None:
    """Create icdf plot based on passed data.

    Args:
        icdf_estimate: Estimate of icdf at supporting points
        bayes: Do we want to plot confidence intervals
    """
    # Create a trace
    my_percentiles = icdf_estimate["x"]
    mean_icdf = go.Scatter(
        x=my_percentiles, y=icdf_estimate["mean"], mode="markers+lines", name="Mean"
    )
    data = [mean_icdf]

    if bayes:
        q5_icdf = go.Scatter(
            x=my_percentiles, y=icdf_estimate["q5"], mode="markers+lines", name="5% quantile"
        )
        data.append(q5_icdf)

        q95_icdf = go.Scatter(
            x=my_percentiles, y=icdf_estimate["q95"], mode="markers+lines", name="95% quantile"
        )
        data.append(q95_icdf)

        median_icdf = go.Scatter(
            x=my_percentiles, y=icdf_estimate["median"], mode="markers+lines", name="median"
        )
        data.append(median_icdf)

    layout = {
        "title": "Inverse Cumulative Density Function",
        "xaxis": {"title": "QOI"},
        "yaxis": {"title": "ICDF"},
    }

    fig = go.Figure(data=data, layout=layout)
    plotly.offline.plot(fig, filename="ICDF.html", auto_open=True)
