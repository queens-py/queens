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
"""Collection of utility functions for post-processing."""

import logging
import pickle
import warnings
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity

from queens.utils.plot_outputs import plot_cdf, plot_icdf, plot_pdf

_logger = logging.getLogger(__name__)


def process_outputs(
    output_data: dict, output_description: dict, input_data: np.ndarray | None = None
) -> dict:
    """Process output from QUEENS models.

    Args:
        output_data: Dictionary containing model output
        output_description: Dictionary describing desired output quantities
        input_data: Array containing model input

    Returns:
        Dictionary with processed results
    """
    processed_results = {}
    try:
        processed_results = do_processing(output_data, output_description)
    except (TypeError, KeyError) as error:
        _logger.warning("Error occurred during result processing: %s", str(error))

    # add the actual raw input and output data
    processed_results["raw_output_data"] = output_data
    if input_data is not None:
        processed_results["input_data"] = input_data

    return processed_results


def do_processing(output_data: dict, output_description: dict) -> dict:
    """Do actual processing of output.

    Args:
        output_data: Dictionary containing model output
        output_description: Dictionary describing desired output quantities

    Returns:
        Dictionary with processed results
    """
    # do we want confidence intervals
    bayesian = output_description.get("bayesian", False)
    # check if we have the data to support this
    if "post_samples" not in output_data and bayesian:
        warnings.warn(
            "Warning: Output data does not contain posterior samples. "
            "Not computing confidence intervals"
        )
        bayesian = False

    # do we want plotting
    plot_results = output_description.get("plot_results", False)

    # result interval
    result_interval = output_description.get("result_interval", None)
    # TODO: we get an error below! # pylint: disable=fixme

    if result_interval is None:
        # estimate interval from results
        result_interval = estimate_result_interval(output_data)

    # get number of support points
    num_support_points = output_description.get("num_support_points", 100)
    support_points = np.linspace(result_interval[0], result_interval[1], num_support_points)

    mean_mean = estimate_mean(output_data)
    var_mean = estimate_var(output_data)

    processed_results: dict = {}
    processed_results["mean"] = mean_mean
    processed_results["var"] = var_mean

    if output_description.get("cov", False):
        cov_mean = estimate_cov(output_data)
        processed_results["cov"] = cov_mean

    # do we want to estimate all the below (i.e. pdf, cdf, icdf)
    est_all = output_description.get("estimate_all", False)

    # do we want pdf estimation
    est_pdf = output_description.get("estimate_pdf", False)
    if est_pdf or est_all:
        pdf_estimate = estimate_pdf(output_data, support_points, bayesian)
        if plot_results:
            plot_pdf(pdf_estimate, support_points, bayesian)

        processed_results["pdf_estimate"] = pdf_estimate

    # do we want cdf estimation
    est_cdf = output_description.get("estimate_cdf", False)
    if est_cdf or est_all:
        cdf_estimate = estimate_cdf(output_data, support_points, bayesian)
        if plot_results:
            plot_cdf(cdf_estimate, support_points, bayesian)

        processed_results["cdf_estimate"] = cdf_estimate

    # do we want icdf estimation
    est_icdf = output_description.get("estimate_icdf", False)
    if est_icdf or est_all:
        icdf_estimate = estimate_icdf(output_data, bayesian)

        if plot_results:
            plot_icdf(icdf_estimate, bayesian)

        processed_results["icdf_estimate"] = icdf_estimate

    return processed_results


def write_results(processed_results: Any, file_path: Path) -> None:
    """Write results to pickle file.

    Args:
        processed_results: Dictionary with results
        file_path: Path to pickle file to write results to
    """
    with open(file_path, "wb") as handle:
        pickle.dump(processed_results, handle, protocol=pickle.HIGHEST_PROTOCOL)


def estimate_result_interval(output_data: dict) -> list:
    """Estimate interval of output data.

    Estimate interval of output data and add small margins.

    Args:
        output_data: Dictionary with output data

    Returns:
        Output interval
    """
    samples = output_data["result"]
    _logger.debug(samples)
    min_data = np.amin(samples)
    _logger.debug(min_data)
    max_data = np.amax(samples)

    interval_length = max_data - min_data
    my_min = min_data - interval_length / 6
    my_max = max_data + interval_length / 6

    return [my_min, my_max]


def estimate_mean(output_data: dict) -> np.ndarray:
    """Estimate mean based on standard unbiased estimator.

    Args:
        output_data: Dictionary with output data

    Returns:
        Unbiased mean estimate
    """
    samples = output_data["result"]
    return np.mean(samples, axis=0)


def estimate_var(output_data: dict) -> np.ndarray:
    """Estimate variance based on standard unbiased estimator.

    Args:
        output_data: Dictionary with output data

    Returns:
        Unbiased variance estimate
    """
    samples = output_data["result"]
    return np.var(samples, ddof=1, axis=0)


def estimate_cov(output_data: dict) -> np.ndarray:
    """Estimate covariance based on standard unbiased estimator.

    Args:
        output_data: Dictionary with output data

    Returns:
        Unbiased covariance estimate
    """
    samples = output_data["result"]

    # we assume that rows represent observations and columns represent variables
    row_variable = False

    cov = np.zeros((samples.shape[1], samples.shape[2], samples.shape[2]))
    for i in range(samples.shape[1]):
        cov[i] = np.cov(samples[:, i, :], rowvar=row_variable)
    return cov


def estimate_cdf(output_data: dict, support_points: np.ndarray, bayesian: bool) -> dict:
    """Compute estimate of CDF based on provided sampling data.

    Args:
        output_data: Dictionary with output data
        support_points: Points where to evaluate cdf
        bayesian: Compute confidence intervals etc.

    Returns:
        Dictionary with cdf estimates
    """
    cdf: dict = {}
    cdf["x"] = support_points
    if not bayesian:
        raw_data = output_data["result"]
        size_data = raw_data.size
        cdf_values_lst = []
        for i in support_points:
            # all the values in raw data less than the ith value in x_values
            temp = raw_data[raw_data <= i]
            # fraction of that value with respect to the size of the x_values
            value = temp.size / size_data
            cdf_values_lst.append(value)
        cdf["mean"] = np.array(cdf_values_lst)
    else:
        raw_data = output_data["post_samples"]
        size_data = len(support_points)
        num_realizations = raw_data.shape[1]
        cdf_values: np.ndarray = np.zeros((num_realizations, len(support_points)))
        for i in range(num_realizations):
            data = raw_data[:, i]
            for j, point in enumerate(support_points):
                # all the values in raw data less than the ith value in x_values
                temp = data[data <= point]
                # fraction of that value with respect to the size of the x_values
                value = temp.size / size_data
                cdf_values[i, j] = value

        cdf["post_samples"] = cdf_values
        # now we compute mean, median cumulative distribution function
        cdf["mean"] = np.mean(cdf_values, axis=0)
        cdf["median"] = np.median(cdf_values, axis=0)
        cdf["q5"] = np.percentile(cdf_values, 5, axis=0)
        cdf["q95"] = np.percentile(cdf_values, 95, axis=0)

    return cdf


def estimate_icdf(output_data: dict, bayesian: bool) -> dict:
    """Compute estimate of inverse CDF based on provided sampling data.

    Args:
        output_data: Dictionary with output data
        bayesian: Compute confidence intervals etc.

    Returns:
        Dictionary with icdf estimates
    """
    my_percentiles = 100 * np.linspace(0 + 1 / 1000, 1 - 1 / 1000, 999)
    icdf = {}
    icdf["x"] = my_percentiles
    if not bayesian:
        samples = output_data["result"]
        icdf_values = np.zeros_like(my_percentiles)
        for i, percentile in enumerate(my_percentiles):
            icdf_values[i] = np.percentile(samples, percentile, axis=0)
        icdf["mean"] = icdf_values
    else:
        raw_data = output_data["post_samples"]
        num_realizations = raw_data.shape[1]
        icdf_values = np.zeros((len(my_percentiles), num_realizations))
        for i, point in enumerate(my_percentiles):
            icdf_values[i, :] = np.percentile(raw_data, point, axis=0)

        icdf["post_samples"] = icdf_values
        # now we compute mean, median cumulative distribution function
        icdf["mean"] = np.mean(icdf_values, axis=1)
        icdf["median"] = np.median(icdf_values, axis=1)
        icdf["q5"] = np.percentile(icdf_values, 5, axis=1)
        icdf["q95"] = np.percentile(icdf_values, 95, axis=1)

    return icdf


def estimate_pdf(output_data: dict, support_points: np.ndarray, bayesian: bool) -> dict:
    """Compute estimate of PDF based on provided sampling data.

    Args:
        output_data: Dictionary with output data
        support_points: Points where to evaluate pdf
        bayesian: Compute confidence intervals etc.

    Returns:
        Dictionary with pdf estimates
    """
    pdf = {}
    pdf["x"] = support_points
    if not bayesian:
        samples = output_data["result"]
        min_samples = np.amin(samples)
        max_samples = np.amax(samples)
        bandwidth = estimate_bandwidth_for_kde(samples, min_samples, max_samples)
        pdf["mean"] = perform_kde(samples, bandwidth, support_points)
    else:
        min_samples = np.amin(support_points)
        max_samples = np.amax(support_points)
        mean_samples = output_data["result"]
        # estimate kernel bandwidth only once
        bandwidth = estimate_bandwidth_for_kde(mean_samples, min_samples, max_samples)
        raw_data = output_data["post_samples"]
        num_realizations = raw_data.shape[1]
        pdf_values = np.zeros((num_realizations, len(support_points)))
        for i in range(num_realizations):
            data = raw_data[:, i]
            pdf_values[i, :] = perform_kde(data, bandwidth, support_points)

        pdf["post_samples"] = pdf_values
        # now we compute mean, median probability density function
        pdf["mean"] = np.mean(pdf_values, axis=0)
        pdf["median"] = np.median(pdf_values, axis=0)
        pdf["q5"] = np.percentile(pdf_values, 5, axis=0)
        pdf["q95"] = np.percentile(pdf_values, 95, axis=0)

    return pdf


def estimate_bandwidth_for_kde(
    samples: np.ndarray, min_samples: float, max_samples: float
) -> float:
    """Estimate optimal bandwidth for kde of pdf.

    Args:
        samples: Samples for which to estimate pdf
        min_samples: Smallest value
        max_samples: Largest value
    Returns:
        Estimate for optimal kernel bandwidth
    """
    kernel_bandwidth = 0
    kernel_bandwidth_upper_bound = (max_samples - min_samples) / 2.0
    kernel_bandwidth_lower_bound = (max_samples - min_samples) / 20.0

    # do 20-fold cross validaton unless we have fewer samples
    num_cv = min(samples.shape[0], 20)
    # cross-validation
    grid = GridSearchCV(
        KernelDensity(),
        {"bandwidth": np.linspace(kernel_bandwidth_lower_bound, kernel_bandwidth_upper_bound, 40)},
        cv=num_cv,
    )

    grid.fit(samples.reshape(-1, 1))
    kernel_bandwidth = grid.best_params_["bandwidth"]

    return kernel_bandwidth


def perform_kde(
    samples: np.ndarray, kernel_bandwidth: float, support_points: np.ndarray
) -> np.ndarray:
    """Estimate pdf using kernel density estimation.

    Args:
        samples: Samples for which to estimate pdf
        kernel_bandwidth: Kernel width to use in kde
        support_points: Points where to evaluate pdf
    Returns:
        PDF estimate at support points
    """
    kde = KernelDensity(kernel="gaussian", bandwidth=kernel_bandwidth).fit(samples.reshape(-1, 1))

    y_density = np.exp(kde.score_samples(support_points.reshape(-1, 1)))
    return y_density
