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
"""Fixtures for the integration tests of iterators."""

import numpy as np
import pandas as pd
import pytest

from example_simulator_functions.gaussian_logpdf import (
    STANDARD_NORMAL,
    gaussian_1d_logpdf,
    gaussian_2d_logpdf,
)
from example_simulator_functions.park91a import X3, X4, park91a_hifi_on_grid


@pytest.fixture(name="_create_experimental_data_park91a_hifi_on_grid")
def fixture_create_experimental_data_park91a_hifi_on_grid(tmp_path):
    """Create a csv file with experimental data."""
    # Fix random seed
    np.random.seed(seed=1)

    # True input values
    x1 = 0.5
    x2 = 0.2

    y_vec = park91a_hifi_on_grid(x1, x2)

    # Artificial noise
    sigma_n = 0.001
    noise_vec = np.random.normal(loc=0, scale=sigma_n, size=(y_vec.size,))

    # Inverse crime: Add artificial noise to model output for the true value
    y_fake = y_vec + noise_vec

    # write fake data to csv
    data_dict = {
        "x3": X3,
        "x4": X4,
        "y_obs": y_fake,
    }
    experimental_data_path = tmp_path / "experimental_data.csv"
    dataframe = pd.DataFrame.from_dict(data_dict)
    dataframe.to_csv(experimental_data_path, index=False)


@pytest.fixture(name="_create_experimental_data_gaussian_1d")
def fixture_create_experimental_data_gaussian_1d(tmp_path):
    """Create a csv file with experimental data from a 1D Gaussian."""
    # generate 10 samples from the same gaussian
    samples = STANDARD_NORMAL.draw(10).flatten()

    # evaluate the gaussian pdf for these 1000 samples
    pdf = []
    for sample in samples:
        pdf.append(gaussian_1d_logpdf(sample))

    pdf = np.array(pdf).flatten()

    # write the data to a csv file in tmp_path
    data_dict = {"y_obs": pdf}
    experimental_data_path = tmp_path / "experimental_data.csv"
    dataframe = pd.DataFrame.from_dict(data_dict)
    dataframe.to_csv(experimental_data_path, index=False)


@pytest.fixture(name="_create_experimental_data_zero")
def fixture_create_experimental_data_zero(tmp_path):
    """Create a csv file with experimental data equal to zero."""
    samples = np.array([0, 0]).flatten()

    # write the data to a csv file in tmp_path
    data_dict = {"y_obs": samples}
    experimental_data_path = tmp_path / "experimental_data.csv"
    dataframe = pd.DataFrame.from_dict(data_dict)
    dataframe.to_csv(experimental_data_path, index=False)


@pytest.fixture(name="target_density_gaussian_1d")
def fixture_target_density_gaussian_1d():
    """A function mimicking a 1D Gaussian distribution."""

    def target_density_gaussian_1d(self, samples):  # pylint: disable=unused-argument
        """Target posterior density."""
        samples = np.atleast_2d(samples)
        log_likelihood = gaussian_1d_logpdf(samples)

        return log_likelihood

    return target_density_gaussian_1d


@pytest.fixture(name="target_density_gaussian_2d")
def fixture_target_density_gaussian_2d():
    """A function mimicking a 2D Gaussian distribution."""

    def target_density_gaussian_2d(self, samples):  # pylint: disable=unused-argument
        """Target likelihood density."""
        samples = np.atleast_2d(samples)
        log_likelihood = gaussian_2d_logpdf(samples)

        return log_likelihood

    return target_density_gaussian_2d


@pytest.fixture(name="target_density_gaussian_2d_with_grad")
def fixture_target_density_gaussian_2d_with_grad():
    """A function mimicking a 2D Gaussian distribution."""

    def target_density_gaussian_2d_with_grad(self, samples):  # pylint: disable=unused-argument
        """Target likelihood density."""
        samples = np.atleast_2d(samples)
        log_likelihood = gaussian_2d_logpdf(samples)

        cov = [[1.0, 0.5], [0.5, 1.0]]
        cov_inverse = np.linalg.inv(cov)
        gradient = -np.dot(cov_inverse, samples.T).T

        return log_likelihood, gradient

    return target_density_gaussian_2d_with_grad
