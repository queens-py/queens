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
"""Global fixtures and configurations for integration tests."""

import numpy as np
import pandas as pd
import pytest

from queens.example_simulator_functions.gaussian_logpdf import (
    GAUSSIAN_2D,
    STANDARD_NORMAL,
    gaussian_1d_logpdf,
    gaussian_2d_logpdf,
)
from queens.example_simulator_functions.park91a import park91a_hifi
from test_utils.integration_tests import get_input_park91a


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


@pytest.fixture(name="_create_experimental_data_gaussian_2d")
def fixture_create_experimental_data_gaussian_2d(tmp_path):
    """Create a csv file with experimental data from a 2D Gaussian."""
    # generate 10 samples from the same gaussian
    samples = GAUSSIAN_2D.draw(10)
    pdf = gaussian_2d_logpdf(samples)

    pdf = np.array(pdf)

    # write the data to a csv file in tmp_path
    data_dict = {"y_obs": pdf}
    experimental_data_path = tmp_path / "experimental_data.csv"
    df = pd.DataFrame.from_dict(data_dict)
    df.to_csv(experimental_data_path, index=False)


@pytest.fixture(name="_create_experimental_data_zero")
def fixture_create_experimental_data_zero(tmp_path):
    """Create a csv file with experimental data equal to zero."""
    samples = np.array([0, 0]).flatten()

    # write the data to a csv file in tmp_path
    data_dict = {"y_obs": samples}
    experimental_data_path = tmp_path / "experimental_data.csv"
    dataframe = pd.DataFrame.from_dict(data_dict)
    dataframe.to_csv(experimental_data_path, index=False)


@pytest.fixture(name="training_data_park91a")
def fixture_training_data_park91a():
    """Create training data from the park91a benchmark function."""
    # create training inputs
    n_train = 7
    x_train, x_3, x_4 = get_input_park91a(n_train)

    # evaluate the testing/benchmark function at training inputs
    y_train = park91a_hifi(x_train[:, 0], x_train[:, 1], x_3, x_4, gradient_bool=False)
    y_train = y_train.reshape(-1, 1)

    return x_train, y_train


@pytest.fixture(name="testing_data_park91a")
def fixture_testing_data_park91a():
    """Create testing data for the park91a benchmark function."""
    # create testing inputs
    n_test = 25
    x_test, x_3, x_4 = get_input_park91a(n_test)

    # evaluate the testing/benchmark function at testing inputs
    mean_ref, gradient_mean_ref = park91a_hifi(
        x_test[:, 0], x_test[:, 1], x_3, x_4, gradient_bool=True
    )
    mean_ref = mean_ref.reshape(-1, 1)
    gradient_mean_ref = np.array(gradient_mean_ref).T
    var_ref = np.zeros(mean_ref.shape)
    gradient_variance_ref = np.zeros(gradient_mean_ref.shape)

    return x_test, mean_ref, var_ref, gradient_mean_ref, gradient_variance_ref


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
