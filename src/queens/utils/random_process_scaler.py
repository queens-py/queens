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
"""Utils for data scaling."""

import abc

import numpy as np


class Scaler(metaclass=abc.ABCMeta):
    """Base class for general scaling classes.

    The purpose of these classes is the scaling of training data.

    Attributes:
        mean (np.array): Mean-values of the data-matrix (column-wise).
        standard_deviation (np.array): Standard deviation of the data-matrix (per column).
    """

    def __init__(self):
        """Initialise scaler.

        Returns:
            Instance of the Scaler Class (obj)
        """
        self.mean = None
        self.standard_deviation = None

    @abc.abstractmethod
    def fit(self, x_mat):
        """Fit/calculate the scaling based on the input samples.

        Args:
            x_mat (np.array): Data matrix that should be standardized
        """

    @abc.abstractmethod
    def transform(self, x_mat):
        """Conduct the scaling transformation on the input samples.

        Args:
            x_mat (np.array): Data matrix that should be standardized
        """

    @abc.abstractmethod
    def inverse_transform_mean(self, x_mat):
        """Conduct the inverse transformation for the mean.

        Args:
            x_mat (np.array): Data matrix that should be standardized
        """

    @abc.abstractmethod
    def inverse_transform_std(self, x_mat):
        """Conduct the inverse transformation.

        Args:
            x_mat (np.array): Data matrix that should be standardized
        """


class StandardScaler(Scaler):
    """Scaler for standardization of data.

    In case a stochastic process is trained on the scaled data, inverse
    rescaling is implemented to recover the correct mean and standard
    deviation prediction for the posterior process.
    """

    def fit(self, x_mat):
        """Fit/calculate the scaling based on the input samples.

        Args:
            x_mat (np.array): Data matrix that should be standardized
        """
        self.mean = np.mean(x_mat)
        self.standard_deviation = np.std(x_mat)

    def transform(self, x_mat):
        """Conduct the scaling transformation on the data matrix.

        Args:
            x_mat (np.array): Data matrix that should be standardized

        Returns:
            transformed_data (np.array): Transformed data-array
        """
        transformed_data = (x_mat - self.mean) / self.standard_deviation
        return transformed_data

    def inverse_transform_mean(self, x_mat):
        """Conduct the inverse scaling transformation on the data matrix.

        Args:
            x_mat (np.array): Data matrix that should be standardized

        Returns:
            transformed_data (np.array): Transformed data-array
        """
        transformed_data = x_mat * self.standard_deviation + self.mean

        return transformed_data

    def inverse_transform_std(self, x_mat):
        """Conduct the inverse scaling transformation.

        The data is transformed based on the standard deviation.

        Args:
            x_mat (np.array): Data matrix that should be standardized

        Returns:
            transformed_data (np.array): Transformed data-array
        """
        transformed_data = x_mat * self.standard_deviation

        return transformed_data

    def inverse_transform_grad_mean(self, grad_mean, standard_deviation_input):
        """Conduct the inverse scaling of the mean gradient.

        Args:
            grad_mean (np.array): gradient of the transformed mean
                                  function
            standard_deviation_input (float): standard deviation of the input data

        Returns:
            transformed_grad (np.array): Inversely transformed gradient of
                                         the mean function
        """
        factor = self.standard_deviation / standard_deviation_input
        transformed_grad = factor * grad_mean
        return transformed_grad

    def inverse_transform_grad_var(self, grad_var, var, trans_var, input_standard_deviation):
        """Conduct the inverse scaling of the variance gradient.

        Args:
            grad_var (np.array): gradient of the transformed variance
            var (np.array): variance of the untransformed data
            trans_var (np.array): variance of the transformed data
            input_standard_deviation (float): standard deviation of the input data

        Returns:
            transformed_grad (np.array): Inversely transformed gradient of
                                         the variance function
        """
        factor = self.standard_deviation / input_standard_deviation
        grad_std = 1 / (2 * np.sqrt(var)) * grad_var
        transformed_grad_std = factor * grad_std
        transformed_grad_var = 2 * np.sqrt(trans_var) * transformed_grad_std
        return transformed_grad_var


class IdentityScaler(Scaler):
    """The identity scaler."""

    def fit(self, x_mat):
        """Fit/calculate the scaling based on the input samples.

        Args:
            x_mat (np.array): Data matrix that should be standardized
        """

    def transform(self, x_mat):
        """Conduct the scaling transformation on the data matrix.

        Args:
            x_mat (np.array): Data matrix that should be standardized

        Returns:
            transformed_data (np.array): Transformed data-array
        """
        return x_mat

    def inverse_transform_mean(self, x_mat):
        """Conduct the inverse scaling transformation on the data matrix.

        Args:
            x_mat (np.array): Data matrix that should be standardized

        Returns:
            transformed_data (np.array): Transformed data-array
        """
        return x_mat

    def inverse_transform_std(self, x_mat):
        """Conduct the inverse scaling.

        Args:
            x_mat (np.array): Data matrix that should be standardized

        Returns:
            transformed_data (np.array): Transformed data-array
        """
        return x_mat

    def inverse_transform_grad_mean(self, grad_mean, *_args):
        """Conduct the inverse scaling of the mean gradient.

        Args:
            grad_mean (np.array): gradient of the transformed mean
                                  function

        Returns:
            transformed_grad (np.array): Inversely transformed gradient of
                                         then mean function
        """
        transformed_grad = grad_mean
        return transformed_grad

    def inverse_transform_grad_var(self, grad_var, *_args):
        """Conduct the inverse scaling of the variance gradient.

        Args:
            grad_var (np.array): gradient of the transformed variance
                                 function

        Returns:
            transformed_grad (np.array): Inversely transformed gradient of
                                         then variance function
        """
        transformed_grad = grad_var
        return transformed_grad


VALID_SCALER = {"standard_scaler": StandardScaler, "identity_scaler": IdentityScaler}
