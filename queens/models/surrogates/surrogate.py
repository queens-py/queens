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
"""Surrogate model class."""

import abc
import logging

import numpy as np
from sklearn.model_selection import KFold

from queens.models.model import Model
from queens.visualization.surrogate_visualization import SurrogateVisualization

_logger = logging.getLogger(__name__)


class Surrogate(Model):
    """Surrogate model class.

    Attributes:
        training_iterator (Iterator): Iterator to evaluate the subordinate model with the purpose of
                                      getting training data
        testing_iterator (Iterator): Iterator to evaluate the subordinate model with the purpose of
                                     getting testing data
        eval_fit (str): How to evaluate goodness of fit
        error_measures (list): List of error measures to compute
        is_trained (bool): true if model is trained
        x_train (np.array): training inputs
        y_train (np.array): training outputs
        visualization (SurrogateVisualization): Surrogate visualization object.
    """

    def __init__(
        self,
        training_iterator=None,
        testing_iterator=None,
        eval_fit=None,
        error_measures=None,
        plotting_options=None,
    ):
        """Initialize data fit.

        Args:possi     training_iterator (Iterator): Iterator to
        evaluate the subordinate model with the purpose of getting
        training data     testing_iterator (Iterator): Iterator to
        evaluate the subordinate model with the purpose of getting
        testing data eval_fit (str): How to evaluate goodness of fit
        error_measures (list): List of error measures to compute
        efficiency should be evaluated     plotting_options (dict):
        plotting options
        """
        super().__init__()

        self.visualization = None
        if plotting_options:
            self.visualization = SurrogateVisualization.from_config_create(plotting_options)

        self.training_iterator = training_iterator
        self.testing_iterator = testing_iterator
        self.eval_fit = eval_fit
        self.error_measures = error_measures
        self.is_trained = False
        self.x_train = None
        self.y_train = None

    def evaluate(self, samples):
        """Evaluate model with current set of input samples.

        Args:
            samples (np.ndarray): Input samples

        Returns:
            dict: Results corresponding to current set of input samples
        """
        if not self.is_trained:
            self.build_approximation()

        self.response = self.predict(samples)
        return self.response

    @abc.abstractmethod
    def grad(self, samples, upstream_gradient):
        r"""Evaluate gradient of model w.r.t. current set of input samples.

        Consider current model f(x) with input samples x, and upstream function g(f). The provided
        upstream gradient is :math:`\frac{\partial g}{\partial f}` and the method returns
        :math:`\frac{\partial g}{\partial f} \frac{df}{dx}`.

        Args:
            samples (np.array): Input samples
            upstream_gradient (np.array): Upstream gradient function evaluated at input samples
                                          :math:`\frac{\partial g}{\partial f}`

        Returns:
            gradient (np.array): Gradient w.r.t. current set of input samples
                                 :math:`\frac{\partial g}{\partial f} \frac{df}{dx}`
        """

    @abc.abstractmethod
    def predict(self, x_test, support="y"):
        """Predict."""

    @abc.abstractmethod
    def setup(self, x_train, y_train):
        """Setup surrogate model.

        Args:
            x_train (np.array): training inputs
            y_train (np.array): training outputs
        """

    @abc.abstractmethod
    def train(self):
        """Train surrogate model."""

    def build_approximation(self):
        """Build underlying approximation."""
        self.training_iterator.run()

        # get samples and results
        x_train, y_train = self._get_data_set(self.training_iterator)

        if self.eval_fit == "kfold":
            error_measures = self.eval_surrogate_accuracy_cv(
                x_test=x_train, y_test=y_train, k_fold=5, measures=self.error_measures
            )
            for measure, error in error_measures.items():
                _logger.info("Error %s is: %s", measure, error)
        # TODO check that final surrogate is on all points # pylint: disable=fixme

        # train regression model on the data
        self.setup(x_train, y_train)
        self.train()
        self.is_trained = True

        # TODO: Passing self is ugly # pylint: disable=fixme
        if self.visualization:
            self.visualization.plot(self.training_iterator.parameters.names, self)

        if self.testing_iterator:
            self.testing_iterator.run()

            x_test, y_test = self._get_data_set(self.testing_iterator)

            error_measures = self.eval_surrogate_accuracy(x_test, y_test, self.error_measures)
            for measure, error in error_measures.items():
                _logger.info("Error %s is: %s", measure, error)

    def eval_surrogate_accuracy(self, x_test, y_test, measures):
        """Evaluate the accuracy of the surrogate model based on test set.

        Evaluate the accuracy of the surrogate model using the provided
        error metrics.

        Args:
            x_test (np.array):  Test inputs
            y_test (np.array):  Test outputs
            measures (list):    List with desired error metrics

        Returns:
            dict: Dictionary with proving error metrics
        """
        if not self.is_trained:
            raise RuntimeError("Cannot compute accuracy on uninitialized model")

        response = self.predict(x_test)
        y_prediction = response["result"].reshape((-1, 1))

        error_info = {}
        if measures is not None:
            error_info = self.compute_error_measures(y_test, y_prediction, measures)
        return error_info

    def eval_surrogate_accuracy_cv(self, x_test, y_test, k_fold, measures):
        """Compute k-fold cross-validation error.

        Args:
            x_test (np.array):       Input array
            y_test (np.array):       Output array
            k_fold (int):       Split dataset in `k_fold` subsets for cv
            measures (list):    List with desired error metrics

        Returns:
            dict:y with error measures and corresponding error values
        """
        response_cv = self.cross_validate(x_test, y_test, k_fold)
        y_prediction = np.reshape(np.array(response_cv), (-1, 1))
        error_info = self.compute_error_measures(y_test, y_prediction, measures)

        return error_info

    def cross_validate(self, x_train, y_train, folds):
        """Cross validation function which calls the regression approximation.

        Args:
            x_train (np.array):   Array of inputs
            y_train (np.array):   Array of outputs
            folds (int):    In how many subsets do we split for cv

        Returns:
            np.array: Array with predictions
        """
        # init output array
        outputs = np.zeros_like(y_train, dtype=float)
        # set random_state=None, shuffle=False)
        # TODO check out randomness feature # pylint: disable=fixme
        kf = KFold(n_splits=folds)
        kf.get_n_splits(x_train)

        for train_index, test_index in kf.split(x_train):
            self.setup(x_train[train_index], y_train[train_index])
            self.train()
            outputs[test_index] = self.predict(x_train[test_index].T, support="f")["result"]

        return outputs

    def compute_error_measures(self, y_test, y_posterior_mean, measures):
        """Compute error measures.

        Compute based on difference between predicted and actual values.

        Args:
            y_test (ndarray): Output values from testing data set
            y_posterior_mean (ndarray): Posterior mean values of the GP
            measures (list):   Dictionary with desired error measures

        Returns:
            dict: Dictionary with error measures and corresponding error values
        """
        error_measures = {}
        for measure in measures:
            error_measures[measure] = self.compute_error(y_test, y_posterior_mean, measure)
        return error_measures

    @staticmethod
    def compute_error(y_test, y_posterior_mean, measure):
        """Compute error for given a specific error measure.

        Args:
            y_test (ndarray): Output values from testing data set
            y_posterior_mean (ndarray): Posterior mean values of the GP
            measure (str):     Desired error metric

        Returns:
            float: Error based on desired metric
        """
        return {
            "sum_squared": lambda: np.sum((y_test - y_posterior_mean) ** 2),
            "mean_squared": lambda: np.mean((y_test - y_posterior_mean) ** 2),
            "root_mean_squared": lambda: np.sqrt(np.mean((y_test - y_posterior_mean) ** 2)),
            "sum_abs": lambda: np.sum(np.abs(y_test - y_posterior_mean)),
            "mean_abs": lambda: np.mean(np.abs(y_test - y_posterior_mean)),
            "abs_max": lambda: np.max(np.abs(y_test - y_posterior_mean)),
            "nash_sutcliffe_efficiency": lambda: Surrogate.compute_nash_sutcliffe_efficiency(
                y_test, y_posterior_mean
            ),
        }.get(measure, NotImplementedError("Desired error measure is unknown!"))()

    @staticmethod
    def compute_nash_sutcliffe_efficiency(y_test, y_posterior_mean):
        r"""Compute Nash-Sutcliffe model efficiency.

        .. math::
            NSE = 1-\frac{\sum_{i=1}^{N}(e_{i}-s_{i})^2}{\sum_{i=1}^{N}(e_{i}-\bar{e})^2}

        Args:
            y_test (ndarray): Output values from testing data set
            y_posterior_mean (ndarray): Posterior mean values of the GP

        Returns:
            efficiency (float): Nash-Sutcliffe model efficiency
        """
        if len(y_test) == len(y_posterior_mean):
            y_posterior_mean, y_test = np.array(y_posterior_mean), np.array(y_test)
            if y_test.shape != y_posterior_mean.shape:
                y_posterior_mean = y_posterior_mean.transpose()

            mean_observed = np.nanmean(y_test)
            numerator = np.nansum((y_test - y_posterior_mean) ** 2)
            denominator = np.nansum((y_test - mean_observed) ** 2)
            efficiency = 1 - (numerator / denominator)
            return efficiency

        _logger.warning("Evaluation and simulation lists does not have the same length.")
        return np.nan

    @staticmethod
    def _get_data_set(iterator):
        """Get input and output from iterator.

        Args:
            iterator (queens.iterators.Iterator): iterator where to get input and output from

        Returns:
            x (ndarray): input (samples)
            y (ndarray): output (response)
        """
        if hasattr(iterator, "samples"):
            x = iterator.samples
        else:
            raise AttributeError(
                f"Your iterator {type(iterator).__name__} has no samples and, thus, cannot be used "
                f"for training or testing a surrogate model."
            )

        if hasattr(iterator, "output"):
            y = iterator.output["result"]
        else:
            raise AttributeError(
                f"Your iterator {type(iterator).__name__} has no output data and, thus, cannot be "
                f"used for training or testing a surrogate model."
            )

        return x, y
