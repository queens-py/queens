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
"""Classifiers for use in convergence classification."""

import pickle
from pathlib import Path

import numpy as np
from skactiveml.classifier import SklearnClassifier
from skactiveml.pool import UncertaintySampling
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from queens.utils.logger_settings import log_init_args

VALID_CLASSIFIER_TYPES = {
    "nn": MLPClassifier,
    "gp": GaussianProcessClassifier,
    "svc": SVC,
}


class Classifier:
    """Classifier wrapper.

    Attributes:
        n_params (int): number of parameters of the solver
        classifier_obj (obj): classifier, e.g. sklearn.svm.SVR
    """

    is_active = False

    def __init__(self, n_params, classifier_obj):
        """Initialise the classifier.

        Args:
            n_params (int): number of parameters
            classifier_obj (obj): classifier, e.g. sklearn.svm.SVR
        """
        self.n_params = n_params
        self.classifier_obj = classifier_obj

    def train(self, x_train, y_train):
        """Train the underlying _clf classifier.

        Args:
            x_train: array with training samples, size: (n_samples, n_params)
            y_train: vector with corresponding training labels, size: (n_samples)
        """
        self.classifier_obj.fit(x_train, y_train)

    def predict(self, x_test):
        """Perform prediction on given parameter combinations.

        Args:
            x_test (np.array): array of parameter combinations (n_samples, n_params)

        Returns:
            y_test: prediction value or vector (n_samples)
        """
        # Binary classification
        return np.round(self.classifier_obj.predict(x_test))

    def load(self, path, file_name):
        """Load pickled classifier.

        Args:
            path (str): Path to export the classifier
            file_name (str): File name without suffix
        """
        pickle_file = Path(path) / (file_name + ".pickle")
        with pickle_file.open("rb") as file:
            self.classifier_obj = pickle.load(file)


class ActiveLearningClassifier(Classifier):
    """Active learning classifier wrapper.

    Attributes:
        n_params (int): number of parameters of the solver
        classifier_obj (obj): classifier, e.g. sklearn.svm.SVR
        active_sampler_obj: query strategy from skactiveml.pool, e.g. UncertaintySampling
    """

    is_active = True

    @log_init_args
    def __init__(self, n_params, classifier_obj, batch_size, active_sampler_obj=None):
        """Initialise active learning classifier.

        Args:
            n_params (int): number of parameters of the solver
            classifier_obj (obj): classifier, e.g. sklearn.svm.SVR
            active_sampler_obj (obj): query strategy from skactiveml.pool, e.g. UncertaintySampling
            batch_size (int): Batch size to query the next samples.
        """
        super().__init__(n_params, SklearnClassifier(classifier_obj, classes=range(2)))
        if active_sampler_obj is not None:
            self.active_sampler_obj = active_sampler_obj
        else:
            self.active_sampler_obj = UncertaintySampling(method="entropy", random_state=0)
        self.batch_size = batch_size

    def train(self, x_train, y_train):
        """Train the underlying _clf classifier.

        Args:
            x_train (np.array): array with training samples, size: (n_samples, n_params)
            y_train (np.array): vector with corresponding training labels, size: (n_samples)

        Returns:
            query_idx (np.array): sample indices in x_train to query next
        """
        self.classifier_obj.fit(x_train, y_train)
        query_idx = self.active_sampler_obj.query(
            X=x_train, y=y_train, clf=self.classifier_obj, batch_size=self.batch_size
        )
        return query_idx


VALID_CLASSIFIER_LEARNING_TYPES = {
    "passive_learning": Classifier,
    "active_learning": ActiveLearningClassifier,
}
