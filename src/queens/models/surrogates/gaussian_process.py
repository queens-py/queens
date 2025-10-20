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
"""Gaussian process implementation in GPFlow."""

import logging

import gpflow as gpf
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from queens.models.surrogates._surrogate import Surrogate
from queens.utils.configure_tensorflow import configure_tensorflow
from queens.utils.gpflow_transformations import init_scaler, set_transform_function
from queens.utils.logger_settings import log_init_args
from queens.utils.numpy_array import extract_block_diag

_logger = logging.getLogger(__name__)

configure_tensorflow(tf)


class GaussianProcess(Surrogate):
    """Class for creating GP regression model based on GPFlow.

    This class constructs a GP regression, using a GPFlow model.

    Attributes:
        number_posterior_samples (int): Number of posterior samples.
        number_restarts (int): Number of restarts.
        number_training_iterations (int): Number of iterations in optimizer for training.
        number_input_dimensions (int): Dimensionality of random features.
        restart_min_value (int): Minimum value for restart.
        restart_max_value (int): Maximum value for restart.
        model (GPFlow.models.GPR): GPFlow based Gaussian process model.
        dimension_lengthscales (int): Dimension of *lengthscales*.
        train_likelihood_variance (bool): if true, likelihood variance is trained
        scaler_x (sklearn scaler object): Scaler for inputs.
        scaler_y (sklearn scaler object): Scaler for outputs.
    """

    @log_init_args
    def __init__(
        self,
        training_iterator=None,
        testing_iterator=None,
        eval_fit=None,
        error_measures=None,
        plotting_options=None,
        number_posterior_samples=None,
        seed_optimizer=42,
        seed_posterior_samples=None,
        restart_min_value=0,
        restart_max_value=5,
        number_restarts=10,
        number_training_iterations=100,
        dimension_lengthscales=None,
        train_likelihood_variance=True,
    ):
        """Initialize an instance of the GPFlow regression model.

        Args:
            training_iterator (Iterator): Iterator to evaluate the subordinate model with the
                                          purpose of getting training data
            testing_iterator (Iterator): Iterator to evaluate the subordinate model with the purpose
                                         of getting testing data
            eval_fit (str): How to evaluate goodness of fit
            error_measures (list): List of error measures to compute
            plotting_options (dict): plotting options
            number_posterior_samples (int): Number of posterior samples
            seed_optimizer (int): Seed for optimizer
            seed_posterior_samples (int): Seed for posterior samples
            restart_min_value (int): Minimum value for restart
            restart_max_value (int): Maximum value for restart
            number_restarts (int): Number of restarts
            number_training_iterations (int): Number of iterations in optimizer for training
            dimension_lengthscales (int): Dimension of lengthscales
            train_likelihood_variance (bool): if true, likelihood variance is trained
        """
        # pylint: disable=duplicate-code
        super().__init__(
            training_iterator=training_iterator,
            testing_iterator=testing_iterator,
            eval_fit=eval_fit,
            error_measures=error_measures,
            plotting_options=plotting_options,
        )
        self.number_posterior_samples = number_posterior_samples
        self.seed_optimizer = seed_optimizer
        self.seed_posterior_samples = seed_posterior_samples
        self.number_restarts = number_restarts
        self.number_training_iterations = number_training_iterations
        self.number_input_dimensions = None
        self.restart_min_value = restart_min_value
        self.restart_max_value = restart_max_value
        self.model = None
        self.dimension_lengthscales = dimension_lengthscales
        self.train_likelihood_variance = train_likelihood_variance
        self.scaler_x = None
        self.scaler_y = None

    def setup(self, x_train, y_train):
        """Setup surrogate model.

        Args:
            x_train (np.array): training inputs
            y_train (np.array): training outputs
        """
        if len(x_train.shape) == 1:
            self.number_input_dimensions = 1
        else:
            self.number_input_dimensions = x_train.shape[1]

        if y_train.shape[0] != x_train.shape[0]:
            y_train = np.reshape(y_train, (-1, 1))

        self.scaler_x, self.x_train = init_scaler(x_train)
        self.scaler_y, self.y_train = init_scaler(y_train)

        # initialize hyperparameters
        lengthscales_0 = 0.1 * np.ones(self.dimension_lengthscales)
        variances_0 = max(abs(np.max(self.y_train) - np.min(self.y_train)), 1e-6)

        # choose kernel
        kernel = gpf.kernels.RBF(lengthscales=lengthscales_0, variance=variances_0)

        # initialize model
        self.model = gpf.models.GPR(
            data=(self.x_train, self.y_train), kernel=kernel, mean_function=None
        )

        if not self.train_likelihood_variance:
            self.model.likelihood.variance.assign(1.1e-6)  # small value for numerical stability
            gpf.utilities.set_trainable(self.model.likelihood.variance, False)

        self.model.kernel.lengthscales = set_transform_function(
            self.model.kernel.lengthscales, tfp.bijectors.Exp()
        )
        self.model.kernel.variance = set_transform_function(
            self.model.kernel.variance, tfp.bijectors.Exp()
        )

    def train(self):
        """Train the GP by maximizing the likelihood."""
        opt = gpf.optimizers.Scipy()

        dimension_hyperparameters = self.get_dimension_hyperparameters()
        loss = np.ones([self.number_restarts + 1])
        train_logs = []
        for i in range(self.number_restarts + 1):
            if i > 0:
                tf.random.set_seed(self.seed_optimizer)
                hyperparameters = tf.random.uniform(
                    [dimension_hyperparameters],
                    minval=self.restart_min_value,
                    maxval=self.restart_max_value,
                    seed=i,
                )
                self.assign_hyperparameters(hyperparameters, transform=False)
            try:
                opt_logs = opt.minimize(
                    self.model.training_loss,
                    self.model.trainable_variables,
                    options={"maxiter": self.number_training_iterations},
                )
                loss[i] = opt_logs.fun
                train_logs.append(opt_logs)
            except tf.errors.InvalidArgumentError:
                loss[i] = np.nan
                train_logs.append("Optimization Failed")
            _logger.info("restart %s/%s    loss = %s", i, self.number_restarts, loss[i])

        hyperparameters = train_logs[int(np.nanargmin(loss))].x

        self.assign_hyperparameters(hyperparameters, transform=True)

        gpf.utilities.print_summary(self.model)

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
        raise NotImplementedError

    def predict(self, x_test, support="y", full_cov=False):
        """Predict the posterior distribution at *x_new*.

        Options:

            - *f(x_test)*: predict the latent function values
            - *y(x_test)*: predict values of the new observations (including noise)

        Args:
            x_test (np.ndarray): New inputs where to make predictions
            support (str): Probabilistic support of random process (default: 'y')
            full_cov (bool): Boolean that specifies whether the entire posterior covariance
                             matrix should be returned or only the posterior variance

        Returns:
            output (dict): Dictionary with mean, variance, and possibly
            posterior samples at *x_test*
        """
        x_test = np.atleast_2d(x_test).reshape((-1, self.number_input_dimensions))
        number_test_samples = x_test.shape[0]
        x_test = self.scaler_x.transform(x_test)

        if support == "y":
            mean, var = self.model.predict_y(x_test, full_cov=False)
        elif support == "f":
            mean, var = self.model.predict_f(x_test, full_cov=full_cov)
        else:
            mean = None
            var = None

        mean = self.scaler_y.inverse_transform(mean.numpy())
        var = var.numpy() * self.scaler_y.var_

        output = {"result": mean.reshape(number_test_samples, -1), "x_test": x_test}
        if support == "f" and full_cov:
            output["variance"] = np.squeeze(var, axis=0)
            output["variance_diagonal"] = extract_block_diag(
                np.squeeze(var, axis=0), output["result"].shape[1]
            )
        else:
            output["variance"] = var

        if self.number_posterior_samples:
            # set seed for reproducibility of posterior samples
            if self.seed_posterior_samples:
                tf.random.set_seed(self.seed_posterior_samples)
                _logger.warning("Beware, the seed for drawing posterior samples is fixed.")

            output["post_samples"] = (
                self.model.predict_f_samples(x_test, self.number_posterior_samples)
                .numpy()
                .reshape((x_test.shape[0], self.number_posterior_samples))
            )

        return output

    def assign_hyperparameters(self, hyperparameters, transform=False):
        """Assign untransformed (constrained) hyperparameters to model.

        Args:
            hyperparameters (np.ndarray):   Hyperparameters of GP
            transform (bool):               If *True*, hyperparameters are transformed from
                                            unconstrained to constrained representation
        """
        hyperparameters = tf.convert_to_tensor(hyperparameters)
        if transform:
            hyperparameters = self.transform_hyperparameters(hyperparameters)

        self.model.kernel.lengthscales.assign(hyperparameters[0 : self.dimension_lengthscales])
        self.model.kernel.variance.assign(hyperparameters[self.dimension_lengthscales])

    def transform_hyperparameters(self, hyperparameters):
        """Transform hyperparameters.

        Transform hyperparameters from unconstrained to constrained
        representation.

        Args:
            hyperparameters (np.ndarray):   Unconstrained representation of hyperparameters
        Returns:
            hyperparameters (np.ndarray): Constrained representation of hyperparameters
        """
        hyperparameters = tf.convert_to_tensor(hyperparameters)

        lengthscales = self.model.kernel.lengthscales.transform.forward(
            hyperparameters[0 : self.dimension_lengthscales]
        )
        variances = tf.reshape(
            self.model.kernel.variance.transform.forward(
                hyperparameters[self.dimension_lengthscales]
            ),
            [-1],
        )

        hyperparameters = tf.concat([lengthscales, variances], 0)

        return hyperparameters

    def get_dimension_hyperparameters(self):
        """Return the dimension of the hyperparameters.

        Returns:
            dimension_hyperparameters (int): Dimension of hyperparameters
        """
        lengthscales = self.model.kernel.lengthscales.unconstrained_variable
        variances = tf.reshape(self.model.kernel.variance.unconstrained_variable, [-1])
        hyperparameters = tf.concat([lengthscales, variances], 0)

        dimension_hyperparameters = hyperparameters.shape.dims[0]

        return dimension_hyperparameters
