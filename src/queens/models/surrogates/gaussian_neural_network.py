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
"""Gaussian Neural Network regression model."""

import logging

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import tf_keras as keras

from queens.models.surrogates._surrogate import Surrogate
from queens.utils.configure_tensorflow import configure_keras, configure_tensorflow
from queens.utils.logger_settings import log_init_args
from queens.utils.scaling import VALID_SCALER
from queens.utils.valid_options import get_option
from queens.visualization.gaussian_neural_network_vis import plot_loss

tfd = tfp.distributions

configure_tensorflow(tf)
configure_keras(keras)

_logger = logging.getLogger(__name__)


class GaussianNeuralNetwork(Surrogate):
    """Class for creating a neural network that parameterizes a Gaussian.

    The network can handle heteroskedastic noise and an arbitrary nonlinear functions.

    Attributes:
        nn_model (tf.model):  Tensorflow based Bayesian neural network model
        num_epochs (int): Number of training epochs for variational optimization
        optimizer_seed (int): Random seed used for initialization of stochastic gradient decent
                              optimizer
        verbosity_on (bool): Boolean for model verbosity during training. True=verbose
        batch_size (int): Size of data-batch (smaller than the training data size)
        scaler_x (obj): Scaler for inputs
        scaler_y (obj): Scaler for outputs
        loss_plot_path (str): Path to determine whether loss plot should be produced
                              (yes if provided). Plot will be saved at path location.
        num_refinements (int): Number of refinements
        refinement_epochs_decay (float): Decrease of epochs in refinements
        mean_function (function): Mean function of the Gaussian Neural Network
        gradient_mean_function (function): Gradient of the mean function of the Gaussian
                                           Neural Network
        adams_training_rate (float): Training rate for the ADAMS gradient decent optimizer
        nodes_per_hidden_layer (lst): List containing number of nodes per hidden layer of
                                      the Neural Network. The length of the list
                                      defines the deepness of the model and the values the
                                      width of the individual layers.
        activation_per_hidden_layer (list): List with strings encoding the activation
                                            function that shall be used for the
                                            respective hidden layer of the  Neural
                                            Network
        kernel_initializer (str): Type of kernel initialization for neural network
        nugget_std (float): Nugget standard deviation for robustness
    """

    @log_init_args
    def __init__(
        self,
        num_epochs=None,
        batch_size=None,
        adams_training_rate=None,
        optimizer_seed=None,
        verbosity_on=None,
        nodes_per_hidden_layer_lst=None,
        activation_per_hidden_layer_lst=None,
        kernel_initializer=None,
        nugget_std=None,
        loss_plot_path=False,
        refinement_epochs_decay=0.75,
        data_scaling=None,
        mean_function_type="zero",
    ):
        """Initialize an instance of the Gaussian Bayesian Neural Network.

        Args:
            num_epochs (int): Number of epochs used for variational training of the BNN
            batch_size (int): Size of data-batch (smaller than the training data size)
            adams_training_rate (float): Training rate for the ADAMS gradient decent optimizer
            optimizer_seed (int): Random seed for stochastic optimization routine
            verbosity_on (bool): Boolean for model verbosity during training. True=verbose
            nodes_per_hidden_layer_lst (lst): List containing number of nodes per hidden layer of
                                          the Neural Network. The length of the list
                                          defines the deepness of the model and the values the
                                          width of the individual layers.
            activation_per_hidden_layer_lst (list): List with strings encoding the activation
                                                function that shall be used for the
                                                respective hidden layer of the  Neural
                                                Network
            kernel_initializer (str): Type of kernel initialization for neural network
            nugget_std (float): Nugget standard deviation for robustness
            loss_plot_path (str): Path to determine whether loss plot should be produced
                                  (yes if provided). Plot will be saved at path location.
            refinement_epochs_decay (float): Decrease of epochs in refinements
            data_scaling (str): Data scaling type
            mean_function_type (str): Mean function type of the Gaussian Neural Network

        Returns:
            Instance of GaussianBayesianNeuralNetwork
        """
        super().__init__()
        # check mean function and subtract from y_train
        valid_mean_function_types = {
            "zero": (lambda x: 0, lambda x: 0),
            "identity_multi_fidelity": (
                lambda x: np.atleast_2d(x[:, 0]).T,
                lambda x: np.hstack((np.ones(x.shape[0]).reshape(-1, 1), np.zeros(x[:, 1:].shape))),
            ),
        }

        mean_function, gradient_mean_function = get_option(
            valid_mean_function_types, mean_function_type, "mean_function_type"
        )

        self.nn_model = None
        self.num_epochs = num_epochs
        self.optimizer_seed = optimizer_seed
        self.verbosity_on = verbosity_on
        self.batch_size = batch_size
        self.scaler_x = get_option(VALID_SCALER, data_scaling)()
        self.scaler_y = get_option(VALID_SCALER, data_scaling)()
        self.loss_plot_path = loss_plot_path
        self.num_refinements = 0
        self.refinement_epochs_decay = refinement_epochs_decay
        self.mean_function = mean_function
        self.gradient_mean_function = gradient_mean_function

        self.adams_training_rate = adams_training_rate
        self.nodes_per_hidden_layer = nodes_per_hidden_layer_lst
        self.activation_per_hidden_layer = activation_per_hidden_layer_lst
        self.kernel_initializer = kernel_initializer
        self.nugget_std = nugget_std

    def _build_model(self):
        """Build/compile the neural network.

        We use a regular densely connected
        NN, which is parameterizing mean and variance of a Gaussian
        distribution. The network can be arbitrary deep and wide and can use
        different (nonlinear) activation functions.

        Returns:
            model (obj): Tensorflow probability model instance
        """
        # hidden layers
        output_dim = self.y_train.shape[1]

        dense_architecture = [
            keras.layers.Dense(
                int(num_nodes),
                activation=activation,
                kernel_initializer=self.kernel_initializer,
            )
            for num_nodes, activation in zip(
                self.nodes_per_hidden_layer, self.activation_per_hidden_layer
            )
        ]

        # Gaussian output layer
        output_layer = [
            keras.layers.Dense(
                2 * output_dim,
                activation="linear",
            ),
            tfp.layers.DistributionLambda(
                lambda d: tfd.Normal(
                    loc=d[..., :output_dim],
                    scale=self.nugget_std + tf.math.softplus(0.1 * d[..., output_dim:]),
                )
            ),
        ]
        dense_architecture.extend(output_layer)
        model = keras.Sequential(dense_architecture)

        # compile the Tensorflow model
        optimizer = keras.optimizers.Adamax(learning_rate=self.adams_training_rate)

        model.compile(
            optimizer=optimizer,
            loss=self.negative_log_likelihood,
        )

        return model

    @staticmethod
    def negative_log_likelihood(y, random_variable_y):
        """Negative log-likelihood of (tensorflow) random variable.

        Args:
            y (float): Value/Realization of the random variable
            random_variable_y (obj): Tensorflow probability random variable object

        Returns:
            negative_log_likelihood (float): Negative logarithmic likelihood of random_variable_y
                                             at y
        """
        negative_log_likelihood = -random_variable_y.log_prob(y)
        return negative_log_likelihood

    def update_training_data(self, x_train, y_train):
        """Update the training data of the model.

        Args:
            x_train (np.array): Training input array
            y_train (np.array): Training output array
        """
        num_old_samples = self.x_train.shape[0]
        x_train_new = self.scaler_x.transform(x_train[num_old_samples:].T).T
        y_train_new = self.scaler_y.transform(y_train[num_old_samples:])
        self.x_train = np.vstack((self.x_train, x_train_new))
        self.y_train = np.vstack((self.y_train, y_train_new))

    def setup(self, x_train, y_train):
        """Setup surrogate model.

        Args:
            x_train (np.array): training inputs
            y_train (np.array): training outputs
        """
        y_train = y_train - self.mean_function(x_train)

        self.scaler_x.fit(x_train)
        self.x_train = self.scaler_x.transform(x_train)
        self.scaler_y.fit(y_train)
        self.y_train = self.scaler_y.transform(y_train)

        self.nn_model = self._build_model()

    def train(self):
        """Train the Bayesian neural network.

        We ues the previous defined optimizers in the model build and
        configuration. We allow tensorflow's early stopping here to stop
        the optimization routine when the loss function starts to
        increase again over several iterations.
        """
        # make epochs adaptive with a simple schedule, lower bound is 1/5 of the initial epoch
        if self.num_refinements > 0:
            self.num_epochs = int(
                max(self.num_epochs * self.refinement_epochs_decay, self.num_epochs / 5)
            )
        self.num_refinements += 1

        # set the random seeds for optimization/training
        keras.utils.set_random_seed(self.optimizer_seed)
        history = self.nn_model.fit(
            self.x_train,
            self.y_train,
            epochs=self.num_epochs,
            verbose=self.verbosity_on,
            batch_size=self.batch_size,
        )

        # print out the model summary
        self.nn_model.summary()

        if self.loss_plot_path:
            plot_loss(history, self.loss_plot_path)

    def grad(self, samples, upstream_gradient):
        """Evaluate gradient of model w.r.t.

        current set of input samples.
        """
        raise NotImplementedError

    def predict(self, x_test, support="y", gradient_bool=False):
        """Predict the output distribution at x_test.

        Args:
            x_test (np.array): Testing input vector for which the posterior distribution,
                               respectively point estimates should be predicted
            support (str, optional): String to define the support of the output distribution
                                    - 'y': Conditional distribution is defined on the output space
                                    - 'f': Conditional distribution is defined on the latent space
            gradient_bool (bool, optional): Boolean to configure whether gradients should be
                                            returned as well

        Returns:
            output (dict): Dictionary with posterior output statistics
        """
        if support == "f":
            raise NotImplementedError('Support "f" is not implemented yet.')

        if gradient_bool:
            output = self.predict_and_gradient(x_test)
        else:
            output = self.predict_y(x_test)

        output["x_test"] = x_test
        return output

    def predict_y(self, x_test):
        """Predict the posterior mean and variance.

        Prediction is conducted w.r.t. to the output space "y".

        Args:
            x_test (np.array): Testing input vector for which the posterior distribution,
                               respectively point estimates should be predicted

        Returns:
            output (dict): Dictionary with posterior output statistics
        """
        x_test_transformed = self.scaler_x.transform(x_test)
        yhat = self.nn_model(x_test_transformed)
        mean_pred = np.atleast_2d(yhat.mean()).T
        var_pred = np.atleast_2d(yhat.variance()).T

        output = {"variance_untransformed": var_pred}
        output["variance"] = (self.scaler_y.inverse_transform_std(np.sqrt(var_pred)) ** 2).reshape(
            -1, 1
        )
        output["result"] = self.scaler_y.inverse_transform_mean(mean_pred).reshape(
            -1, 1
        ) + self.mean_function(x_test)

        return output

    def predict_and_gradient(self, x_test):
        """Predict the mean, variance and their gradients at x_test.

        Args:
            x_test (np.array): Testing input vector for which the posterior
                               distribution, respectively point estimates should be
                               predicted

        Returns:
            output (dict): Dictionary with posterior output statistics
        """
        x_test_transformed = self.scaler_x.transform(x_test)
        x_test_tensorflow = tf.Variable(x_test_transformed)
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x_test_tensorflow)
            yhat = self.nn_model(x_test_tensorflow)
            mean_pred = yhat.mean()
            var_pred = yhat.variance()

        grad_mean = tape.gradient(mean_pred, x_test_tensorflow).numpy()
        grad_var = tape.gradient(var_pred, x_test_tensorflow).numpy()

        mean_pred = np.array(mean_pred.numpy()).reshape(-1, 1)
        var_pred_untransformed = np.array(var_pred.numpy()).reshape(-1, 1)

        # write mean and variance to output dictionary
        output = {
            "result": self.scaler_y.inverse_transform_mean(mean_pred).reshape(-1, 1)
            + self.mean_function(x_test)
        }
        output["variance"] = (
            self.scaler_y.inverse_transform_std(np.sqrt(var_pred_untransformed)) ** 2
        ).reshape(-1, 1)

        # write gradients to output dictionary
        output["grad_mean"] = self.scaler_y.inverse_transform_grad_mean(
            grad_mean, self.scaler_x.standard_deviation
        ) + self.gradient_mean_function(x_test)

        output["grad_var"] = self.scaler_y.inverse_transform_grad_var(
            grad_var,
            var_pred_untransformed,
            output["variance"],
            self.scaler_x.standard_deviation,
        )

        return output
