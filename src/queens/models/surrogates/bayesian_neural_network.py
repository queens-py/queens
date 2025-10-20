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
"""Implementation of a Bayesian Neural Network."""

import logging

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import tf_keras as keras

from queens.models.surrogates._surrogate import Surrogate
from queens.utils.configure_tensorflow import configure_keras, configure_tensorflow
from queens.utils.logger_settings import log_init_args

_logger = logging.getLogger(__name__)

tfd = tfp.distributions
DenseVar = tfp.layers.DenseVariational

configure_tensorflow(tf)
configure_keras(keras)


class GaussianBayesianNeuralNetwork(Surrogate):
    """A Bayesian Neural network.

    Class for creating a Bayesian neural network with Gaussian conditional
    distribution based on Tensorflow Probability.

    The network can handle heteroskedastic noise and an arbitrary nonlinear functional. As we use
    Tensorflow variational layers and learn the mean and variance function of a Gaussian
    distribution, the network is able to handle epistemic and aleatory uncertainty.

    Attributes:
        num_posterior_samples (int): Number of posterior sample functions (realizations of
                                     Bayesian neural network).
        num_samples_statistics (int): Number of samples to approximate posterior statistics.
        bnn_model (tf.model):  Tensorflow based Bayesian neural network model.
        num_epochs (int): Number of training epochs for variational optimization.
        optimizer_seed (int): Random seed used for initialization of stochastic gradient decent
                              optimizer.
        verbosity_on (bool): Boolean for model verbosity during training (*True=verbose*).
        model_realizations_lst (lst): List with different neural network realizations
                                      (epistemic uncertainty).
        adams_training_rate (float): Training rate for the ADAMS gradient decent optimizer
        nodes_per_hidden_layer_lst (lst): List containing number of nodes per hidden layer of the
                                          Bayesian Neural Network. The length of the list defines
                                          the deepness of the model and the values the width of the
                                          individual layers.
        activation_per_hidden_layer_lst (lst): List with strings encoding the activation function
                                               that shall be used for the respective hidden layer of
                                               the Bayesian Neural Network
    """

    @log_init_args
    def __init__(
        self,
        training_iterator=None,
        testing_iterator=None,
        eval_fit=None,
        error_measures=None,
        plotting_options=None,
        num_posterior_samples=None,
        num_samples_statistics=None,
        adams_training_rate=None,
        nodes_per_hidden_layer_lst=None,
        activation_per_hidden_layer_lst=None,
        num_epochs=None,
        optimizer_seed=None,
        verbosity_on=None,
    ):
        """Initialize an instance of the Gaussian Bayesian Neural Network.

        Args:
            training_iterator (Iterator): Iterator to evaluate the subordinate model with the
                                          purpose of getting training data
            testing_iterator (Iterator): Iterator to evaluate the subordinate model with the purpose
                                         of getting testing data
            eval_fit (str): How to evaluate goodness of fit
            error_measures (list): List of error measures to compute
            plotting_options (dict): plotting options
            num_posterior_samples (int): Number of posterior sample functions
            num_samples_statistics (int): Number of samples to approximate posterior statistics
            adams_training_rate (float): Training rate for the ADAMS gradient decent optimizer
            nodes_per_hidden_layer_lst (lst): List containing number of nodes per hidden layer of
                                              the Bayesian Neural Network. The length of the list
                                              defines the deepness of the model and the values the
                                              width of the individual layers.
            activation_per_hidden_layer_lst (lst): List with strings encoding the activation
                                                   function that shall be used for the
                                                   respective hidden layer of the Bayesian Neural
                                                   Network
            num_epochs (int): Number of epochs used for variational training of the BNN
            optimizer_seed (int): Random seed for stochastic optimization routine
            verbosity_on (bool): Boolean for model verbosity during training. True=verbose

        Returns:
            Instance of GaussianBayesianNeuralNetwork
        """
        # pylint: disable=duplicate-code
        super().__init__(
            training_iterator=training_iterator,
            testing_iterator=testing_iterator,
            eval_fit=eval_fit,
            error_measures=error_measures,
            plotting_options=plotting_options,
        )
        self.num_posterior_samples = num_posterior_samples
        self.num_samples_statistics = num_samples_statistics
        self.bnn_model = None
        self.num_epochs = num_epochs
        self.optimizer_seed = optimizer_seed
        self.verbosity_on = verbosity_on
        self.model_realizations_lst = None
        self.adams_training_rate = adams_training_rate
        self.nodes_per_hidden_layer_lst = nodes_per_hidden_layer_lst
        self.activation_per_hidden_layer_lst = activation_per_hidden_layer_lst

    def _build_model(self):
        """Build and compile the neural network.

        Build/compile the Bayesian neural network. We use a regular
        densely connected NN, which is parameterizing mean and variance
        of a Gaussian distribution. The network can be arbitrary deep
        and wide and can use different ( nonlinear) activation
        functions.
        """
        dense_architecture = [
            DenseVar(
                int(num_nodes),
                self.mean_field_variational_distribution,
                self.prior_trainable,
                activation=activation,
                kl_weight=1 / self.x_train.shape[0],
            )
            for num_nodes, activation in zip(
                self.nodes_per_hidden_layer_lst, self.activation_per_hidden_layer_lst
            )
        ]
        output_layer = [
            DenseVar(
                2,
                self.mean_field_variational_distribution,
                self.prior_trainable,
                kl_weight=1 / self.x_train.shape[0],
            ),
            tfp.layers.DistributionLambda(
                lambda d: tfd.Normal(
                    loc=d[..., :1], scale=1e-3 + tf.math.softplus(0.01 * d[..., 1:])
                )
            ),
        ]
        dense_architecture.extend(output_layer)
        self.bnn_model = keras.Sequential(dense_architecture)

        # compile the Tensorflow model
        self.bnn_model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.adams_training_rate),
            loss=self.negative_log_likelihood,
        )

    @staticmethod
    def negative_log_likelihood(y, random_variable_y):
        """Evaluate the negative log.-likelihood of the random variable.

        Negative logarithmic likelihood of (tensorflow) random variable
        *random_variable_y* evaluated at location *y*.

        Args:
            y (float): Value/Realization of the random variable
            random_variable_y (obj): Tensorflow probability random variable object

        Returns:
            negloglik (float): Negative logarithmic likelihood of *random_variable_y* at *y*
        """
        negloglik = -random_variable_y.log_prob(y)
        return negloglik

    @staticmethod
    def prior_trainable(kernel_size, bias_size=0, dtype=None):
        """Specify the prior over *keras.layers.Dense*, *kernel* and *bias*.

        **Note:** This is a special hybrid case of prior which is actually trainable.
        See "Empirical Bayes" for more background on this topic.

        Args:
            kernel_size (int): Number of weight parameters in the NN
            bias_size (int): Number of bias parameters in the NN
            dtype (str):  DataType string

        Returns:
            prior (obj): Tensorflow probability Gaussian prior distribution over biases and
            weights of the Bayesian NN, taking the mean of the Gaussian as an input
            variable for the prior distribution and fixing the variance
        """
        # TODO check this prior and make it full Bayesian? # pylint: disable=fixme
        n = kernel_size + bias_size

        # note: the mean of the Gaussian is an input variable here, hence we have n*mean input
        # variables; one mean per node of the NN
        prior_distribution = keras.Sequential(
            [
                tfp.layers.VariableLayer(n, dtype=dtype),
                tfp.layers.DistributionLambda(
                    lambda mean: tfd.Independent(
                        tfd.Normal(loc=mean, scale=1), reinterpreted_batch_ndims=1
                    )
                ),
            ]
        )
        return prior_distribution

    @staticmethod
    def mean_field_variational_distribution(kernel_size, bias_size=0, dtype=None):
        """Variational distribution that approximates the true posterior.

        Here, we use a Gaussian mean field approach, such that every parameter in the
        NN is approximated with a Gaussian distribution, parameterized by a
        mean and variance.

        Args:
            kernel_size (int): Number of weight parameters in the NN
            bias_size (int): Number of bias parameters in the NN
            dtype (str):  DataType string

        Returns:
            mean_field_variational_distr (obj): Tensorflow Probability mean field variational
            distribution for weights and biases. Each node (with weight and bias) has two Gaussians
            that are individually optimized via their mean and variance.
            Hence the parameterization of the variational distribution has n * 2 parameters with
            n being the number of nodes in the NN.
        """
        n = kernel_size + bias_size
        c = np.log(np.expm1(1.0))
        mean_field_variational_distr = keras.Sequential(
            [
                tfp.layers.VariableLayer(2 * n, dtype=dtype),
                tfp.layers.DistributionLambda(
                    lambda t: tfd.Independent(
                        tfd.Normal(loc=t[..., :n], scale=1e-5 + tf.nn.softplus(c + t[..., n:])),
                        reinterpreted_batch_ndims=1,
                    )
                ),
            ]
        )
        return mean_field_variational_distr

    def setup(self, x_train, y_train):
        """Setup surrogate model."""
        self.x_train = x_train
        self.y_train = y_train
        self._build_model()

    def train(self):
        """Train the Bayesian neural network.

        Train the Bayesian neural network using the previous defined
        optimizers in the model build and configuration. We allow
        Tensorflow's early stopping here to stop the optimization
        routine when the loss function starts to increase again over
        several iterations.
        """
        # set the random seeds for optimization/training
        np.random.seed(self.optimizer_seed)
        tf.random.set_seed(self.optimizer_seed)

        self.bnn_model.fit(
            self.x_train,
            self.y_train,
            epochs=self.num_epochs,
            verbose=self.verbosity_on,
        )

        # print out the model summary
        self.bnn_model.summary()

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
        """Make a prediction with the Bayesian neural network.

        Args:
            x_test (np.array): Testing input vector for which the posterior distribution,
                               respectively point estimates should be predicted
            support (str): String to determine which output
            full_cov (bool): Boolean flag for prediction posterior covariance matrix (*True*) or
                             posterior variance (*False*) at *x_test*

        Returns:
            output (dict): Dictionary with posterior output statistics
        """
        if self.model_realizations_lst is None:
            self.model_realizations_lst = [
                self.bnn_model for _ in range(self.num_samples_statistics)
            ]

        if support == "y":
            predict_method = self.predict_y
        elif support == "f":
            predict_method = self.predict_f
        else:
            raise RuntimeError("No suitable prediction method could be selected. Abort...")

        output = predict_method(x_test, full_cov=full_cov)

        return output

    def predict_y(self, x_test, full_cov=False):
        """Predict the posterior mean, variance/covariance.

        Predictions are conducted for the posterior distribution of the random output variable at
        *x_test* (w.r.t. *y*), that combines epistemic and aleatory uncertainty.

        Args:
            x_test (np.array): Testing input vector for which the posterior distribution,
                               respectively point estimates should be predicted
            full_cov (bool): Boolean flag for prediction posterior covariance matrix (*True*) or
                             posterior variance (*False*) at *x_test*

        Returns:
            output (dict): Dictionary with posterior output statistics
        """
        output = {}

        # sample over possible models (epistemic uncertainty)
        y_random_variable_models = [
            model_realization(x_test) for model_realization in self.model_realizations_lst
        ]
        # get mean of the individual gaussian models (aleatory uncertainty)
        y_random_variable_model_means = np.array(
            [y_random_variable.mean() for y_random_variable in y_random_variable_models]
        ).squeeze()
        # combine both sources of uncertainty averaging them
        output["result"] = np.atleast_2d(y_random_variable_model_means.mean(axis=0)).T

        # repeat the former process for variance/covariance estimates
        if full_cov:
            y_random_variable_model_var = (
                (
                    np.array(
                        [
                            y_random_variable.variance()
                            for y_random_variable in y_random_variable_models
                        ]
                    )
                )
                .squeeze()
                .mean(axis=0)
            )
            y_random_variable_model_cov = np.cov(y_random_variable_model_means.T) + np.diag(
                y_random_variable_model_var
            )
            output["variance"] = y_random_variable_model_cov
            return output

        y_random_variable_model_var = np.array(
            [y_random_variable.variance() for y_random_variable in y_random_variable_models]
        )
        output["variance"] = y_random_variable_model_var.mean(axis=0)

        return output

    def predict_f(self, x_test, full_cov=False):
        """Predict the posterior statistics of the latent function 'f'.

        Predict the posterior mean, variance or covariance of the posterior
        distribution w.r.t. the latent function 'f' at test points *x_test*.
        This prediction only accounts for the epistemic uncertainty and
        neglects the aleatory uncertainty.

        Args:
            x_test (np.array): Testing input vector for which the posterior distribution,
                               respectively point estimates should be predicted
            full_cov (bool): Boolean flag for prediction posterior covariance matrix (*True*) or
                             posterior variance (*False*) at *x_test*

        Returns:
            output (dict): Dictionary with posterior output statistics
        """
        output = {}

        # sample over possible models (epistemic uncertainty)
        y_random_variable_models = [
            model_realization(x_test) for model_realization in self.model_realizations_lst
        ]
        y_random_variable_model_means = np.array(
            [y_random_variable.mean() for y_random_variable in y_random_variable_models]
        )
        output["result"] = y_random_variable_model_means.mean(axis=0)

        # repeat the former process for variance/covariance estimates
        if full_cov:
            samples = self.predict_f_samples(x_test, self.num_posterior_samples)
            output["variance"] = np.cov(samples)
            return output
        output["variance"] = y_random_variable_model_means.var(axis=0)
        return output

    def predict_f_samples(self, x_test, num_samples):
        """Predict posterior samples of the latent function 'f'.

        Sample the latent function (without noise), that is in this
        case realizations of the mean predictions of the Bayesian neural
        network which are themselves uncertain.

        Args:
            x_test (np.array): Input testing point at which the samples should be realized
            num_samples (int): Number of posterior samples of the latent function 'f'

        Returns:
            samples (np.array): Samples of the latent function at locations *x_test*
        """
        y_random_variable_models = [
            model_realization(x_test) for model_realization in self.model_realizations_lst
        ]
        y_random_variable_model_mean_samples = np.array(
            [y_random_variable.mean() for y_random_variable in y_random_variable_models]
        )
        samples = np.reshape(y_random_variable_model_mean_samples, (x_test.shape[0], num_samples))

        return samples

    def predict_y_samples(self, x_test, num_samples):
        """Sample from the posterior distribution of 'y'.

        Sampling from the posterior (w.r.t. *y*) that combines epistemic and
        aleatory uncertainty. This will generally lead to noisy samples in
        contrast to smooth samples from the latent function as noise assumption
        is conditionally independent for each point *x_test*.

        Args:
            x_test (np.array): Testing input locations for the posterior distribution
            num_samples (int): Number of posterior samples at *x_test*

        Returns:
            samples (np.array): Posterior samples w.r.t. *y*
        """
        # sample the different model realizations (epistemic uncertainty)
        y_random_variable_models = [
            model_realization(x_test) for model_realization in self.model_realizations_lst
        ]
        # from each model draw one sample
        y_random_variable_model_samples = np.array(
            [
                y_random_variable.sample(
                    int(self.num_posterior_samples / self.num_samples_statistics)
                )
                for y_random_variable in y_random_variable_models
            ]
        )

        samples = np.reshape(y_random_variable_model_samples, (x_test.shape[0], num_samples))
        return samples
