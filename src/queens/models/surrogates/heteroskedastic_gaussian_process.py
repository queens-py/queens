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
"""Implementation of a heteroskedastic Gaussian process models using GPFlow."""

import logging

import gpflow as gpf
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import tf_keras as keras
from sklearn.cluster import KMeans

from queens.models.surrogates._surrogate import Surrogate
from queens.utils.configure_tensorflow import configure_keras, configure_tensorflow
from queens.utils.logger_settings import log_init_args

_logger = logging.getLogger(__name__)

configure_tensorflow(tf)
configure_keras(keras)


class HeteroskedasticGaussianProcess(Surrogate):
    """Class for creating heteroskedastic GP based regression model.

    Class for creating heteroskedastic GP based regression model based on
    GPFlow.

    This class constructs a GP regression, currently using a GPFlow model.
    Currently, a lot of parameters are still hard coded, which will be
    improved in the future.

    The basic idea of this latent variable GP model can be found in [1-3].

    Attributes:
        num_posterior_samples (int): Number of posterior GP samples (realizations of posterior GP).
        num_inducing_points (int): Number of inducing points for variational GPs.
        model (gpf.model):  Gpflow based heteroskedastic Gaussian process model.
        optimizer (obj): Tensorflow optimization object.
        adams_training_rate (float): Training rate for the ADAMS gradient decent optimizer
        num_epochs (int): Number of training epochs for variational optimization.
        random_seed (int): Random seed used for initialization of stochastic gradient decent
                              optimizer.
        posterior_cov_mat_y (np.array): Posterior covariance matrix of heteroskedastic GP w.r.t.
                                        the y-coordinate.
        num_samples_stats (int): Number of samples used to calculate empirical
                                    variance/covariance.

    References:
        [1]: https://gpflow.github.io/GPflow/2.6.3/notebooks/advanced/heteroskedastic.html

        [2]: Saul, A. D., Hensman, J., Vehtari, A., & Lawrence, N. D. (2016, May).
             Chained gaussian processes. In Artificial Intelligence and Statistics (pp. 1431-1440).

        [3]: Hensman, J., Fusi, N., & Lawrence, N. D. (2013). Gaussian processes for big data.
             arXiv preprint arXiv:1309.6835.
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
        num_inducing_points=None,
        num_epochs=None,
        adams_training_rate=None,
        random_seed=None,
        num_samples_stats=None,
    ):
        """Initialize an instance of the Heteroskedastic GPflow class.

        Args:
            training_iterator (Iterator): Iterator to evaluate the subordinate model with the
                                          purpose of getting training data
            testing_iterator (Iterator): Iterator to evaluate the subordinate model with the purpose
                                         of getting testing data
            eval_fit (str): How to evaluate goodness of fit
            error_measures (list): List of error measures to compute
            plotting_options (dict): plotting options
            num_posterior_samples: Number of posterior GP samples
            num_inducing_points: Number of inducing points for variational GP approximation
            num_epochs (int): Number of epochs used for variational training of the GP
            adams_training_rate (float): Training rate for the ADAMS gradient decent optimizer
            random_seed (int): Random seed for stochastic optimization routine and samples
            num_samples_stats (int): Number of samples used to calculate empirical
                                     variance/covariance
        """
        # pylint: disable=duplicate-code
        super().__init__(
            training_iterator=training_iterator,
            testing_iterator=testing_iterator,
            eval_fit=eval_fit,
            error_measures=error_measures,
            plotting_options=plotting_options,
        )
        if num_samples_stats is None or num_samples_stats < 100:
            raise RuntimeError(
                f"You configured {num_samples_stats} number of samples for the calculation "
                "of the empirical posterior statistics. This number is either too low for "
                "reliable results or not a valid input. Please provide a valide integrer input "
                "greater than 100. Abort ..."
            )
        self.num_posterior_samples = num_posterior_samples
        self.num_inducing_points = num_inducing_points
        self.model = None
        self.optimizer = None
        self.adams_training_rate = adams_training_rate
        self.num_epochs = num_epochs
        self.random_seed = random_seed
        self.posterior_cov_mat_y = None
        self.num_samples_stats = num_samples_stats

    def setup(self, x_train, y_train):
        """Setup surrogate model.

        Args:
            x_train (np.array): training inputs
            y_train (np.array): training outputs
        """
        self.x_train = x_train
        self.y_train = y_train
        self._build_model()
        self.optimizer = self._build_optimizer()

    def train(self):
        """Train the variational.

        Train the variational by minimizing the variational loss in
        variational EM step.
        """
        np.random.seed(self.random_seed)
        _logger.info("# ---- Train the GPFlow model ----- #\n")
        for epoch in range(1, self.num_epochs + 1):
            self.optimizer()
            if epoch % 20 == 0 and epoch > 0:
                data = (self.x_train, self.y_train)
                loss_fun = self.model.training_loss_closure(data)
                _logger.info(
                    "Progress: %.2f %%, Epoch %s, Loss: %.2f",
                    epoch / self.num_epochs * 100,
                    epoch,
                    loss_fun().numpy(),
                )

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

    def predict(self, x_test, support=None, full_cov=False):
        """Predict the posterior distribution at the given test inputs.

        Predict the posterior distribution at *Xnew* with respect to the
        data *y*.

        Args:
            x_test (np.array): Inputs at which to evaluate or test the latent function 'f'
            support (str): Predict w.r.t. the latent variable 'f' or 'y'; not needed here as only
                           'y' makes sense
            full_cov (bool): Boolean that specifies whether the entire posterior covariance matrix
                             should be returned or only the posterior variance

        Returns:
            output (dict): Dictionary with mean, variance, and possibly
            posterior samples at *Xnew*
        """
        output = self.predict_y(x_test, full_cov=full_cov)

        return output

    def predict_y(self, x_test, full_cov=False):
        """Compute the posterior distribution at test inputs considering 'y'.

        Compute the posterior distribution at *x_test* with respect to the
        data 'y'.

        Args:
            x_test (np.array): Inputs at which to evaluate latent function 'f'
            full_cov (bool): Boolean to decide if we want posterior variance (*full_cov=False*) or
                             the full posterior covariance matrix (*full_cov=True*)

        Returns:
            output (dict): Dictionary with mean, variance, and possibly
            posterior samples of latent function at *x_test*
        """
        x_test = np.atleast_2d(x_test).reshape((-1, self.x_train.shape[1]))
        output = {}
        # TODO this method call is super slow as Cholesky decomp gets computed in every call #pylint: disable=fixme
        mean, variance = self.model.predict_y(x_test)
        mean = mean.numpy()
        variance = variance.numpy()
        if full_cov:
            posterior_samples_y = []
            posterior_samples_mean, posterior_samples_noise = self.predict_f_samples(
                x_test, self.num_samples_stats
            )
            for mean_sample, noise_sample in zip(posterior_samples_mean, posterior_samples_noise):
                posterior_samples_y.append(np.random.normal(mean_sample, np.exp(noise_sample)))

            posterior_samples_y = np.array(posterior_samples_y)
            self.posterior_cov_mat_y = np.cov(posterior_samples_y.T)
            variance = self.posterior_cov_mat_y

        output["result"] = mean
        output["variance"] = variance

        if self.num_posterior_samples:
            output["post_samples"] = self.predict_f_samples(x_test, self.num_posterior_samples)

        return output

    def predict_f_samples(self, x_test, num_samples):
        """Produce samples from the posterior latent function *Xnew*.

        Args:
            x_test (np.array):  Inputs at which to evaluate latent function 'f'
            num_samples (int):  Number of posterior realizations of GP

        Returns:
            post_samples (np.array): Posterior samples of latent functions at *x_test*
            (the latter might be a vector/matrix of points)
        """
        np.random.seed(self.random_seed)
        post_samples = self.model.predict_f_samples(x_test, num_samples).numpy()

        return post_samples[:, :, 0], post_samples[:, :, 1]

    def _build_model(self):
        """Build the GPflow heteroskedastic GP model."""
        # heteroskedastic conditional likelihood
        likelihood = gpf.likelihoods.HeteroskedasticTFPConditional(
            distribution_class=tfp.distributions.Normal,  # Gaussian Likelihood
            scale_transform=tfp.bijectors.Exp(),  # Exponential Transform
        )

        # separate independent kernels
        kernel = gpf.kernels.SeparateIndependent(
            [
                gpf.kernels.SquaredExponential(),  # This is k1, the kernel of f1
                gpf.kernels.SquaredExponential(),  # this is k2, the kernel of f2
            ]
        )

        # Initial inducing points position Z determined by clustering training data and take centers
        # of inferred clusters
        kmeans = KMeans(n_clusters=self.num_inducing_points)
        kmeans.fit(self.x_train)
        initial_inducing_points = kmeans.cluster_centers_

        inducing_variable = gpf.inducing_variables.SeparateIndependentInducingVariables(
            [
                gpf.inducing_variables.InducingPoints(
                    initial_inducing_points
                ),  # This is U1 = f1(Z1)
                gpf.inducing_variables.InducingPoints(
                    initial_inducing_points
                ),  # This is U2 = f2(Z2)
            ]
        )

        # the final model
        self.model = gpf.models.SVGP(
            kernel=kernel,
            likelihood=likelihood,
            inducing_variable=inducing_variable,
            num_latent_gps=likelihood.latent_dim,
        )

        _logger.info("The GPFlow model used in this analysis is constructed as follows:")
        gpf.utilities.print_summary(self.model)
        _logger.info("\n")

    def _build_optimizer(self):
        """Build the optimization step for the variational EM step.

        Returns:
            optimization_step_fun (obj): Tensorflow optimization EM-step function for variational
                                         optimization of the GP
        """
        # TODO pull below out to json # pylint: disable=fixme
        np.random.seed(self.random_seed)
        tf.random.set_seed(self.random_seed)
        data = (self.x_train, self.y_train)
        loss_fn = self.model.training_loss_closure(data)

        gpf.utilities.set_trainable(self.model.q_mu, False)
        gpf.utilities.set_trainable(self.model.q_sqrt, False)

        variational_vars = [(self.model.q_mu, self.model.q_sqrt)]
        natgrad_opt = gpf.optimizers.NaturalGradient(gamma=0.1)

        adam_vars = self.model.trainable_variables
        adam_opt = keras.optimizers.Adam(self.adams_training_rate)

        # here we conduct a two step optimization
        @tf.function
        def optimization_step_fun():
            """Perform a two-step variational EM for latent variable GPs.

            This function performs variational Expectation-Maximization using two different
            optimization routines:
            1. Natural Gradient Optimization: Minimizes the loss function with respect to the
               variational variables.
            2. Adam Optimization: Minimizes the loss function with respect to the Adam-specific
               variables.
            """
            natgrad_opt.minimize(loss_fn, variational_vars)
            adam_opt.minimize(loss_fn, adam_vars)

        return optimization_step_fun
