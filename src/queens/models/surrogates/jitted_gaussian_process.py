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
"""A fast, jitted version of Gaussian Process regression."""

import logging

import numpy as np
from scipy.linalg import cho_solve

import queens.models.surrogates.utils.kernel_jitted as utils_jitted
from queens.models.surrogates._surrogate import Surrogate
from queens.utils.logger_settings import log_init_args
from queens.utils.scaling import VALID_SCALER
from queens.utils.valid_options import get_option
from queens.visualization.gnuplot_vis import gnuplot_gp_convergence

_logger = logging.getLogger(__name__)


class JittedGaussianProcess(Surrogate):
    """A jitted Gaussian process implementation using numba.

    It just-in-time compiles linear algebra operations.
    The GP also allows to specify a Gamma hyper-prior or the length scale,
    but only computes the MAP estimate and does not
    marginalize the hyper-parameters.

    Attributes:
        k_mat_inv (np.array): Inverse of the assembled covariance matrix.
        cholesky_k_mat (np.array): Lower Cholesky decomposition of the covariance matrix.
        k_mat (np.array): Assembled covariance matrix of the GP.
        partial_derivatives_hyper_params (list): List of partial derivatives of the
                                                 kernel function w.r.t. the hyper-parameters.
        mean_function (function): Mean function of the GP
        gradient_mean_function (function): Gradient of the mean function of the GP
        stochastic_optimizer (obj): Stochastic optimizer object.
        scaler_x (obj): Scaler for inputs.
        scaler_y (obj): Scaler for outputs.
        grad_log_evidence_value (np.array): Current gradient of the log marginal likelihood w.r.t.
                                            the parameterization.
        hyper_params (list): List of hyper-parameters
        noise_variance_lower_bound (float): Lower bound for Gaussian noise variance in RBF kernel.
        plot_refresh_rate (int): Refresh rate of the plot (every n-iterations).
        kernel_type (str): Type of kernel function.
    """

    valid_kernels_dict = {
        "squared_exponential": (
            utils_jitted.squared_exponential,
            utils_jitted.posterior_mean_squared_exponential,
            utils_jitted.posterior_var_squared_exponential,
            utils_jitted.grad_log_evidence_squared_exponential,
            utils_jitted.grad_posterior_mean_squared_exponential,
            utils_jitted.grad_posterior_var_squared_exponential,
        ),
        "matern_3_2": (
            utils_jitted.matern_3_2,
            utils_jitted.posterior_mean_matern_3_2,
            utils_jitted.posterior_var_matern_3_2,
            utils_jitted.grad_log_evidence_matern_3_2,
            utils_jitted.grad_posterior_mean_matern_3_2,
            utils_jitted.grad_posterior_var_matern_3_2,
        ),
    }

    @log_init_args
    def __init__(
        self,
        stochastic_optimizer,
        initial_hyper_params_lst=None,
        kernel_type=None,
        data_scaling=None,
        mean_function_type="zero",
        plot_refresh_rate=None,
        noise_var_lb=None,
    ):
        """Instantiate the jitted Gaussian Process.

        Args:
            stochastic_optimizer (obj): Stochastic optimizer object.
            initial_hyper_params_lst (list): List of initial hyper-parameters
            kernel_type (str): Type of kernel used in the GP
            data_scaling (str): Data scaling type
            mean_function_type (str): Mean function type of the GP
            plot_refresh_rate (int): Refresh rate of the plot (every n-iterations).
            noise_var_lb (float): Lower bound for Gaussian noise variance in RBF kernel.
        """
        super().__init__()
        if initial_hyper_params_lst is None:
            raise ValueError("The initial hyper-parameters were not provided!")

        if kernel_type is None:
            raise ValueError(
                "You did not specify a valid kernel! Valid kernels are "
                f"{JittedGaussianProcess.valid_kernels_dict.keys()}, but you specified "
                f"{kernel_type}."
            )

        scaler_x = get_option(VALID_SCALER, data_scaling)()
        scaler_y = get_option(VALID_SCALER, data_scaling)()

        # check mean function and subtract from y_train
        valid_mean_function_types = {
            "zero": (
                JittedGaussianProcess.zero_mean_fun,
                JittedGaussianProcess.gradient_zero_mean_fun,
            ),
            "identity_multi_fidelity": (
                JittedGaussianProcess.identity_multi_fidelity_mean_fun,
                JittedGaussianProcess.gradient_identity_multi_fidelity_mean_fun,
            ),
        }

        mean_function, gradient_mean_function = get_option(
            valid_mean_function_types, mean_function_type
        )

        self.k_mat_inv = None
        self.cholesky_k_mat = None
        self.k_mat = None
        self.partial_derivatives_hyper_params = []
        self.mean_function = mean_function
        self.gradient_mean_function = gradient_mean_function
        self.stochastic_optimizer = stochastic_optimizer
        self.scaler_x = scaler_x
        self.scaler_y = scaler_y
        self.grad_log_evidence_value = None
        self.hyper_params = initial_hyper_params_lst
        self.noise_variance_lower_bound = noise_var_lb
        self.plot_refresh_rate = plot_refresh_rate
        self.kernel_type = kernel_type

    def log_evidence(self):
        """Log evidence/log marginal likelihood of the GP.

        Returns:
            log_evidence (float): Evidence of the GP for current choice of
                                  hyper-parameters
        """
        log_evidence = (
            -0.5 * np.dot(np.dot(self.y_train.T, self.k_mat_inv), self.y_train)
            - (np.sum(np.log(np.diag(self.cholesky_k_mat))))
            - self.k_mat.shape[0] / 2 * np.log(2 * np.pi)
        )
        return log_evidence.flatten()

    def setup(self, x_train, y_train):
        """Setup surrogate model.

        Args:
            x_train (np.array): training inputs
            y_train (np.array): training outputs
        """
        y_train = y_train - self.mean_function(x_train)

        # scale the data
        self.scaler_x.fit(x_train.T)
        self.x_train = self.scaler_x.transform(x_train.T).T
        self.scaler_y.fit(y_train)
        self.y_train = self.scaler_y.transform(y_train)

    def train(self):
        """Train the Gaussian Process.

        Training is conducted by maximizing the evidence/marginal
        likelihood by minimizing the negative log evidence.
        """
        # initialize hyper-parameters and associated linear algebra
        x_0 = np.log(np.array(self.hyper_params))
        hyper_params = self.hyper_params  # Store hyper_params outside the loop

        jitted_kernel, _, _, grad_log_evidence, _, _ = self._get_jitted_objects()
        self._set_jitted_kernel(jitted_kernel)

        _logger.info("Initiating training of the GP model...")

        # set-up stochastic optimizer
        self.stochastic_optimizer.current_variational_parameters = x_0

        def gradient_fn(param_vec):
            return grad_log_evidence(
                param_vec,
                self.y_train,
                self.x_train,
                self.k_mat_inv,
                self.partial_derivatives_hyper_params,
            )

        self.stochastic_optimizer.gradient = gradient_fn

        log_evidence_max = -np.inf
        log_evidence_history = []
        iterations = []
        params_ev_max = None
        k_mat_ev_max = None
        k_mat_inv_ev_max = None
        cholesky_k_mat_ev_max = None
        for params in self.stochastic_optimizer:
            rel_l2_change_params = self.stochastic_optimizer.rel_l2_change
            iteration = self.stochastic_optimizer.iteration

            # update parameters and associated linear algebra
            hyper_params = list(np.exp(params))  # Update hyper_params from stored value
            hyper_params[-1] = np.maximum(hyper_params[-1], self.noise_variance_lower_bound)
            self.hyper_params = hyper_params

            self.grad_log_evidence_value = self.stochastic_optimizer.current_gradient_value

            jitted_kernel, _, _, grad_log_evidence, _, _ = self._get_jitted_objects()
            self._set_jitted_kernel(jitted_kernel)

            self.stochastic_optimizer.gradient = gradient_fn  # Use the captured gradient_fn

            log_evidence = self.log_evidence()

            iterations.append(iteration)
            log_evidence_history.append(log_evidence)

            if self.plot_refresh_rate:
                if iteration % int(self.plot_refresh_rate) == 0:
                    # make some funky gnuplot terminal plots
                    gnuplot_gp_convergence(iterations, log_evidence_history)

                    # Verbose output
                    _logger.info(
                        "Iter %s, parameters %s, gradient log evidence: "
                        "%s, rel L2 change "
                        "%.6f, log-evidence: %s",
                        iteration,
                        params,
                        self.grad_log_evidence_value,
                        rel_l2_change_params,
                        log_evidence,
                    )

            # store the max value for log evidence along with the parameters
            if log_evidence > log_evidence_max:
                log_evidence_max = log_evidence
                params_ev_max = params
                k_mat_ev_max = self.k_mat
                k_mat_inv_ev_max = self.k_mat_inv
                cholesky_k_mat_ev_max = self.cholesky_k_mat

        # use the params that yielded the max log evidence
        if params_ev_max is None:
            raise ValueError("The log evidence was not maximized during training!")

        self.hyper_params = list(np.exp(params_ev_max))
        self.hyper_params[-1] = np.maximum(self.noise_variance_lower_bound, self.hyper_params[-1])

        self.k_mat = k_mat_ev_max
        self.k_mat_inv = k_mat_inv_ev_max
        self.cholesky_k_mat = cholesky_k_mat_ev_max

        _logger.info("GP model trained successfully!")

    def _get_jitted_objects(self):
        """Get the jitted kernel method.

        Get the jitted kernel method as specified in the input file.

        Returns:
            jitted_kernel (obj): Jitted kernel method.
        """
        jitted_kernel = JittedGaussianProcess.valid_kernels_dict.get(self.kernel_type)
        if jitted_kernel is None:
            raise ValueError(
                "You did not specify a valid kernel type in the input file!"
                f"Valid kernel types are {JittedGaussianProcess.valid_kernels_dict.keys()} "
                f"but you specified {self.kernel_type}."
                "Abort..."
            )

        return jitted_kernel

    def _set_jitted_kernel(self, jitted_kernel):
        """Set the inputs for the jitted kernel method.

        Call the jitted kernel method with a set of inputs and
        hyper-parameters and get the resulting matrices and derivatives.

        Args:
            jitted_kernel (obj): Jitted kernel function

        Returns:
            None
        """
        (
            self.k_mat,
            self.cholesky_k_mat,
            self.partial_derivatives_hyper_params,
        ) = jitted_kernel(self.x_train, self.hyper_params)

        # get inverse by solving an equation system with cholesky
        identity = np.eye(self.k_mat.shape[0])
        self.k_mat_inv = cho_solve(
            (self.cholesky_k_mat, True),
            identity,
            check_finite=False,
            overwrite_b=True,
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

    def predict(self, x_test, support="f", gradient_bool=False):
        """Predict the posterior distribution of the trained GP at x_test.

        Args:
            x_test (np.array): Testing matrix for GP with row-wise (vector-valued) testing points
            support (str): Type of support for which the GP posterior is computed; If:

                - 'f':
                    Posterior w.r.t. the latent function f
                - 'y':
                    Latent function is marginalized such that posterior is defined w.r.t. the
                    output y (introduces extra variance)

            gradient_bool (bool, optional): Boolean to configure whether gradients should be
                returned as well

        Returns:
            output (dict): Output dictionary containing the posterior of the GP
        """
        (
            _,
            posterior_mean_fun,
            posterior_covariance_fun,
            _,
            grad_posterior_mean_fun,
            grad_posterior_var_fun,
        ) = self._get_jitted_objects()

        x_test_transformed = self.scaler_x.transform(x_test)
        posterior_mean_test_vec = posterior_mean_fun(
            self.k_mat_inv,
            x_test_transformed,
            self.x_train,
            self.y_train.flatten(),
            self.hyper_params,
        )

        var = posterior_covariance_fun(
            self.k_mat_inv,
            x_test_transformed,
            self.x_train,
            self.hyper_params,
            support,
        )

        if np.any(var.flatten() <= 0.0):
            raise ValueError(
                "Posterior variance has negative values! It seems like the condition of your "
                "covariance matrix is rather bad. Please increase the noise variance lower bound!"
                f"Your current noise variance lower bound is: {self.noise_variance_lower_bound}."
            )

        output = {"x_test": x_test}
        output["result"] = self.scaler_y.inverse_transform_mean(posterior_mean_test_vec).reshape(
            -1, 1
        ) + self.mean_function(x_test)
        output["variance"] = (self.scaler_y.inverse_transform_std(np.sqrt(var)) ** 2).reshape(-1, 1)

        if gradient_bool:
            grad_post_mean_test_mat = grad_posterior_mean_fun(
                self.k_mat_inv,
                x_test_transformed,
                self.x_train,
                self.y_train,
                self.hyper_params,
            )
            grad_post_var_test_vec = grad_posterior_var_fun(
                self.k_mat_inv,
                x_test_transformed,
                self.x_train,
                self.hyper_params,
            )
            output["grad_mean"] = self.scaler_y.inverse_transform_grad_mean(
                grad_post_mean_test_mat, self.scaler_x.standard_deviation
            ) + self.gradient_mean_function(x_test)
            output["grad_var"] = self.scaler_y.inverse_transform_grad_var(
                grad_post_var_test_vec,
                var,
                output["variance"],
                self.scaler_x.standard_deviation,
            )

        return output

    def get_state(self):
        """Get the current hyper-parameters of the model.

        Returns:
            state_dict (dict): Dictionary with the current state settings
            of the probabilistic mapping object
        """
        state_dict = {
            "hyper_params_lst": self.hyper_params,
            "k_mat": self.k_mat,
            "k_mat_inv": self.k_mat_inv,
            "cholesky_k_mat": self.cholesky_k_mat,
        }
        return state_dict

    def set_state(self, state_dict):
        """Update and set new hyper-parameters for the model.

        Args:
            state_dict (dict): Dictionary with the current state settings
                               of the probabilistic mapping object
        """
        # conduct some checks
        valid_keys = [
            "hyper_params_lst",
            "k_mat",
            "k_mat_inv",
            "cholesky_k_mat",
        ]

        keys = list(state_dict.keys())
        if keys != valid_keys:
            raise ValueError("The provided dictionary does not contain valid keys!")

        # Actually set the new state of the object
        self.hyper_params = state_dict["hyper_params_lst"]
        self.k_mat = state_dict["k_mat"]
        self.k_mat_inv = state_dict["k_mat_inv"]
        self.cholesky_k_mat = state_dict["cholesky_k_mat"]

    @staticmethod
    def zero_mean_fun(_samples):
        """Return zero mean function."""
        return 0

    @staticmethod
    def gradient_zero_mean_fun(_samples):
        """Return gradient of zero mean function."""
        return 0

    @staticmethod
    def identity_multi_fidelity_mean_fun(samples):
        """Return identity mean function for multi-fidelity."""
        mean = np.atleast_2d(samples[:, 0]).T
        return mean

    @staticmethod
    def gradient_identity_multi_fidelity_mean_fun(samples):
        """Return gradient of mean function for multi-fidelity."""
        grad_mean = np.atleast_2d(np.ones(samples.shape[0])).T
        return grad_mean
