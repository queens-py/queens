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
"""GPLogpdf model."""

import logging
import time
from functools import partial

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
import tensorflow_probability.substrates.jax as tfp
from jax import jit, vmap
from scipy import stats

from queens.models._model import Model
from queens.utils import jax_minimize_wrapper
from queens.utils.gpflow_transformations import init_scaler
from queens.utils.numpy_linalg import safe_cholesky

_logger = logging.getLogger(__name__)
jax.config.update("jax_enable_x64", True)


class LogpdfGaussianProcess(Model):
    """LogpdfGP Class.

    Attributes:
        approx_type (str): Approximation type (GPMAP-I', 'CGPMAP-II' or 'CFBGP')
        num_hyper (int): Number of hyperparameter samples (if CFBGP)
        num_optimizations (int): Number of hyperparameter optimization restarts
        hmc_burn_in (int): Number of HMC burn-in steps (if CFBGP)
        hmc_steps (int): Number of HMC steps (if CFBGP)
        prior_rate (np.ndarray): Rates of exponential priors for hyperparameters
                                      [lengthscales, signal_std, noise_var]
        prior_gp_mean (float): Transformed prior GP mean. Range: [-1, 0]
        upper_bound (float): Transformed upper bound for Gaussian process. Range: [-1, 0].
                                  If None provided, it is derived from the number of
                                  observations.
        quantile (float): Confidence quantile
        jitter (float): Nugget term for numerical stability of Cholesky decomposition
        x_train (np.ndarray): Training input samples
        y_train (np.ndarray): Training likelihood output samples
        scaler_x (float): Scaler for training input samples
        scaler_y (float): Scaler for training likelihood output samples
        num_dim (np.ndarray): number of dimensions
        hyperparameters (np.ndarray): Hyperparameter (samples)
        chol_k_train_train (np.ndarray): Cholesky decomposition of Gram matrix evaluated at the
                                         training samples
        v_train (np.ndarray): Matrix product of inverse of Gram matrix evaluated at training
                              samples and training output samples
        jit_func_generate_output (obj): Jitted partial 'generate_output_*' function
        partial_hyperparameter_log_prob (obj): Jitted partial function of hyperparameter log
                                               posterior probability
        batch_size (int): Batch size for concurrent prediction evaluations
    """

    def __init__(
        self,
        approx_type,
        num_hyper=100,
        num_optimizations=3,
        hmc_burn_in=1000,
        hmc_steps=2000,
        prior_rate=None,
        prior_gp_mean=-1.0,
        upper_bound=None,
        quantile=0.9,
        jitter=1.0e-16,
    ):
        """Initialize LogpdfGP.

        Args:
            approx_type (str): Approximation type (GPMAP-I', 'CGPMAP-II' or 'CFBGP')
            num_hyper (int, opt): Number of hyperparameter samples (if CFBGP)
            num_optimizations (int, opt): Number of hyperparameter optimization restarts
            hmc_burn_in (int, opt): Number of HMC burn-in steps (if CFBGP)
            hmc_steps (int, opt): Number of HMC steps (if CFBGP)
            prior_rate (array-like, opt): Rates of exponential priors for hyperparameters
                                          [lengthscales, signal_std, noise_var]
            prior_gp_mean (float, opt): Transformed prior GP mean. Range: [-1, 0]
            upper_bound (float, opt): Transformed upper bound for Gaussian process. Range: [-1, 0].
                                      If None provided, it is derived from the number of
                                      observations.
            quantile (float, opt): Confidence quantile
            jitter (float, opt): Nugget term for numerical stability of Cholesky decomposition
        """
        if approx_type not in ["GPMAP-I", "CGPMAP-II", "CFBGP"]:
            raise ValueError(f"Invalid approximation type: {approx_type}")
        self.approx_type = approx_type
        self.num_hyper = num_hyper
        self.num_optimizations = num_optimizations
        self.hmc_burn_in = hmc_burn_in
        self.hmc_steps = hmc_steps
        self.prior_rate = np.array(prior_rate)
        if prior_rate is None:
            self.prior_rate = np.array([1e-1, 10, 1e8])
        self.prior_gp_mean = prior_gp_mean
        self.upper_bound = upper_bound
        self.quantile = quantile
        self.jitter = jitter
        self.x_train = None
        self.y_train = None
        self.scaler_x = None
        self.scaler_y = None
        self.num_dim = None
        self.hyperparameters = None
        self.chol_k_train_train = None
        self.v_train = None
        self.jit_func_generate_output = None
        self.partial_hyperparameter_log_prob = None
        self.batch_size = int(4e8)

        super().__init__()

    def initialize(self, x_train, y_train, num_observations):
        """Initialize Gaussian process model.

        Args:
            x_train (np.ndarray): Training input samples
            y_train (np.ndarray): Training likelihood output samples
            num_observations (int): Number of observations
        """
        x_train = x_train.reshape(y_train.size, -1)
        y_train = y_train.reshape(-1, 1)

        self.num_dim = x_train.shape[1]
        self.scaler_x, self.x_train = init_scaler(x_train)
        self.scaler_y = np.max(np.abs(y_train))
        self.y_train = y_train / self.scaler_y - self.prior_gp_mean
        if self.upper_bound is None:
            self.upper_bound = -0.5 * stats.chi2(num_observations).ppf(0.05)
        self.upper_bound = np.array(max(y_train.max(), self.upper_bound))
        _logger.info(
            "Upper bound: %f,  y_min: %f, y_max: %f, num_train: %i",
            self.upper_bound,
            y_train.min(),
            y_train.max(),
            y_train.size,
        )

        self.partial_hyperparameter_log_prob = jit(
            partial(
                self.hyperparameter_log_prob,
                x_train=self.x_train,
                y_train=self.y_train,
                jitter=self.jitter,
                log_likelihood_func=self.hyperparameter_log_likelihood,
                log_prior_func=self.hyperparameter_log_prior,
                prior_rate=self.prior_rate,
            )
        )
        if self.approx_type == "CFBGP":
            self.batch_size = int(4e8 / (y_train.size * self.num_dim * self.num_hyper))
            with jax.default_device(jax.devices("cpu")[0]):
                hyperparameters = self.sample_hyperparameters()
            _logger.info(
                "Hyperparameters mean: %s, Hyperparameters std: %s",
                np.mean(hyperparameters, axis=0),
                np.std(hyperparameters, axis=0),
            )
            index_choice = np.random.choice(
                np.arange(0, hyperparameters.shape[0]), self.num_hyper, replace=False
            )
            self.hyperparameters = hyperparameters[index_choice]

            self.chol_k_train_train = np.zeros(
                (self.num_hyper, self.y_train.size, self.y_train.size)
            )
            self.v_train = np.zeros((self.num_hyper, self.y_train.size, 1))
            for i, hyperparameter in enumerate(self.hyperparameters):
                self.chol_k_train_train[i], self.v_train[i] = self.calc_train_factor(hyperparameter)
        else:
            self.batch_size = int(4e8 / (y_train.size * self.num_dim))
            with jax.default_device(jax.devices("cpu")[0]):
                self.hyperparameters = self.optimize_hyperparameters()
            self.chol_k_train_train, self.v_train = self.calc_train_factor(self.hyperparameters)

        eval_mean_and_std = partial(
            self.evaluate_mean_and_std,
            prior_gp_mean=self.prior_gp_mean,
            scaler_y=self.scaler_y,
        )

        if self.approx_type in ["CFBGP", "CGPMAP-II"]:
            if self.approx_type == "CFBGP":
                eval_mean_and_std = vmap(eval_mean_and_std, in_axes=(None, 0, 0, 0))
                generate_output_func = self.generate_output_cfbgp
            else:
                generate_output_func = self.generate_output_cgpmap_2

            self.jit_func_generate_output = jit(
                partial(
                    generate_output_func,
                    x_train=self.x_train,
                    hyperparameters=self.hyperparameters,
                    v_train=self.v_train,
                    chol_k_train_train=self.chol_k_train_train,
                    upper_bound=self.upper_bound,
                    eval_mean_and_std=eval_mean_and_std,
                    quantile=self.quantile,
                )
            )

        else:
            self.jit_func_generate_output = jit(
                partial(
                    self.generate_output_gpmap_1,
                    x_train=self.x_train,
                    hyperparameters=self.hyperparameters,
                    v_train=self.v_train,
                    prior_gp_mean=self.prior_gp_mean,
                    scaler_y=self.scaler_y,
                )
            )

    def calc_train_factor(self, hyperparameters):
        """Calculate training factors.

        Args:
            hyperparameters (np.ndarray): Hyperparameters

        Returns:
            chol_k_train_train (np.ndarray): Cholesky decomposition of Gram matrix evaluated at the
                                             training samples
            v_train (np.ndarray): Matrix product of inverse of Gram matrix evaluated at training
                                  samples and training output samples
        """
        k_train_train = rbf(self.x_train, self.x_train, hyperparameters[:-1])
        k_train_train += jnp.eye(k_train_train.shape[0]) * hyperparameters[-1]
        k_train_train = k_train_train + jnp.eye(k_train_train.shape[0]) * self.jitter
        chol_k_train_train = safe_cholesky(k_train_train, hyperparameters[-1])
        v_train = jnp.linalg.solve(
            chol_k_train_train.T, jnp.linalg.solve(chol_k_train_train, self.y_train)
        )
        return chol_k_train_train, v_train

    def _evaluate(self, samples):
        """Evaluate model with current set of input samples.

        Args:
            samples (np.ndarray): Input samples

        Returns:
            log_likelihood (np.ndarray): Approximation of log-likelihood values at input samples
        """
        x_test = self.scaler_x.transform(samples.reshape(-1, self.num_dim))
        log_likelihood = np.zeros(x_test.shape[0])
        lower_index = 0
        upper_index = min(self.batch_size, x_test.shape[0])
        while lower_index < x_test.shape[0]:
            x_batch = x_test[lower_index:upper_index]
            # pylint: disable-next=not-callable
            log_likelihood[lower_index:upper_index] = self.jit_func_generate_output(x_batch)
            lower_index = upper_index
            upper_index = min(upper_index + self.batch_size, x_test.shape[0])
        log_likelihood = np.array(log_likelihood)
        return {"result": log_likelihood}

    def grad(self, samples, upstream_gradient):
        """Evaluate gradient of model w.r.t.

        current set of input samples.
        """
        raise NotImplementedError

    def optimize_hyperparameters(self):
        """Optimize hyperparameters.

        Returns:
            hyperparameters (np.ndarray): Optimized hyperparameters
        """

        def loss(x):
            return -self.partial_hyperparameter_log_prob(x)  # pylint: disable=not-callable

        initial_samples = np.random.exponential(
            scale=1.0e0, size=(self.num_optimizations, self.num_dim + 2)
        )
        initial_samples[:, :-2] = initial_samples[:, :-2] / self.prior_rate[0]
        initial_samples[:, -2] = initial_samples[:, -2] / self.prior_rate[1]
        initial_samples[:, -1] = initial_samples[:, -1] / self.prior_rate[2]
        initial_samples_unconstrained = np.log(initial_samples)
        loss(initial_samples_unconstrained[0])
        jax.grad(loss)(initial_samples_unconstrained[0])
        start = time.time()
        positions = np.zeros((self.num_optimizations, self.num_dim + 2))
        objectives = np.zeros(self.num_optimizations)
        for i in range(self.num_optimizations):
            result = jax_minimize_wrapper.minimize(
                loss, initial_samples_unconstrained[i], "L-BFGS-B"
            )
            positions[i] = result["x"]
            objectives[i] = result["fun"]
        _logger.info("Optimization Time: %f s", time.time() - start)
        _logger.info("Optimized Loss Value: %f", np.nanmin(np.array(objectives)))
        _logger.info(
            "Number of failed optimizations: %i / %i",
            np.sum(np.isnan(np.array(objectives))),
            self.num_optimizations,
        )
        hyperparameters = np.exp(positions[np.nanargmin(objectives)])
        _logger.info("Optimized hyperparameters: %s", hyperparameters)
        return hyperparameters

    def sample_hyperparameters(self):
        """Draw samples from hyperparameter posterior.

        Returns:
            hyperparameters (np.ndarray): Samples of hyperparameter posterior
        """
        initial_hyperparameters = self.optimize_hyperparameters()

        nuts = tfp.mcmc.NoUTurnSampler(
            target_log_prob_fn=self.partial_hyperparameter_log_prob,
            max_tree_depth=10,
            step_size=0.01,
        )
        adaptive_nuts = tfp.mcmc.SimpleStepSizeAdaptation(
            nuts,
            num_adaptation_steps=int(0.8 * self.hmc_burn_in),
            target_accept_prob=jnp.float64(0.75),
            adaptation_rate=0.1,
        )

        _, sample_key = jax.random.split(jax.random.PRNGKey(0))

        @jit
        def run_chain_fn(initial_state):
            return tfp.mcmc.sample_chain(
                num_results=self.hmc_steps,
                num_burnin_steps=self.hmc_burn_in,
                current_state=initial_state,
                kernel=adaptive_nuts,
                trace_fn=None,
                return_final_kernel_results=False,
                seed=sample_key,
            )

        start = time.time()
        hyperparameter_samples = run_chain_fn(np.log(initial_hyperparameters))
        _logger.info("Sampling Time: %f s", time.time() - start)
        hyperparameter_samples = np.exp(hyperparameter_samples)
        return hyperparameter_samples

    @staticmethod
    def evaluate_mean_and_std(
        dists_test_train,
        hyperparameters,
        v_train,
        chol_k_train_train,
        prior_gp_mean,
        scaler_y,
    ):
        """Mean and standard deviation of unconstrained GP at test samples.

        Args:
            dists_test_train (np.ndarray): Distance Matrix between test and train samples
            hyperparameters (np.ndarray): Hyperparameters
            v_train (np.ndarray): Matrix product of inverse of Gram matrix evaluated at training
                                  samples and training output samples
            chol_k_train_train (np.ndarray): Cholesky decomposition of Gram matrix evaluated at the
                                             training samples
            prior_gp_mean (float, opt): Transformed prior GP mean. Range: [-1, 0]
            scaler_y (float): Scaler for training likelihood output samples

        Returns:
            mean (np.ndarray): Mean of unconstrained GP at test samples
            std (np.ndarray): Standard deviation of unconstrained GP at test samples.
        """
        k_test_train = rbf_by_dists(dists_test_train, hyperparameters[:-1])
        k_test_test = hyperparameters[-2] ** 2
        mean = jnp.dot(k_test_train, v_train)
        var = k_test_test - jnp.sum(
            jsp.linalg.solve_triangular(chol_k_train_train, k_test_train.T, lower=True) ** 2,
            axis=0,
        )
        var = var * scaler_y**2
        std = var ** (1 / 2)  # + 1e-30
        std = std.reshape(-1)
        mean = (mean.reshape(-1) + prior_gp_mean) * scaler_y
        return mean, std

    @staticmethod
    def generate_output_cgpmap_2(
        x_test,
        x_train,
        hyperparameters,
        v_train,
        chol_k_train_train,
        eval_mean_and_std,
        upper_bound,
        quantile,
    ):
        """Approximation of log-likelihood using CGPMAP-II approach.

        Args:
            x_test (np.ndarray): Input test samples
            x_train (np.ndarray): Training input samples
            hyperparameters (np.ndarray): Hyperparameters
            v_train (np.ndarray): Matrix product of inverse of Gram matrix evaluated at training
                                  samples and training output samples
            chol_k_train_train (np.ndarray): Cholesky decomposition of Gram matrix evaluated at the
                                             training samples
            eval_mean_and_std (obj): eval_mean_and_std function
            upper_bound (float): Transformed upper bound for Gaussian process
            quantile (float): Confidence quantile

        Returns:
            log_likelihood (np.ndarray): Approximation of log-likelihood values at x_test
        """
        dists_test_train = distances(x_test, x_train)
        mean, std = eval_mean_and_std(
            dists_test_train, hyperparameters, v_train, chol_k_train_train
        )

        erfc_arg = (mean - upper_bound) / (jnp.sqrt(2) * std)
        erfc_term = jsp.special.erfc(erfc_arg)
        log_likelihood = jnp.where(
            erfc_arg < 5, jsp.special.erfinv(quantile * erfc_term - 1), -erfc_arg
        )
        log_likelihood = log_likelihood * jnp.sqrt(2) * std + mean
        log_likelihood = jnp.clip(log_likelihood, min=None, max=upper_bound)
        return log_likelihood

    @staticmethod
    def generate_output_cfbgp(
        x_test,
        x_train,
        hyperparameters,
        v_train,
        chol_k_train_train,
        eval_mean_and_std,
        upper_bound,
        quantile,
    ):
        """Approximation of log-likelihood using CGPMAP-II approach.

        Args:
            x_test (np.ndarray): Input test samples
            x_train (np.ndarray): Training input samples
            hyperparameters (np.ndarray): Hyperparameters
            v_train (np.ndarray): Matrix product of inverse of Gram matrix evaluated at training
                                  samples and training output samples
            chol_k_train_train (np.ndarray): Cholesky decomposition of Gram matrix evaluated at the
                                             training samples
            eval_mean_and_std (obj): eval_mean_and_std function
            upper_bound (float): Transformed upper bound for Gaussian process
            quantile (float): Confidence quantile

        Returns:
            log_likelihood (np.ndarray): Approximation of log-likelihood values at x_test
        """
        dists_test_train = distances(x_test, x_train)
        mean, std = eval_mean_and_std(
            dists_test_train, hyperparameters, v_train, chol_k_train_train
        )
        mean = mean.reshape(-1, x_test.shape[0])
        std = std.reshape(-1, x_test.shape[0])

        def objective(log_like, m, s, denominator):
            numerators = jsp.special.erfc((m - log_like) / (jnp.sqrt(2) * s))
            return 1 / m.size * jnp.sum(numerators / denominator) - quantile

        def eval_log_like(m, s):
            denominator = jsp.special.erfc((m - upper_bound) / (jnp.sqrt(2) * s))
            objective_func = partial(objective, m=m, s=s, denominator=denominator)
            solution = tfp.math.find_root_chandrupatla(
                objective_func, -1e12, upper_bound, position_tolerance=1e-4, max_iterations=1e4
            )
            return solution.estimated_root

        log_likelihood = jax.vmap(eval_log_like, in_axes=(1, 1))(mean, std)
        return log_likelihood

    @staticmethod
    def generate_output_gpmap_1(x_test, x_train, hyperparameters, v_train, prior_gp_mean, scaler_y):
        """Approximation of log-likelihood using CGPMAP-II approach.

        Args:
            x_test (np.ndarray): Input test samples
            x_train (np.ndarray): Training input samples
            hyperparameters (np.ndarray): Hyperparameters
            v_train (np.ndarray): Matrix product of inverse of Gram matrix evaluated at training
                                  samples and training output samples
            prior_gp_mean (float, opt): Transformed prior GP mean
            scaler_y (float): Scaler for training likelihood output samples

        Returns:
            log_likelihood (np.ndarray): Approximation of log-likelihood values at x_test
        """
        k_test_train = rbf(x_test, x_train, hyperparameters[:-1])
        log_likelihood = (jnp.dot(k_test_train, v_train).reshape(-1) + prior_gp_mean) * scaler_y
        return log_likelihood

    @staticmethod
    def hyperparameter_log_likelihood(hyperparameters, x_train, y_train, jitter):
        """Log likelihood function for hyperparameters.

        Args:
            hyperparameters (np.ndarray): Hyperparameters
            x_train (np.ndarray): Training input samples
            y_train (np.ndarray): Training output samples
            jitter (float): Nugget term for numerical stability of Cholesky decomposition

        Returns:
            log_likelihood (np.ndarray): Log likelihood of data given hyperparameters
        """
        k_train_train = rbf(x_train, x_train, hyperparameters[:-1])
        k_train_train = k_train_train + jnp.eye(k_train_train.shape[0]) * hyperparameters[-1]
        k_train_train = k_train_train + jnp.eye(k_train_train.shape[0]) * jitter
        v_train = jsp.linalg.solve(k_train_train, y_train, assume_a="pos")
        logdet = jnp.linalg.slogdet(k_train_train)[1]
        log_likelihood = -0.5 * (jnp.sum(y_train * v_train) + logdet)
        log_likelihood = log_likelihood - v_train.size / 2 * jnp.log(2 * jnp.pi)
        return log_likelihood

    @staticmethod
    def hyperparameter_log_prior(hyperparameters, prior_rate):
        """Log prior function for hyperparameters.

        Args:
            hyperparameters (np.ndarray): Hyperparameters
            prior_rate (np.ndarray): prior_rate (np.ndarray): Rates of exponential priors for
                                                              hyperparameters

        Returns:
            joint_log_prior (np.ndarray): Joint log prior of hyperparameters
        """
        # Exponential distribution for lengthscales and signal std
        log_prior = jnp.log(prior_rate[0]) - prior_rate[0] * hyperparameters[:-2]
        joint_log_prior = jnp.sum(log_prior)
        joint_log_prior += jnp.log(prior_rate[1]) - prior_rate[1] * hyperparameters[-2]
        joint_log_prior += jnp.log(prior_rate[2]) - prior_rate[2] * hyperparameters[-1]
        return joint_log_prior

    @staticmethod
    def hyperparameter_log_prob(
        transformed_hyperparameters,
        x_train,
        y_train,
        jitter,
        log_likelihood_func,
        log_prior_func,
        prior_rate,
    ):
        """Log joint probability function for hyperparameters.

        Args:
            transformed_hyperparameters (np.ndarray): Transformed hyperparameters
            x_train (np.ndarray): Training input samples
            y_train (np.ndarray): Training output samples
            jitter (float): Nugget term for numerical stability of Cholesky decomposition
            log_likelihood_func (obj): Log likelihood function of hyperparameters
            log_prior_func (obj): log prior function of hyperparameters
            prior_rate (np.ndarray): prior_rate (np.ndarray): Rates of exponential priors for
                                                              hyperparameters

        Returns:
            np.ndarray: Log joint probability of hyperparameters
        """
        hyperparameters = jnp.exp(transformed_hyperparameters)
        log_likelihood = log_likelihood_func(hyperparameters, x_train, y_train, jitter)
        log_prior = log_prior_func(hyperparameters, prior_rate)
        forward_log_det = jnp.sum(transformed_hyperparameters)
        return log_likelihood + log_prior + forward_log_det


def distances(x1, x2):
    """Distance Matrix between two sample sets.

    Args:
         x1 (np.ndarray): input samples
         x2 (np.ndarray): input samples

    Returns:
         dists (np.ndarray): Distance Matrix
    """
    num_dim = x1.shape[1]
    dists = x1.reshape(-1, 1, num_dim) - x2.reshape(1, -1, num_dim)
    return dists


def rbf_by_dists(dists, hyperparameters):
    """Gram Matrix of RBF Kernel.

    Args:
         dists (np.ndarray): Distance Matrix
         hyperparameters (np.ndarray): hyperparameters of kernel

    Returns:
         k_x1_x2 (np.ndarray): Gram Matrix of RBF Kernel
    """
    lengthscales = hyperparameters[:-1]
    signal_std = hyperparameters[-1]
    dist_norm = jnp.sum(dists**2 / (2 * lengthscales**2), axis=2)
    k_x1_x2 = signal_std**2 * jnp.exp(-dist_norm)
    return k_x1_x2


def rbf(x1, x2, hyperparameters):
    """Gram Matrix of RBF Kernel.

    Args:
         x1 (np.ndarray): input samples
         x2 (np.ndarray): input samples
         hyperparameters (np.ndarray): hyperparameters of kernel

    Returns:
         k_x1_x2 (np.ndarray): Gram Matrix of RBF Kernel
    """
    dists = distances(x1, x2)
    k_x1_x2 = rbf_by_dists(dists, hyperparameters)
    return k_x1_x2
