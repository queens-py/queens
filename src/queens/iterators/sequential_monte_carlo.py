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
"""Sequential Monte Carlo algorithm.

References:
    [1]: Del Moral, P., Doucet, A. and Jasra, A. (2007)
        ‘Sequential monte carlo for bayesian computation’,
        in Bernardo, J. M. et al. (eds) Bayesian Statistics 8.
        Oxford University Press, pp. 1–34.
    [2]: Koutsourelakis, P. S. (2009)
        ‘A multi-resolution, non-parametric, Bayesian framework for
        identification of spatially-varying model parameters’,
        Journal of Computational Physics, 228(17), pp. 6184–6211.
        doi: 10.1016/j.jcp.2009.05.016.
    [3]: Minson, S. E., Simons, M. and Beck, J. L. (2013)
        ‘Bayesian inversion for finite fault earthquake source models
        I-theory and algorithm’,
        Geophysical Journal International, 194(3), pp. 1701–1726.
        doi: 10.1093/gji/ggt180.
    [4]: Del Moral, P., Doucet, A. and Jasra, A. (2006)
        ‘Sequential Monte Carlo samplers’,
        Journal of the Royal Statistical Society.
        Series B: Statistical Methodology.
        Blackwell Publishing Ltd, 68(3), pp. 411–436.
        doi: 10.1111/j.1467-9868.2006.00553.x.
"""

import logging

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import scipy

from queens.iterators._iterator import Iterator
from queens.iterators.metropolis_hastings import MetropolisHastings
from queens.utils import sequential_monte_carlo as smc_utils
from queens.utils.logger_settings import log_init_args
from queens.utils.process_outputs import process_outputs, write_results

_logger = logging.getLogger(__name__)


class SequentialMonteCarlo(Iterator):
    """Iterator based on Sequential Monte Carlo algorithm.

    The Sequential Monte Carlo algorithm is a very general algorithm for
    sampling from complex, intractable probability distributions from
    which direct sampling is difficult or impossible.
    The implemented version is based on [1, 2, 3, 4].

    Attributes:
        plot_trace_every (int): Print the current trace every *plot_trace_every*-th iteration.
                                Default: 0 (do not print the trace).
        result_description (dict): Description of desired results.
        seed (int): Seed for random number generator.
        mcmc_kernel (MetropolisHastings): Forward kernel for the rejuvenation steps.
        num_particles (int): Number of particles.
        num_variables (int): Number of primary variables.
        particles (ndarray): Array holding the current particles.
        weights (ndarray): Array holding the current weights.
        log_likelihood (ndarray): log of pdf of likelihood at particles.
        log_prior (ndarray): log of pdf of prior at particles.
        log_posterior (ndarray): log of pdf of posterior at particles.
        ess (list): List storing the values of the effective sample size.
        ess_cur (float): Current effective sample size.
        temper (function): Tempering function that defines the transition to the goal
                           distribution.
        gamma_cur (float): Current tempering parameter, sometimes called (reciprocal) temperature.
        gammas (list): List to store values of the tempering parameter.
        a (float): Parameter for the scaling of the covariance matrix of the proposal distribution
                   of the MCMC kernel.
        b (float): Parameter for the scaling of the covariance matrix of the proposal distribution
                   of the MCMC kernel.
    """

    @log_init_args
    def __init__(
        self,
        model,
        parameters,
        global_settings,
        num_particles,
        result_description,
        seed,
        temper_type,
        mcmc_proposal_distribution,
        num_rejuvenation_steps,
        plot_trace_every=0,
    ):
        """Initialize the SequentialMonteCarlo class.

        Args:
            model (Model): Model to be evaluated by iterator
            parameters (Parameters): Parameters object
            global_settings (GlobalSettings): settings of the QUEENS experiment including its name
                                              and the output directory
            num_particles (int): Number of particles
            result_description (dict): Description of desired results.
            seed (int): Seed for random number generator.
            temper_type (str): Temper type ('bayes' or 'generic')
            mcmc_proposal_distribution (obj): Proposal distribution for the mcmc kernel
            num_rejuvenation_steps (int): number of samples per rejuvenation
            plot_trace_every (int): Print the current trace every *plot_trace_every*-th iteration.
                                    Default: 0 (do not print the trace).
        """
        super().__init__(model, parameters, global_settings)

        self.mcmc_kernel = MetropolisHastings(
            model=model,
            parameters=parameters,
            global_settings=global_settings,
            result_description=None,
            proposal_distribution=mcmc_proposal_distribution,
            num_samples=num_rejuvenation_steps,
            seed=seed,
            tune=False,
            tune_interval=100,
            scale_covariance=1.0,
            num_burn_in=0,
            num_chains=num_particles,
            as_smc_rejuvenation_step=True,
            temper_type=temper_type,
        )

        self.plot_trace_every = plot_trace_every
        self.result_description = result_description
        self.seed = seed

        self.num_particles = num_particles

        # check agreement of dimensions of proposal distribution and
        # parameter space
        self.num_variables = self.parameters.num_parameters

        if self.num_variables is not self.mcmc_kernel.proposal_distribution.dimension:
            raise ValueError(
                "Dimensions of MCMC kernel proposal"
                " distribution and parameter space"
                " do not agree."
            )

        # init particles (location, weights, posterior)
        self.particles = np.zeros((self.num_particles, self.num_variables))
        # TODO: use normalised weights? # pylint: disable=fixme
        self.weights = np.ones((self.num_particles, 1))

        self.log_likelihood = np.zeros(self.num_particles)
        self.log_prior = np.zeros(self.num_particles)
        self.log_posterior = np.zeros(self.num_particles)

        self.ess = []
        self.ess_cur = 0.0

        self.temper = smc_utils.temper_factory(temper_type)

        # tempering parameter (linked to counter/ time index)
        self.gamma_cur = 0.0
        self.gammas = []

        # parameters for the scaling of the covariance matrix
        # values of a and b are taken from [3] p.1706
        self.a = 1.0 / 9.0
        self.b = 8.0 / 9.0

    def eval_log_prior(self, sample_batch):
        """Evaluate natural logarithm of prior at sample.

        Args:
            sample_batch (np.array): Array of input samples

        Returns:
            log_prior_array (np.array): Array of log-prior values for input samples
        """
        return self.parameters.joint_logpdf(sample_batch)

    def eval_log_likelihood(self, sample_batch):
        """Evaluate natural logarithm of likelihood at sample batch.

        Args:
            sample_batch (np.array): Batch of samples

        Returns:
            log_likelihood (np.array): Logarithm of likelihood for the sample batch.
        """
        log_likelihood = self.model.evaluate(sample_batch)["result"]
        return log_likelihood

    def pre_run(self):
        """Draw initial sample."""
        _logger.info("Initialize run.")
        np.random.seed(self.seed)

        # draw initial particles from prior distribution
        self.particles = self.parameters.draw_samples(self.num_particles)
        self.log_likelihood = self.eval_log_likelihood(self.particles)
        self.log_prior = self.eval_log_prior(self.particles)
        self.log_posterior = self.log_likelihood + self.log_prior
        # initialize importance weights
        self.weights = np.ones(self.num_particles)
        self.ess_cur = self.num_particles
        self.ess.append(self.ess_cur)

        self.gamma_cur = 0.0
        self.gammas.append(self.gamma_cur)

        if self.plot_trace_every:
            self.draw_trace(0)

    def calc_new_weights(self, gamma_new, gamma_old):
        """Calculate the weights at new gamma value.

        This is a core equation of the SMC algorithm. See for example:

        - Eq.(22) with Eq.(14) in [1]
        - Table 1: (2) in [2]
        - Eq.(31) with Eq.(11) in [4]

        We use the exp-log trick here to avoid numerical problems and normalize the
        particles in this method.

        Args:
            gamma_new (float): Old value of gamma blending parameter
            gamma_old (float): New value of gamma blending parameter

        Returns:
            weights_new (np.array): New and normalized weights
        """
        weights_scaling = self.temper(self.log_prior, self.log_likelihood, gamma_new) - self.temper(
            self.log_prior, self.log_likelihood, gamma_old
        )
        a_norm = np.max(weights_scaling)

        log_weights_new = np.log(self.weights) + weights_scaling

        log_normalizer = a_norm + np.log(np.sum(self.weights * np.exp(weights_scaling - a_norm)))
        log_weights_new_normalized = log_weights_new - log_normalizer
        weights_new = np.exp(log_weights_new_normalized)

        return weights_new

    def calc_new_ess(self, gamma_new, gamma_old):
        """Calculate predicted Effective Sample Size at *gamma_new*.

        Args:
            gamma_new (float): New gamma value.
            gamma_old (float): Previous gamma value.

        Returns:
            ess (float): Effective sample size for the new gamma value.
        """
        weights_new = self.calc_new_weights(gamma_new, gamma_old)
        ess = smc_utils.calc_ess(weights_new)
        return ess

    def calc_new_gamma(self, gamma_cur):
        """Calculate the new gamma value.

        Based on the current gamma, calculate the new gamma such that
        the ESS at the new gamma is equal to zeta times current gamma.
        This ensures only a small reduction of the ESS.

        Args:
            gamma_cur (float): Current gamma value.

        Returns:
            gamma_new (float): Updated gamma value.
        """
        zeta = 0.95

        def f(gamma_new):
            ess_new = self.calc_new_ess(gamma_new, gamma_cur)

            f = ess_new - zeta * self.ess_cur
            return f

        if f(1.0) >= 0:
            return 1.0
        # TODO: adjust accuracy - method, xtol, rtol, maxiter # pylint: disable=fixme
        # TODO: new ESS is already calculated in this step # pylint: disable=fixme
        search_interval = [gamma_cur, 1.0]
        root_result = scipy.optimize.root_scalar(f, bracket=search_interval, method="toms748")
        gamma_new = root_result.root
        return gamma_new

    def update_ess(self, resampled=False):
        """Update effective sample size (ESS) and store current value.

        Based on the current weights, calculate the corresponding ESS.
        Store the new ESS value.
        In case of resampling, the weights have been reset in the
        current time step and therefore also the ESS has to be reset.

        Args:
            resampled (bool): Indicates whether current weights
                                 are base on a resampling step
        """
        self.ess_cur = smc_utils.calc_ess(self.weights)

        if resampled:
            self.ess[-1] = self.ess_cur
        else:
            self.ess.append(self.ess_cur)

    def update_gamma(self, gamma_new):
        """Update the current gamma value and store old value.

        Args:
            gamma_new (float): New gamma value to update.
        """
        self.gamma_cur = gamma_new
        self.gammas.append(self.gamma_cur)

    def update_weights(self, weights_new):
        """Update the weights to their new values.

        Args:
            weights_new (np.array): New weights for the particles.
        """
        self.weights = weights_new

    def resample(self):
        """Resample particle distribution based on their weights.

        Resampling reduces the variance of the particle approximation by
        eliminating particles with small weights and duplicating
        particles with large weights (see 2.2.1 in [2]).

        Returns:
            Tuple of updated particles, resampled weights, log-likelihood, and log-prior.
        """
        # draw from multinomial distribution to decide
        # the frequency of individual particles
        particle_freq = np.random.multinomial(self.num_particles, self.weights)

        idx_list = []
        for idx, freq in enumerate(particle_freq):
            idx_list += [idx] * freq

        resampled_weights = np.ones(self.num_particles)

        return (
            self.particles[idx_list],
            resampled_weights,
            self.log_likelihood[idx_list],
            self.log_prior[idx_list],
        )

    def core_run(self):
        """Core run of Sequential Monte Carlo iterator."""
        _logger.info("Welcome to SMC core run.")
        # counter
        step = 0
        # average accept rate of MCMC kernel
        avg_accept = 1.0
        while self.gamma_cur < 1:
            step += 1

            # Adapt step size
            self.update_gamma(self.calc_new_gamma(self.gamma_cur))

            # Reweigh
            self.update_weights(self.calc_new_weights(self.gamma_cur, self.gammas[-2]))
            self.update_ess()

            # Resample
            if self.ess_cur <= 0.5 * self.num_particles:
                _logger.info("Resampling...")
                (
                    particles_resampled,
                    weights_resampled,
                    log_like_resampled,
                    log_prior_resampled,
                ) = self.resample()

                # update algorithm parameters
                self.particles = particles_resampled
                self.log_likelihood = log_like_resampled
                self.log_prior = log_prior_resampled
                self.log_posterior = self.log_likelihood + self.log_prior
                self.update_weights(weights_resampled)

                self.update_ess(resampled=True)

            _logger.info("step %s gamma: %.5f ESS: %.5f", step, self.gamma_cur, self.ess_cur)

            # estimate current covariance matrix
            cov_mat = np.atleast_2d(
                np.cov(self.particles, ddof=0, aweights=self.weights, rowvar=False)
            )

            # scale covariance based on average acceptance rate of last rejuvenation step
            scale_prop_cov = self.a + self.b * avg_accept
            cov_mat *= scale_prop_cov**2

            # Rejuvenate
            self.mcmc_kernel.pre_run(
                self.particles, self.log_likelihood, self.log_prior, self.gamma_cur, cov_mat
            )
            self.mcmc_kernel.core_run()
            (
                self.particles,
                self.log_likelihood,
                self.log_prior,
                self.log_posterior,
                avg_accept,
            ) = self.mcmc_kernel.post_run()

            # plot the trace every plot_trace_every-th iteration
            if self.plot_trace_every and not step % self.plot_trace_every:
                self.draw_trace(step)

    def post_run(self):
        """Analyze the resulting importance sample."""
        normalized_weights = self.weights / np.sum(self.weights)

        particles_resampled, _, _, _ = self.resample()
        if self.result_description:
            # TODO # pylint: disable=fixme
            # interpret the resampled particles as a single markov chain -> in accordance with the
            # Metropolis Hastings iterator add a dimension to the numpy array
            # this enables the calculation of the covariance matrix
            results = process_outputs(
                # pylint: disable=duplicate-code
                {
                    "result": particles_resampled[:, np.newaxis, :],
                    "particles": self.particles,
                    "weights": normalized_weights,
                    "log_likelihood": self.log_likelihood,
                    "log_prior": self.log_prior,
                    "log_posterior": self.log_posterior,
                },
                self.result_description,
            )
            if self.result_description["write_results"]:
                write_results(results, self.global_settings.result_file(".pickle"))

            if self.result_description["plot_results"]:
                self.draw_trace("final")

            mean = np.sum(np.multiply(np.squeeze(normalized_weights), self.particles.T).T, axis=0)
            var = np.sum(
                np.multiply(np.squeeze(normalized_weights), np.power(self.particles - mean, 2).T).T,
                axis=0,
            )
            std = np.sqrt(var)

            _logger.info("\tESS: %s", self.ess_cur)
            _logger.info("\tIS mean±std: %s±%s", mean, std)

            _logger.info(
                "\tmean±std: %s±%s",
                results.get("mean", np.nan),
                np.sqrt(results.get("var", np.nan)),
            )
            _logger.info("\tvar: %s", (results.get("var", np.nan)))
            _logger.info("\tcov: %s", results.get("cov", np.nan))

    def draw_trace(self, step):
        """Plot the trace of the current particle approximation.

        Args:
            step (int): Current step index
        """
        particles_resampled, _, _, _ = self.resample()
        data_dict = {
            variable_name: particles_resampled[:, i]
            for i, variable_name in enumerate(self.parameters.parameters_keys)
        }
        inference_data = az.convert_to_inference_data(data_dict)
        az.plot_trace(inference_data)
        plt.savefig(self.global_settings.result_file(suffix=f"_trace_{step}", extension=".png"))
        plt.close("all")
