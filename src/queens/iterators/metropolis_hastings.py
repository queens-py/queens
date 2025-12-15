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
"""Metropolis-Hastings algorithm.

The Metropolis-Hastings algorithm is a Markov Chain Monte Carlo (MCMC)
method for obtaining a sequence of random samples from a probability
distribution from which direct sampling is difficult." [1]

References:
    [1]: https://en.wikipedia.org/wiki/Metropolis%E2%80%93Hastings_algorithm
"""

import logging

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from queens.distributions.normal import Normal
from queens.iterators._iterator import Iterator
from queens.utils import mcmc as mcmc_utils
from queens.utils import sequential_monte_carlo as smc_utils
from queens.utils.logger_settings import log_init_args
from queens.utils.process_outputs import process_outputs, write_results

_logger = logging.getLogger(__name__)


class MetropolisHastings(Iterator):
    """Iterator based on Metropolis-Hastings algorithm.

    The Metropolis-Hastings algorithm can be considered the benchmark
    Markov Chain Monte Carlo (MCMC) algorithm. It may be used to sample
    from complex, intractable probability distributions from which
    direct sampling is difficult or impossible.
    The implemented version is a random walk Metropolis-Hastings
    algorithm.

    Attributes:
        num_chains (int): Number of independent chains to run.
        num_samples (int): Number of samples to draw per chain.
        proposal_distribution (obj): Proposal distribution used for generating candidate samples.
        result_description (dict): Description of the desired results.
        as_smc_rejuvenation_step (bool): Indicates whether this iterator is used as a rejuvenation
                                         step for an SMC iterator or as the main iterator itself.
        tune (bool): Whether to tune the scale of the proposal distribution's covariance.
        scale_covariance (float or np.array): Scale of covariance matrix.
        num_burn_in (int): Number of initial samples to discard as burn-in.
        temper (function): Tempering function used in the acceptance probability calculation.
        gamma (float): Tempering parameter.
        tune_interval (int): Interval for tuning the scale of the covariance matrix.
        tot_num_samples (int): Total number of samples to be drawn per chain, including burn-in.
        chains (np.array): Array storing all the samples drawn across all chains.
        log_likelihood (np.array): Logarithms of the likelihood of the samples.
        log_prior (np.array): Logarithms of the prior probabilities of the samples.
        log_posterior (np.array): Logarithms of the posterior probabilities of the samples.
        seed (int): Seed for random number generation.
        accepted (np.array): Number of accepted proposals per chain.
        accepted_interval (np.array): Number of proposals per chain in current tuning interval.
    """

    @log_init_args
    def __init__(
        self,
        model,
        parameters,
        global_settings,
        result_description,
        proposal_distribution,
        num_samples,
        seed,
        tune=False,
        tune_interval=100,
        scale_covariance=1.0,
        num_burn_in=0,
        num_chains=1,
        as_smc_rejuvenation_step=False,
        temper_type="bayes",
    ):
        """Initialize Metropolis-Hastings iterator.

        Args:
            model (Model): Model to be evaluated by iterator
            parameters (Parameters): Parameters object
            global_settings (GlobalSettings): settings of the QUEENS experiment including its name
                                              and the output directory
            result_description (dict): Description of desired results.
            proposal_distribution (obj): Proposal distribution.
            num_samples (int): Number of samples per chain.
            seed (int): Seed for random number generator.
            tune (bool): Tune the scale of covariance.
            tune_interval (int): Tune the scale of the covariance every *tune_interval*-th step.
            scale_covariance: scale_covariance (float): Scale of covariance matrix of gaussian
                                                        proposal distribution.
            num_burn_in (int): Number of burn-in samples.
            num_chains (int): Number of independent chains.
            as_smc_rejuvenation_step (bool): Indicates whether this iterator is used as a
                                             rejuvenation step for an SMC iterator or as the main
                                             iterator itself.
            temper_type (str): Temper type ('bayes' or 'generic')
        """
        super().__init__(model, parameters, global_settings)

        self.num_chains = num_chains
        self.num_samples = num_samples

        self.proposal_distribution = proposal_distribution

        self.result_description = result_description
        self.as_smc_rejuvenation_step = as_smc_rejuvenation_step

        self.tune = tune
        # TODO change back to a scalar value. # pylint: disable=fixme
        #  Otherwise the diagnostics tools might not make sense
        self.scale_covariance = np.ones(self.num_chains) * scale_covariance
        self.num_burn_in = num_burn_in

        if not isinstance(self.proposal_distribution, Normal):
            raise RuntimeError("Currently only Normal proposals are supported as MCMC Kernel.")

        self.temper = smc_utils.temper_factory(temper_type)
        # fixed within MH, adapted by SMC if needed
        self.gamma = 1.0

        self.tune_interval = tune_interval

        # check agreement of dimensions of proposal distribution and
        # parameter space
        num_parameters = self.parameters.num_parameters
        if num_parameters != self.proposal_distribution.dimension:
            raise ValueError(
                "Dimensions of proposal distribution and parameter space do not agree."
            )

        self.tot_num_samples = self.num_samples + self.num_burn_in + 1
        self.chains = np.zeros((self.tot_num_samples, self.num_chains, num_parameters))
        self.log_likelihood = np.zeros((self.tot_num_samples, self.num_chains))
        self.log_prior = np.zeros((self.tot_num_samples, self.num_chains))
        self.log_posterior = np.zeros((self.tot_num_samples, self.num_chains))

        self.seed = seed

        self.accepted = np.zeros(self.num_chains)
        self.accepted_interval = np.zeros(self.num_chains)

    def eval_log_prior(self, samples):
        """Evaluate natural logarithm of prior at samples of chains.

        Args:
            samples (np.array): Samples for which to evaluate the prior.

        Returns:
            np.array: Logarithms of the prior probabilities for each sample.
        """
        return self.parameters.joint_logpdf(samples)

    def eval_log_likelihood(self, samples):
        """Evaluate natural logarithm of likelihood at samples of chains.

        Args:
            samples (np.array): Samples for which to evaluate the likelihood.

        Returns:
            np.array: Logarithms of the likelihood for each sample.
        """
        log_likelihood = self.model.evaluate(samples)["result"]
        return log_likelihood

    def do_mh_step(self, step_id):
        """Metropolis (Hastings) step.

        Args:
            step_id (int): Current step index for the MCMC run.
        """
        # tune covariance of proposal
        if not step_id % self.tune_interval and self.tune:
            accept_rate_interval = np.exp(
                np.log(self.accepted_interval) - np.log(self.tune_interval)
            )
            if not self.as_smc_rejuvenation_step:
                _logger.info("Current acceptance rate: %s.", accept_rate_interval)
            self.scale_covariance = mcmc_utils.tune_scale_covariance(
                self.scale_covariance, accept_rate_interval
            )
            self.accepted_interval = np.zeros(self.num_chains)

        cur_sample = self.chains[step_id - 1]
        # the scaling only holds for random walks
        delta_proposal = (
            self.proposal_distribution.draw(num_draws=self.num_chains)
            * self.scale_covariance[:, np.newaxis]
        )
        proposal = cur_sample + delta_proposal

        log_likelihood_prop = self.eval_log_likelihood(proposal)
        log_prior_prop = self.eval_log_prior(proposal)

        log_posterior_prop = self.temper(log_prior_prop, log_likelihood_prop, self.gamma)
        log_accept_prob = log_posterior_prop - self.log_posterior[step_id - 1]

        new_sample, accepted = mcmc_utils.mh_select(log_accept_prob, cur_sample, proposal)
        self.accepted += accepted
        self.accepted_interval += accepted

        self.chains[step_id] = new_sample

        self.log_likelihood[step_id] = np.where(
            accepted, log_likelihood_prop, self.log_likelihood[step_id - 1]
        )
        self.log_prior[step_id] = np.where(accepted, log_prior_prop, self.log_prior[step_id - 1])
        self.log_posterior[step_id] = np.where(
            accepted, log_posterior_prop, self.log_posterior[step_id - 1]
        )

    def pre_run(
        self,
        initial_samples=None,
        initial_log_like=None,
        initial_log_prior=None,
        gamma=1.0,
        cov_mat=None,
    ):
        """Draw initial sample.

        Args:
            initial_samples (np.array, optional): Initial samples for the chains.
            initial_log_like (np.array, optional): Initial log-likelihood values.
            initial_log_prior (np.array, optional): Initial log-prior values.
            gamma (float, optional): Tempering parameter for the posterior calculation.
            cov_mat (np.array, optional): Covariance matrix for the proposal distribution.
        """
        if not self.as_smc_rejuvenation_step:
            _logger.info("Initialize Metropolis-Hastings run.")

            np.random.seed(self.seed)

            # draw initial sample from prior distribution
            initial_samples = self.parameters.draw_samples(self.num_chains)
            initial_log_like = self.eval_log_likelihood(initial_samples)
            initial_log_prior = self.eval_log_prior(initial_samples)
        else:
            # create random walk normal proposal
            mean = np.zeros(cov_mat.shape[0])
            self.proposal_distribution = Normal(mean=mean, covariance=cov_mat)

        self.gamma = gamma

        self.chains[0] = initial_samples
        self.log_likelihood[0] = initial_log_like
        self.log_prior[0] = initial_log_prior

        self.log_posterior[0] = self.temper(self.log_prior[0], self.log_likelihood[0], self.gamma)

    def core_run(self):
        """Core run of Metropolis-Hastings iterator.

        1. Burn-in phase
        2. Sampling phase
        """
        if not self.as_smc_rejuvenation_step:
            _logger.info("Metropolis-Hastings core run.")

        # Burn-in phase
        for i in tqdm(range(1, self.num_burn_in + 1)):
            self.do_mh_step(i)

        if self.num_burn_in:
            burn_in_accept_rate = np.exp(np.log(self.accepted) - np.log(self.num_burn_in))
            _logger.info("Acceptance rate during burn in: %s", burn_in_accept_rate)
        # reset number of accepted samples
        self.accepted = np.zeros(self.num_chains)
        self.accepted_interval = 0

        # Sampling phase
        for i in tqdm(range(self.num_burn_in + 1, self.num_burn_in + self.num_samples + 1)):
            self.do_mh_step(i)

    def post_run(self):
        """Analyze the resulting chain."""
        avg_accept_rate = np.exp(
            np.log(np.sum(self.accepted)) - np.log((self.num_samples * self.num_chains))
        )
        if self.as_smc_rejuvenation_step:
            # the iterator is used as MCMC kernel for the Sequential Monte Carlo iterator
            return [
                self.chains[-1],
                self.log_likelihood[-1],
                self.log_prior[-1],
                self.log_posterior[-1],
                avg_accept_rate,
            ]
        if self.result_description:
            initial_samples = self.chains[0]
            chain_burn_in = self.chains[1 : self.num_burn_in + 1]
            chain_core = self.chains[self.num_burn_in + 1 : self.num_samples + self.num_burn_in + 1]

            accept_rate = np.exp(np.log(self.accepted) - np.log(self.num_samples))

            # process output takes a dict as input with key 'result'
            results = process_outputs(
                # pylint: disable=duplicate-code
                {
                    "result": chain_core,
                    "accept_rate": accept_rate,
                    "chain_burn_in": chain_burn_in,
                    "initial_sample": initial_samples,
                    "log_likelihood": self.log_likelihood,
                    "log_prior": self.log_prior,
                    "log_posterior": self.log_posterior,
                },
                self.result_description,
            )
            if self.result_description["write_results"]:
                write_results(results, self.global_settings.result_file(".pickle"))

            _logger.info("Size of outputs %s", chain_core.shape)
            for i in range(self.num_chains):
                _logger.info("#############################################")
                _logger.info("Chain %d", i + 1)
                _logger.info("\tAcceptance rate: %s", accept_rate[i])
                _logger.info(
                    "\tCovariance of proposal : %s",
                    (self.scale_covariance[i] * self.proposal_distribution.covariance).tolist(),
                )
                _logger.info(
                    "\tmean±std: %s±%s",
                    results.get("result", np.array([np.nan] * self.num_chains))[i],
                    np.sqrt(results.get("var", np.array([np.nan] * self.num_chains))[i]),
                )
                _logger.info(
                    "\tvar: %s", results.get("var", np.array([np.nan] * self.num_chains))[i]
                )
                _logger.info(
                    "\tcov: %s",
                    results.get("cov", np.array([np.nan] * self.num_chains))[i].tolist(),
                )
            _logger.info("#############################################")

            data_dict = {
                variable_name: np.swapaxes(chain_core[:, :, i], 1, 0)
                for i, variable_name in enumerate(self.parameters.parameters_keys)
            }
            inference_data = az.convert_to_inference_data(data_dict)

            rhat = az.rhat(inference_data)
            _logger.info(rhat)
            ess = az.ess(inference_data, relative=True)
            _logger.info(ess)
            az.plot_trace(inference_data)

            plt.savefig(self.global_settings.result_file(suffix="_trace", extension=".png"))

            az.plot_autocorr(inference_data)
            plt.savefig(self.global_settings.result_file(suffix="_autocorr", extension=".png"))
            plt.close("all")

        return None
