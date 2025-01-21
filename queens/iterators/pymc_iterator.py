#
# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (c) 2025, QUEENS contributors.
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
"""PyMC Iterators base class."""

import abc
import logging

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm
import pytensor.tensor as pt

from queens.iterators.iterator import Iterator
from queens.utils.process_outputs import process_outputs, write_results
from queens.utils.pymc import PymcDistributionWrapper, from_config_create_pymc_distribution_dict

_logger = logging.getLogger(__name__)


class PyMCIterator(Iterator):
    """Iterator based on PyMC.

    References:
        [1]: Salvatier et al. "Probabilistic programming in Python using PyMC3". PeerJ Computer
        Science. 2016.

    Attributes:
        result_description (dict): Settings for storing and visualizing the results
        discard_tuned_samples (boolean): Setting to discard the samples of the burin-in period
        num_chains (int): Number of chains to sample
        num_burn_in (int):  Number of burn-in steps
        num_samples (int): Number of samples to generate per chain, excluding burn-in period
        chains (np.array): Array with all samples
        seed (int): Seed for the random number generators
        pymc_model (obj): PyMC Model as inference environment
        step (obj): PyMC MCMC method to be used for sampling
        use_queens_prior (boolean): Setting for using the PyMC priors or the QUEENS prior functions
        progressbar (boolean): Setting for printing progress bar while sampling
        log_prior (fun): Function to evaluate the QUEENS joint log-prior
        log_like (fun): Function to evaluate QUEENS log-likelihood
        results (obj): PyMC inference object with sampling results
        results_dict (dict): PyMC inference results as dict
        summary (bool): Print sampler summary
        pymc_sampler_stats (bool): Compute additional sampler statistics
        as_inference_dict (bool): Return inference_data object instead of trace object
        initvals (dict): Dict with distribution names and starting point of chains
        model_fwd_evals (int): Number of model forward calls
        model_grad_evals (int): Number of model gradient calls
        buffered_samples (np.array): Most recent evalutated samples by the likelihood function
        buffered_gradients (np.array): Gradient of the most recent evaluated samples
        buffered_likelihoods (np.array): Most recent evalutated likelihoods
    """

    def __init__(
        self,
        model,
        parameters,
        global_settings,
        num_burn_in,
        num_chains,
        num_samples,
        discard_tuned_samples,
        result_description,
        summary,
        pymc_sampler_stats,
        as_inference_dict,
        seed,
        use_queens_prior,
        progressbar,
    ):
        """Initialize PyMC iterator.

        Args:
            model (Model): Model to be evaluated by iterator
            parameters (Parameters): Parameters object
            global_settings (GlobalSettings): settings of the QUEENS experiment including its name
                                              and the output directory
            num_burn_in (int): Number of burn-in steps
            num_chains (int): Number of chains to sample
            num_samples (int): Number of samples to generate per chain, excluding burn-in period
            discard_tuned_samples (boolean): If true, samples of the burin-in period are discarded
            result_description (dict): Settings for storing and visualizing the results
            summary (bool):  Print sampler summary
            pymc_sampler_stats (bool): Compute additional sampler statistics
            as_inference_dict (bool): Return inference_data object instead of trace object
            seed (int): Seed for rng
            use_queens_prior (boolean): Setting for using the PyMC priors or the QUEENS prior
                                        functions
            progressbar (boolean): Setting for printing progress bar while sampling
        """
        super().__init__(model, parameters, global_settings)
        self.result_description = result_description
        self.summary = summary
        self.pymc_sampler_stats = pymc_sampler_stats
        self.as_inference_dict = as_inference_dict
        if self.as_inference_dict or self.pymc_sampler_stats:
            _logger.warning(
                "Returning data as inference dict or getting additional sampler stats can lead to "
                "numpy memory errors for large chains."
            )

        self.discard_tuned_samples = discard_tuned_samples

        if num_chains > 1:
            raise ValueError(
                "Parallel sampling is currently not supported. Multiple chains are not"
                "indpendent in PyMC."
            )

        self.num_chains = num_chains
        self.num_burn_in = num_burn_in

        self.num_samples = num_samples

        num_parameters = self.parameters.num_parameters

        self.seed = seed
        np.random.seed(seed)

        self.pymc_model = pm.Model()
        self.step = None
        self.use_queens_prior = use_queens_prior

        draws = self.num_samples
        if not discard_tuned_samples:
            draws += num_burn_in
        self.chains = np.zeros((draws, self.num_chains, num_parameters))

        self.progressbar = progressbar
        if self.use_queens_prior:
            self.log_prior = None
        self.log_like = None
        self.results = None
        self.results_dict = None
        self.initvals = None

        self.model_fwd_evals = 0
        self.model_grad_evals = 0

        self.buffered_samples = None
        self.buffered_gradients = None
        self.buffered_likelihoods = None

    def eval_log_prior(self, samples):
        """Evaluate natural logarithm of prior at samples of chains.

        Args:
            samples (np.array): Samples to evaluate the prior at

        Returns:
            log_prior (np.array): Prior log-pdf
        """
        log_prior = self.parameters.joint_logpdf(samples).reshape(-1)
        return log_prior

    def eval_log_prior_grad(self, samples):
        """Evaluate the gradient of the log-prior.

        Args:
            samples (np.array): Samples to evaluate the gradient at

        Returns:
            log_prior_grad (np.array): Gradients of the log prior
        """
        log_prior_grad = self.parameters.grad_joint_logpdf(samples)
        return log_prior_grad

    def eval_log_likelihood(self, samples):
        """Evaluate the log-likelihood.

        Args:
             samples (np.array): Samples to evaluate the likelihood at

        Returns:
            log_likelihood (np.array): log-likelihoods
        """
        if np.array_equal(self.buffered_samples, samples):
            log_likelihood = self.buffered_likelihoods
        else:
            self.model_fwd_evals += self.num_chains
            self.model_grad_evals += self.num_chains
            self.buffered_samples = samples.copy()
            log_likelihood, gradient = self.model.evaluate_and_gradient(samples)
            self.buffered_likelihoods = log_likelihood.copy()
            self.buffered_gradients = gradient.copy()
        return log_likelihood

    def eval_log_likelihood_grad(self, samples):
        """Evaluate the gradient of the log-likelihood.

        Args:
            samples (np.array): Samples to evaluate the gradient at

        Returns:
            gradient (np.array): Gradients of the log likelihood
        """
        # pylint: disable-next=fixme
        # TODO: find better way to do this evaluation
        if not np.array_equal(self.buffered_samples, samples):
            self.eval_log_likelihood(samples)

        gradient = self.parameters.latent_grad(self.buffered_gradients)
        return gradient.reshape(-1, self.parameters.num_parameters)

    @abc.abstractmethod
    def init_mcmc_method(self):
        """Init the PyMC MCMC Model."""

    def init_distribution_wrapper(self):
        """Init the PyMC wrapper for the QUEENS distributions."""
        self.log_like = PymcDistributionWrapper(
            self.eval_log_likelihood, self.eval_log_likelihood_grad
        )
        if self.use_queens_prior:
            self.log_prior = PymcDistributionWrapper(self.eval_log_prior, self.eval_log_prior_grad)

    def pre_run(self):
        """Prepare MCMC run."""
        self.init_distribution_wrapper()
        # pylint: disable-next=unnecessary-dunder-call
        self.pymc_model.__enter__()
        if self.use_queens_prior:
            _logger.info("Use QUEENS Priors")

            def random(*_, **kwargs):
                if kwargs["size"] == (self.num_chains, self.parameters.num_parameters):
                    return self.parameters.draw_samples(self.num_chains)
                raise ValueError("Wrong shape of rng values")

            name = "parameters"
            prior = pm.DensityDist(
                name,
                logp=self.log_prior,
                random=random,
                shape=(self.num_chains, self.parameters.num_parameters),
            )
            initvals_value = self.parameters.draw_samples(self.num_chains)
            self.initvals = {name: initvals_value}
        else:
            _logger.info("Use PyMC Priors")
            prior_list = from_config_create_pymc_distribution_dict(self.parameters, self.num_chains)
            prior = pm.math.concatenate(prior_list, axis=1)

        prior_tensor = pt.as_tensor_variable(prior)
        # pylint: disable-next=not-callable
        pm.Potential("likelihood", self.log_like(prior_tensor))

        # pylint: disable-next=unnecessary-dunder-call
        self.pymc_model.__exit__(None, None, None)
        self.step = self.init_mcmc_method()

    def core_run(self):
        """Core run of PyMC iterator."""
        self.results = pm.sample(
            draws=self.num_samples,
            step=self.step,
            cores=1,
            chains=1,
            initvals=self.initvals,
            tune=self.num_burn_in,
            random_seed=self.seed,
            discard_tuned_samples=self.discard_tuned_samples,
            progressbar=self.progressbar,
            compute_convergence_checks=self.pymc_sampler_stats,
            return_inferencedata=self.as_inference_dict,
            model=self.pymc_model,
        )

    def post_run(self):
        """Post-Processing of Results."""
        _logger.info("MCMC by PyMC results:")

        # get the chain as numpy array and dict
        inference_data_dict = {}
        is_inference_data = isinstance(self.results, az.InferenceData)
        parameter_names = list(self.pymc_model.named_vars.keys())
        # remove 'likelihood' from variable names
        parameter_names.pop(-1)
        current_index = 0

        for name in parameter_names:
            if is_inference_data:
                # get inference data
                values = np.squeeze(
                    self.results.posterior.get(name).to_numpy(),
                    axis=0,
                )
            else:
                # get data from trace
                values = self.results.get_values(name)

            self.chains[:, :, current_index : current_index + values.shape[2]] = values
            inference_data_dict[name] = np.swapaxes(values, 0, 1)

            current_index += values.shape[2]

        # sample_stats
        if is_inference_data:
            sample_stats = self.results
        else:
            sample_stats = {}
            for sampler_stats in self.results.stat_names:
                sample_stats[sampler_stats] = self.results.get_sampler_stats(sampler_stats)
            sample_stats["sampling_time"] = self.results.report.t_sampling
            sample_stats["chain_ok"] = self.results.report.ok
            # pylint: disable-next=protected-access
            sample_stats["chain_warnings"] = self.results.report._chain_warnings
            sample_stats["number_of_draws"] = self.results.report.n_draws
            sample_stats["number_of_tuning_steps"] = self.results.report.n_tune
            # pylint: disable-next=protected-access
            sample_stats["global_warnings"] = self.results.report._global_warnings
            sample_stats["model_forward_evals"] = self.model_fwd_evals
            sample_stats["model_gradient_evals"] = self.model_grad_evals

        # process output takes a dict as input with key 'mean'
        results_dict = az.convert_to_inference_data(inference_data_dict)
        results = process_outputs(
            {"sample_stats": sample_stats, "result": self.chains, "inference_data": results_dict},
            self.result_description,
        )
        if self.result_description["write_results"]:
            write_results(results, self.global_settings.result_file(".pickle"))

        self.results_dict = results_dict
        if self.summary:
            _logger.info("Inference summary:")
            _logger.info(az.summary(self.results_dict))
            _logger.info("Model forward evaluations: %i", self.model_fwd_evals)
            _logger.info("Model gradient evaluations: %i", self.model_grad_evals)

        if self.result_description["plot_results"]:
            _logger.info("Generate convergence plots, ignoring divergences for trace plotting.")

            _axes = az.plot_trace(self.results_dict, divergences=None)
            plt.savefig(self.global_settings.result_file(suffix="_trace", extension=".png"))

            _axes = az.plot_autocorr(self.results_dict)
            plt.savefig(self.global_settings.result_file(suffix="_autocorr", extension=".png"))

            _axes = az.plot_forest(
                self.results_dict,
                combined=True,
                hdi_prob=0.95,
                r_hat=True,
                ess=True,
                kind="ridgeplot",
                ridgeplot_overlap=4,
                ridgeplot_alpha=0.5,
                ridgeplot_truncate=False,
            )
            plt.savefig(self.global_settings.result_file(suffix="_forest", extension=".png"))

            if self.parameters.num_parameters < 17:
                az.plot_density(self.results_dict, hdi_prob=0.99)
                plt.savefig(self.global_settings.result_file(suffix="_marginals", extension=".png"))

            plt.close("all")

        _logger.info("MCMC by PyMC results finished")
