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
"""HMC algorithm.

The Hamiltonian Monte Carlo sampler is a gradient based MCMC algortihm.
It is used to sample from arbitrary probability distributions.
"""

import logging

import pymc as pm

from queens.iterators._pymc import PyMC
from queens.utils.logger_settings import log_init_args

_logger = logging.getLogger(__name__)


class HamiltonianMonteCarlo(PyMC):
    """Iterator based on HMC algorithm.

    The HMC sampler is a state of the art MCMC sampler. It is based on the Hamiltonian mechanics.

    Attributes:
        max_steps (int): Maximum of leapfrog steps to take in one iteration
        target_accept (float): Target accpetance rate which should be conistent after burn-in
        path_length (float): Maximum length of particle trajectory
        step_size (float): Step size, scaled by 1/(parameter dimension^0.25)
        scaling (np.array): The inverse mass, or precision matrix
        is_cov (boolean): Setting if the scaling is a mass or covariance matrix
        init_strategy (str): Strategy to tune mass damping matrix
        advi_iterations (int): Number of iteration steps of ADVI based init strategies
    """

    @log_init_args
    def __init__(
        self,
        model,
        parameters,
        global_settings,
        num_samples,
        seed,
        num_burn_in=100,
        num_chains=1,
        discard_tuned_samples=True,
        result_description=None,
        summary=True,
        pymc_sampler_stats=False,
        as_inference_dict=False,
        use_queens_prior=False,
        progressbar=False,
        max_steps=100,
        target_accept=0.65,
        path_length=2.0,
        step_size=0.25,
        scaling=None,
        is_cov=False,
        init_strategy="auto",
        advi_iterations=50_000,
    ):
        """Initialize HMC iterator.

        Args:
            model (Model): Model to be evaluated by iterator
            parameters (Parameters): Parameters object
            global_settings (GlobalSettings): settings of the QUEENS experiment including its name
                                              and the output directory
            num_samples (int): Number of samples to generate per chain, excluding burn-in period
            seed (int): Seed for rng
            num_burn_in (int, opt): Number of burn-in steps
            num_chains (int, opt): Number of chains to sample
            discard_tuned_samples (boolean, opt): Setting to discard the samples of the burin-in
                                                  period
            result_description (dict, opt): Settings for storing and visualizing the results
            summary (bool, opt):  Print sampler summary
            pymc_sampler_stats (bool, opt): Compute additional sampler statistics
            as_inference_dict (bool, opt): Return inference_data object instead of trace object
            use_queens_prior (boolean, opt): Setting for using the PyMC priors or the QUEENS prior
                                             functions
            progressbar (boolean, opt): Setting for printing progress bar while sampling
            max_steps (int, opt): Maximum of leapfrog steps to take in one iteration
            target_accept (float, opt): Target accpetance rate which should be conistent after
                                        burn-in
            path_length (float, opt): Maximum length of particle trajectory
            step_size (float, opt): Step size, scaled by 1/(parameter dimension^0.25)
            scaling (np.array, opt): The inverse mass, or precision matrix
            is_cov (boolean, opt): Setting if the scaling is a mass or covariance matrix
            init_strategy (str, opt): Strategy to tune mass damping matrix
            advi_iterations (int, opt): Number of iteration steps of ADVI based init strategies
        """
        # pylint: disable=duplicate-code
        super().__init__(
            model=model,
            parameters=parameters,
            global_settings=global_settings,
            num_burn_in=num_burn_in,
            num_chains=num_chains,
            num_samples=num_samples,
            discard_tuned_samples=discard_tuned_samples,
            result_description=result_description,
            summary=summary,
            pymc_sampler_stats=pymc_sampler_stats,
            as_inference_dict=as_inference_dict,
            seed=seed,
            use_queens_prior=use_queens_prior,
            progressbar=progressbar,
        )
        self.max_steps = max_steps
        self.target_accept = target_accept
        self.path_length = path_length
        self.step_size = step_size
        self.scaling = scaling
        self.is_cov = is_cov
        self.init_strategy = init_strategy
        self.advi_iterations = advi_iterations

    def init_mcmc_method(self):
        """Init the PyMC MCMC Model.

        Returns:
            step (obj): The MCMC Method within the PyMC Model
        """
        # have only scaling or potential as mass matrix
        potential = None
        if self.scaling is None:
            # use NUTS init to get potential for the init
            _logger.info("Using NUTS initialization to init HMC, ignore next line.")
            self.initvals, step_helper = pm.init_nuts(
                init=self.init_strategy,
                chains=1,
                initvals=self.initvals,
                progressbar=self.progressbar,
                n_init=self.advi_iterations,
                random_seed=self.seed,
                model=self.pymc_model,
            )
            potential = step_helper.potential
        step = pm.HamiltonianMC(
            target_accept=self.target_accept,
            max_steps=self.max_steps,
            path_length=self.path_length,
            step_scale=self.step_size,
            scaling=self.scaling,
            is_cov=self.is_cov,
            potential=potential,
            model=self.pymc_model,
        )
        return step
