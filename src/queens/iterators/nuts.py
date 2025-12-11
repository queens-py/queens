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
"""No-U-Turn algorithm.

The No-U-Turn sampler is a gradient based MCMC algortihm. It builds on
the Hamiltonian Monte Carlo sampler to sample from (high dimensional)
arbitrary probability distributions.
"""

import logging

import pymc as pm

from queens.iterators._pymc import PyMC
from queens.utils.logger_settings import log_init_args

_logger = logging.getLogger(__name__)


class NUTS(PyMC):
    """Iterator based on HMC algorithm.

    References:
        [1]: Hoffman et al. The No-U-Turn Sampler: Adaptively Setting Path Lengths in Hamiltonian
        Monte Carlo. 2011.

    The No-U-Turn sampler is a state of the art MCMC sampler. It is based on the Hamiltonian Monte
    Carlo sampler but eliminates the need for an specificed number of integration step by checking
    if the trajectory turns around. The algorithm is based on a building up a tree and selecting a
    random note as proposal.

    Attributes:
        max_treedepth (int): Maximum depth for the tree-search
        early_max_treedepth (int): Max tree depth of first 200 tuning samples
        step_size (float): Step size, scaled by 1/(parameter dimension ^ 0.25)
        target_accept (float): Target accpetance rate which should be conistent after burn-in
        scaling (np.array): The inverse mass, or precision matrix
        is_cov (boolean): Setting if the scaling is a mass or covariance matrix
        init_strategy (str): Strategy to tune mass damping matrix
        advi_iterations (int): Number of iteration steps of ADVI based init strategies

    Returns:
        nuts_iterator (obj): Instance of NUTS Iterator
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
        max_treedepth=10,
        early_max_treedepth=8,
        step_size=0.25,
        target_accept=0.8,
        scaling=None,
        is_cov=False,
        init_strategy="auto",
        advi_iterations=500_000,
    ):
        """Initialize NUTS iterator.

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
            max_treedepth (int): Maximum depth for the tree-search
            early_max_treedepth (int): Max tree depth of first 200 tuning samples
            step_size (float): Step size, scaled by 1/(parameter dimension ^ 0.25)
            target_accept (float): Target accpetance rate which should be conistent after burn-in
            scaling (np.array): The inverse mass, or precision matrix
            is_cov (boolean): Setting if the scaling is a mass or covariance matrix
            init_strategy (str): Strategy to tune mass damping matrix
            advi_iterations (int): Number of iteration steps of ADVI based init strategies
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
        self.max_treedepth = max_treedepth
        self.early_max_treedepth = early_max_treedepth
        self.step_size = step_size
        self.target_accept = target_accept
        self.scaling = scaling
        self.is_cov = is_cov
        self.init_strategy = init_strategy
        self.advi_iterations = advi_iterations

    def init_mcmc_method(self):
        """Init the PyMC MCMC Model.

        Returns:
            step (obj): The MCMC Method within the PyMC Model
        """
        # can only specify scaling or potential
        # init strategies are handled by potentials
        if self.scaling is None:
            self.initvals, step = pm.init_nuts(
                init=self.init_strategy,
                chains=1,
                random_seed=self.seed,
                progressbar=self.progressbar,
                initvals=self.initvals,
                target_accept=self.target_accept,
                max_treedepth=self.max_treedepth,
                early_max_treedepth=self.early_max_treedepth,
                step_scale=self.step_size,
                n_init=self.advi_iterations,
                model=self.pymc_model,
            )
        else:
            step = pm.NUTS(
                target_accept=self.target_accept,
                max_treedepth=self.max_treedepth,
                early_max_treedepth=self.early_max_treedepth,
                step_scale=self.step_size,
                scaling=self.scaling,
                is_cov=self.is_cov,
                model=self.pymc_model,
            )
        return step
