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
"""Sequential Monte Carlo implementation using *particles* package."""

import logging
import re
from collections import OrderedDict

import numpy as np
import particles
from particles import collectors as col
from particles import distributions as dists
from particles.smc_samplers import AdaptiveTempering

from queens.distributions.distributions import Distribution
from queens.iterators.iterator import Iterator
from queens.utils import smc_utils
from queens.utils.logger_settings import log_init_args
from queens.utils.print_utils import get_str_table
from queens.utils.process_outputs import process_outputs, write_results

_logger = logging.getLogger(__name__)


class SequentialMonteCarloChopinIterator(Iterator):
    """Sequential Monte Carlo algorithm from Chopin et al.

    Sequential Monte Carlo algorithm based on the book [1] (especially chapter 17)
    and the particles library (https://github.com/nchopin/particles/)


    References:
        [1]: Chopin N. and Papaspiliopoulos O. (2020), An Introduction to Sequential Monte Carlo,
             10.1007/978-3-030-47845-2 , Springer.

    Attributes:
        result_description (dict): Settings for storing and visualizing the results.
        seed (int): Seed for random number generator.
        num_particles (int): Number of particles.
        num_variables (int): Number of primary variables.
        n_sims (int): Number of model calls.
        max_feval (int): Maximum number of model calls.
        prior (object): Particles Prior object.
        smc_obj (object): Particles SMC object.
        resampling_threshold (float): Ratio of ESS to particle number at which to resample.
        resampling_method (str): Resampling method implemented in particles.
        feynman_kac_model (str): Feynman Kac model for the smc object.
        num_rejuvenation_steps (int): Number of rejuvenation steps (e.g. MCMC steps).
        waste_free (bool): If *True*, all intermediate Markov steps are kept.
    """

    @log_init_args
    def __init__(
        self,
        model,
        parameters,
        global_settings,
        result_description,
        num_particles,
        max_feval,
        seed,
        resampling_threshold,
        resampling_method,
        feynman_kac_model,
        num_rejuvenation_steps,
        waste_free,
    ):
        """Initialize the SMC iterator.

        Args:
            model (Model): Model to be evaluated by iterator
            parameters (Parameters): Parameters object
            global_settings (GlobalSettings): settings of the QUEENS experiment including its name
                                              and the output directory
            result_description (dict): Settings for storing and visualizing the results
            num_particles (int): Number of particles
            max_feval (int): Maximum number of model calls
            seed (int): Seed for random number generator
            resampling_threshold (float): Ratio of ESS to particle number at which to resample
            resampling_method (str): Resampling method implemented in particles
            feynman_kac_model (str): Feynman Kac model for the smc object
            num_rejuvenation_steps (int): Number of rejuvenation steps (e.g. MCMC steps)
            waste_free (bool): if True, all intermediate Markov steps are kept
        """
        super().__init__(model, parameters, global_settings)
        self.result_description = result_description
        self.seed = seed
        self.num_particles = num_particles
        self.num_variables = self.parameters.num_parameters
        self.n_sims = 0
        self.max_feval = max_feval
        self.prior = None
        self.smc_obj = None
        self.resampling_threshold = resampling_threshold
        self.resampling_method = resampling_method
        self.feynman_kac_model = feynman_kac_model
        self.num_rejuvenation_steps = num_rejuvenation_steps
        self.waste_free = waste_free
        self._initialize_prior_model()

    def eval_log_likelihood(self, samples):
        """Evaluate natural logarithm of likelihood at sample.

        Args:
            samples (np.array): Samples/particles of the SMC.

        Returns:
            log_likelihood (np.array): Value of log-likelihood for samples.
        """
        log_likelihood = self.model.evaluate(samples)["result"]
        return log_likelihood

    def _initialize_prior_model(self):
        """Initialize the prior model form the problem description."""
        if self.parameters.random_field_flag:
            raise NotImplementedError(
                "Particles SMC for random fields is not yet implemented! Abort..."
            )

        # Important that has to be a OrderedDict otherwise there is a mismatch between particles
        # and QUEENS parameters
        prior_dict = OrderedDict()

        # Generate prior using the particles library
        for key, parameter in self.parameters.dict.items():

            # native QUEENS distribution
            if isinstance(parameter, Distribution):
                prior_dict[key] = ParticlesChopinDistribution(parameter)

            # native particles distribution
            elif isinstance(parameter, dists.ProbDist):
                prior_dict[key] = parameter
            else:
                raise TypeError(
                    "The prior distribution has to be of type queens.Distribution or of type"
                    " particles distributions.ProbDist"
                )
        self.prior = dists.StructDist(prior_dict)

        # Sanity check
        if (qparams := list(prior_dict.keys())) != (pparams := list(self.prior.laws.keys())):
            raise ValueError(
                f"Order of parameters between QUEENS {qparams} and particles {pparams} is "
                "mismatching! "
            )

    def initialize_feynman_kac(self, static_model):
        """Initialize the Feynman Kac model for the SMC approach.

        Args:
            static_model (StaticModel): Static model from the particles library

        Returns:
            feynman_kac_model (FKSMCsampler): Model for the smc object
        """
        if self.feynman_kac_model == "adaptive_tempering":
            feynman_kac_model = AdaptiveTempering(
                static_model,
                ESSrmin=self.resampling_threshold,
                wastefree=self.waste_free,
                len_chain=self.num_rejuvenation_steps + 1,  # is handle this way in the library
            )
        else:
            raise NotImplementedError(
                "The allowed Feynman Kac models are: 'tempering' and 'adaptive_tempering'"
            )
        return feynman_kac_model

    def pre_run(self):
        """Draw initial sample."""
        _logger.info("Initialize run.")
        np.random.seed(self.seed)

        # Likelihood function for the static model based on the QUEENS function
        log_likelihood = self.eval_log_likelihood

        # Static model for the Feynman Kac model
        static_model = smc_utils.StaticStateSpaceModel(
            data=None,
            prior=self.prior,
            likelihood_model=log_likelihood,
        )

        # Feynman Kac model for the SMC algorithm
        feynman_kac_model = self.initialize_feynman_kac(static_model)

        # SMC object
        self.smc_obj = particles.SMC(
            fk=feynman_kac_model,
            N=self.num_particles,
            verbose=False,
            collect=[col.Moments()],
            resampling=self.resampling_method,
            qmc=False,  # QMC can not be used in this static setting in particles (currently)
        )

    def core_run(self):
        """Core run of Sequential Monte Carlo iterator.

        The particles library is generator based. Hence, one step of the
        SMC algorithm is done using *next(self.smc)*. As the *next()*
        function is called during the for loop, we only need to add some
        logging and check if the number of model runs is exceeded.
        """
        _logger.info("Welcome to SMC (particles) core run.")

        for _ in self.smc_obj:
            _logger.info(re.sub(r"t=.*?,", f"t={self.smc_obj.t -1},", str(self.smc_obj)))
            self.n_sims = self.smc_obj.fk.model.n_sims
            _logger.info("Total number of forward runs %s", self.n_sims)
            _logger.info("-" * 70)
            if self.n_sims >= self.max_feval:
                _logger.warning("Maximum number of model evaluations reached!")
                _logger.warning("Stopping SMC...")
                break

    def post_run(self):
        """Analyze the resulting importance sample."""
        # SMC data
        particles_smc = self.smc_obj.fk.model.particles_array_to_numpy(self.smc_obj.X.theta)
        weights = self.smc_obj.W.reshape(-1, 1)

        # First and second moment
        mean = self.smc_obj.fk.model.particles_array_to_numpy(
            self.smc_obj.summaries.moments[-1]["mean"]
        )[0]
        variance = self.smc_obj.fk.model.particles_array_to_numpy(
            self.smc_obj.summaries.moments[-1]["var"]
        )[0]
        if self.result_description:
            results = process_outputs(
                {
                    "particles": particles_smc,
                    "weights": weights,
                    "log_posterior": self.smc_obj.X.lpost,
                    "mean": mean,
                    "var": variance,
                    "n_sims": self.n_sims,
                },
                self.result_description,
            )
            if self.result_description["write_results"]:
                write_results(results, self.global_settings.result_file(".pickle"))
            _logger.info("Post run data exported!")


# Interface from QUEENS to Particles distributions
class ParticlesChopinDistribution(dists.ProbDist):
    """Distribution interfacing QUEENS  distributions to particles."""

    def __init__(self, queens_distribution):
        """Initialize distribution.

        Args:
            queens_distribution (queens.Distribution): QUEENS distribution
        """
        self.queens_distribution = queens_distribution

    @property
    def dim(self):
        """Dimension of the distribution.

        Returns:
            int: dimension of the RV
        """
        return self.queens_distribution.dimension

    def logpdf(self, x):
        """Logpdf of the distribution.

        Args:
            x (np.ndarray): Input locations

        Returns:
            np.ndarray: logpdf values
        """
        return self.queens_distribution.logpdf(x)

    def pdf(self, x):
        """Pdf of the distribution.

        Args:
            x (np.ndarray): Input locations

        Returns:
            np.ndarray: pdf values
        """
        return self.queens_distribution.pdf(x)

    def rvs(self, size=None):
        """Draw samples of the distribution.

        size basically is the number of samples.

        Args:
            size (np.ndarray, optional): Shape of the outputs. Defaults to None.

        Returns:
            np.ndarray: samples of the distribution
        """
        if size is None:
            size = self.dim  # This is a strange particles thing

        samples = self.queens_distribution.draw(num_draws=size)

        if self.dim == 1:
            samples = samples.flatten()
        return samples

    def ppf(self, u):
        """Ppf of the distribution.

        Args:
            u (np.ndarray): Input locations

        Returns:
            np.ndarray: ppf values
        """
        return self.queens_distribution.ppf(u)

    def __str__(self):
        """String method of the distribution.

        Returns:
            str: description of the distribution
        """
        distribution_type = (
            f"{type(self).__name__} wrapper for {type(self.queens_distribution).__name__}"
        )
        return get_str_table(distribution_type, self.queens_distribution.export_dict())
