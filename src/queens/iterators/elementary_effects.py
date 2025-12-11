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
"""Elementary Effects iterator module.

Elementary Effects (also called Morris method) is a global sensitivity
analysis method, which can be used for parameter fixing (ranking).
"""

import logging

import numpy as np
from SALib.analyze import morris as morris_analyzer
from SALib.sample import morris

from queens.distributions.uniform import Uniform
from queens.iterators._iterator import Iterator
from queens.utils.logger_settings import log_init_args
from queens.utils.process_outputs import write_results
from queens.visualization.sa_visualization import SAVisualization

_logger = logging.getLogger(__name__)


class ElementaryEffects(Iterator):
    """Iterator to compute Elementary Effects (Morris method).

    Attributes:
        num_trajectories (int): Number of trajectories to generate.
        local_optimization (bool):  Flag whether to use local optimization according to Ruano et
                                    al. (2012). Speeds up the process tremendously for larger number
                                    of trajectories and *num_levels*. If set to *False*, brute force
                                    method is used.
        num_optimal_trajectories (int): Number of optimal trajectories to sample (between 2 and N).
        num_levels (int): Number of grid levels.
        seed (int): Seed for random number generation.
        confidence_level (float): Size of confidence interval.
        num_bootstrap_samples (int): Number of bootstrap samples used to compute confidence
                                     intervals for sensitivity measures.
        result_description (dict): Dictionary with desired result description.
        samples (np.array): Samples at which the model is evaluated.
        output (np.array): Results at samples.
        salib_problem (dict): Dictionary with SALib problem description.
        si (dict): Dictionary with all sensitivity indices.
        visualization (SAVisualization): Visualization object for SA.
    """

    @log_init_args
    def __init__(
        self,
        model,
        parameters,
        global_settings,
        num_trajectories,
        local_optimization,
        num_optimal_trajectories,
        number_of_levels,
        seed,
        confidence_level,
        num_bootstrap_samples,
        result_description,
    ):
        """Initialize ElementaryEffects.

        Args:
            model (Model): Model to be evaluated by iterator
            parameters (Parameters): Parameters object
            global_settings (GlobalSettings): settings of the QUEENS experiment including its name
                                              and the output directory
            num_trajectories (int): number of trajectories to generate
            local_optimization (bool): flag whether to use local optimization according to Ruano
                                       et al. (2012). Speeds up the process tremendously for
                                       larger number of trajectories and num_levels. If set to
                                       ``False`` brute force method is used.
            num_optimal_trajectories (int): number of optimal trajectories to sample (between 2
                                            and N)
            number_of_levels (int): number of grid levels
            seed (int): seed for random number generation
            confidence_level (float): size of confidence interval
            num_bootstrap_samples (int): number of bootstrap samples used to compute confidence
                                         intervals for sensitivity measures
            result_description (dict): dictionary with desired result description
        """
        super().__init__(model, parameters, global_settings)
        self.num_trajectories = num_trajectories
        self.local_optimization = local_optimization
        self.num_optimal_trajectories = num_optimal_trajectories
        self.num_levels = number_of_levels
        self.seed = seed
        self.confidence_level = confidence_level
        self.num_bootstrap_samples = num_bootstrap_samples
        self.result_description = result_description

        self.samples = None
        self.output = None
        self.salib_problem = {}
        self.si = {}

        self.visualization = None
        if result_description.get("plotting_options"):
            self.visualization = SAVisualization.from_config_create(
                result_description.get("plotting_options")
            )

    def pre_run(self):
        """Generate samples for subsequent analysis and update model."""
        bounds = []
        for parameter in self.parameters.dict.values():
            if not isinstance(parameter, Uniform) or parameter.dimension != 1:
                raise ValueError("Parameters must be 1D uniformly distributed.")
            bounds.append([parameter.lower_bound.squeeze(), parameter.upper_bound.squeeze()])

        self.salib_problem = {
            "num_vars": self.parameters.num_parameters,
            "names": self.parameters.names,
            "bounds": bounds,
            "groups": None,
        }

        self.samples = morris.sample(
            self.salib_problem,
            self.num_trajectories,
            num_levels=self.num_levels,
            optimal_trajectories=self.num_optimal_trajectories,
            local_optimization=self.local_optimization,
            seed=self.seed,
        )

    def core_run(self):
        """Run Analysis on model."""
        self.output = self.model.evaluate(self.samples)

        self.si = morris_analyzer.analyze(
            self.salib_problem,
            self.samples,
            np.reshape(self.output["result"], (-1)),
            num_resamples=self.num_bootstrap_samples,
            conf_level=self.confidence_level,
            print_to_console=False,
            num_levels=self.num_levels,
            seed=self.seed,
        )

    def post_run(self):
        """Analyze the results."""
        results = self.process_results()
        if self.result_description is not None:
            self.print_results(results)

            if self.result_description["write_results"]:
                write_results(results, self.global_settings.result_file(".pickle"))

            if self.visualization:
                self.visualization.plot(results)

    def process_results(self):
        """Write all results to self contained dictionary."""
        results = {"parameter_names": self.parameters.names, "sensitivity_indices": self.si}
        return results

    def print_results(self, results):
        """Print results to log.

        Args:
            results (dict): Dictionary with the results of the sensitivity analysis, including:

                - `parameter_names`: List of parameter names.
                - `sensitivity_indices`: Contains indices like:

                    - `names`: Parameter names.
                    - `mu_star`: Mean absolute effect.
                    - `mu`: Mean effect.
                    - `mu_star_conf`: Confidence interval for `mu_star`.
                    - `sigma`: Standard deviation of the effect.
        """
        _logger.info(
            "%-20s %10s %10s %15s %10s", "Parameter", "Mu_Star", "Mu", "Mu_Star_Conf", "Sigma"
        )

        for j in range(self.parameters.num_parameters):
            _logger.info(
                "%-20s %10.2e %10.2e %15.2e %10.2e",
                results["sensitivity_indices"]["names"][j],
                results["sensitivity_indices"]["mu_star"][j],
                results["sensitivity_indices"]["mu"][j],
                results["sensitivity_indices"]["mu_star_conf"][j],
                results["sensitivity_indices"]["sigma"][j],
            )
