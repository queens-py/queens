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
"""Polynomial chaos iterator.

Is a wrapper on the *chaospy* library.
"""

import logging

import chaospy as cp
import numpy as np
from chaospy.distributions.sampler.generator import (
    SAMPLER_NAMES as collocation_valid_sampling_rules,
)
from chaospy.quadrature.frontend import SHORT_NAME_TABLE as projection_node_location_rules

from queens.distributions import beta, lognormal, normal, uniform
from queens.iterators.iterator import Iterator
from queens.utils.logger_settings import log_init_args
from queens.utils.process_outputs import write_results
from queens.utils.valid_options_utils import get_option

_logger = logging.getLogger(__name__)


class PolynomialChaosIterator(Iterator):
    """Collocation-based polynomial chaos iterator.

    Attributes:
        seed (int): Seed for random number generation.
        num_collocation_points (int): Number of samples to compute.
        sampling_rule (dict): Rule according to which samples are drawn.
        polynomial_order (int): Order of polynomial expansion.
        result_description (dict): Description of desired results.
        sparse (bool): For pseudo-spectral, if *True* uses sparse collocation points.
        polynomial_chaos_approach (str): Approach for the polynomial chaos approach.
        distribution (cp.distribution): Joint input distribution.
        samples: Sample points used in the computation.
        result_dict: Dictionary storing results including expansion, mean, and covariance.
    """

    @log_init_args
    def __init__(
        self,
        model,
        parameters,
        global_settings,
        num_collocation_points,
        polynomial_order,
        approach,
        result_description,
        sparse=None,
        sampling_rule=None,
        seed=42,
    ):
        """Initialise polynomial chaos iterator.

        Args:
            model (Model): Model to be evaluated by iterator
            parameters (Parameters): Parameters object
            global_settings (GlobalSettings): settings of the QUEENS experiment including its name
                                              and the output directory
            num_collocation_points (int): Number of samples to compute
            polynomial_order (int): Order of polynomial expansion
            approach (str): Approach for the polynomial chaos approach
            result_description (dict): Description of desired results
            sparse (bool, opt): For pseudo project, if *True* uses sparse collocation points
            sampling_rule (dict, opt): Rule according to which samples are drawn
            seed (int, opt): Seed for random number generation
        """
        super().__init__(model, parameters, global_settings)

        if not isinstance(num_collocation_points, int) or num_collocation_points < 1:
            raise ValueError("Number of samples for the polynomial must be a positive integer!")

        if approach == "collocation":
            valid_sampling_rules = collocation_valid_sampling_rules
        elif approach == "pseudo_spectral":
            _logger.info(
                "Maximum number of collocation points was set to %s.", num_collocation_points
            )
            valid_sampling_rules = projection_node_location_rules
            if sampling_rule is None:
                # Chaospy default
                sampling_rule = "clenshaw_curtis"

            if not isinstance(sparse, bool):
                raise ValueError(
                    f"Sparse input attribute needs to be set to true or false, not to {sparse}"
                )
        else:
            raise ValueError(
                f"Approach '{approach}' unknown. "
                f"Valid options are 'pseudo_spectral' and 'collocation'."
            )

        self.sampling_rule = get_option(
            valid_sampling_rules,
            sampling_rule,
            "Chaospy sampling rule unknown",
        )
        if not isinstance(polynomial_order, int) or polynomial_order < 0:
            raise ValueError(
                f"Polynomial expansion order has to be a non-negative integer. You provided "
                f"{polynomial_order}"
            )

        self.seed = seed
        self.num_collocation_points = num_collocation_points
        self.sampling_rule = sampling_rule
        self.polynomial_order = polynomial_order
        self.result_description = result_description
        self.sparse = sparse
        self.polynomial_chaos_approach = approach
        self.distribution = create_chaospy_joint_distribution(self.parameters)
        self.samples = None
        self.result_dict = None

    def pre_run(self):
        """Initialize run."""
        np.random.seed(self.seed)

    def core_run(self):
        """Core run for the polynomial chaos iterator."""
        _logger.info("Polynomial chaos using a %s approach", self.polynomial_chaos_approach)
        if self.polynomial_chaos_approach == "collocation":
            polynomial_expansion, collocation_points = self._regression_based_pc()
        else:
            polynomial_expansion, collocation_points = self._projection_based_pc()

        mean = cp.E(polynomial_expansion, self.distribution)
        covariance = cp.Cov(polynomial_expansion, self.distribution)
        self.result_dict = {}
        self.result_dict["polynomial_expansion"] = polynomial_expansion
        self.result_dict["distribution"] = self.distribution
        self.result_dict["mean"] = mean
        self.result_dict["covariance"] = covariance
        self.result_dict["collocation_points"] = collocation_points
        self.result_dict["polynomial_chaos_approach"] = self.polynomial_chaos_approach

    def _projection_based_pc(self):
        """Projection based polynomial chaos.

        Computes the coefficients of the polynomial expansion using a projection approach.

        Returns:
            polynomial_expansion (cp.polynomial): Projected expansion
            nodes (np.array): Column-wise locations to evaluate the model
        """
        # Generate nodes and weights for the polynomial chaos method
        nodes, weights = cp.generate_quadrature(
            self.polynomial_order, self.distribution, sparse=self.sparse, rule=self.sampling_rule
        )
        num_collocation_points = len(nodes)
        if num_collocation_points > self.num_collocation_points:
            raise ValueError(
                f"Maximum number of collocation points were set to {self.num_collocation_points}."
                f"However the current setup with polynomial degree {self.polynomial_order} would"
                f" lead to {num_collocation_points} collocation points"
            )
        _logger.info("Number of collocation points: %s", num_collocation_points)
        evaluations = self.model.evaluate(nodes.T)["result"]

        # Generate the polynomial chaos expansion based on the distribution
        expansion = cp.generate_expansion(self.polynomial_order, self.distribution)

        polynomial_expansion = cp.fit_quadrature(expansion, nodes, weights, evaluations)

        return polynomial_expansion, nodes

    def _regression_based_pc(self):
        """Regression based polynomial chaos.

        Computes the coefficients of the polynomial expansion using regression.

        Returns:
            polynomial_expansion (cp.polynomial): Fitted polynomial expansion
            nodes (np.array): Column-wise locations to evaluate the model
        """
        # Generate samples for the polynomial regression
        collocation_points = self.distribution.sample(
            self.num_collocation_points, rule=self.sampling_rule
        )

        evaluations = self.model.evaluate(collocation_points.T)["result"]

        # Generate the polynomial chaos expansion based on the distribution
        expansion = cp.generate_expansion(self.polynomial_order, self.distribution)

        polynomial_expansion = cp.fit_regression(expansion, collocation_points, evaluations)

        return polynomial_expansion, collocation_points

    def post_run(self):
        """Analyze the results."""
        if self.result_description is not None:
            if self.result_description["write_results"]:
                write_results(self.result_dict, self.global_settings.result_file(".pickle"))


def create_chaospy_joint_distribution(parameters):
    """Get random variables in chaospy distribution format.

    Args:
        parameters (obj): Parameters object

    Returns:
        chaospy distribution
    """
    cp_distribution_list = []

    # loop over random_variables and create joint
    for parameter in parameters.to_distribution_list():
        if parameter.dimension != 1:
            raise ValueError("Multidimensional random variables are not supported yet.")

        cp_distribution_list.append(create_chaospy_distribution(parameter))

    # Pass the distribution list as arguments
    return cp.J(*cp_distribution_list)


def create_chaospy_distribution(distribution):
    """Create chaospy distribution object from queens distribution.

    Args:
        distribution (obj): Queens distribution object

    Returns:
        distribution: Distribution object in chaospy format
    """
    if isinstance(distribution, normal.Normal):
        distribution = cp.Normal(mu=distribution.mean, sigma=distribution.covariance ** (1 / 2))
    elif isinstance(distribution, uniform.Uniform):
        distribution = cp.Uniform(
            lower=distribution.lower_bound,
            upper=distribution.upper_bound,
        )
    elif isinstance(distribution, lognormal.LogNormal):
        distribution = cp.LogNormal(
            mu=distribution.normal_mean,
            sigma=distribution.normal_covariance ** (1 / 2),
        )
    elif isinstance(distribution, beta.Beta):
        distribution = cp.Beta(
            alpha=distribution.a,
            beta=distribution.b,
            lower=distribution.lower_bound,
            upper=distribution.upper_bound,
        )
    return distribution
