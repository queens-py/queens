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
"""Latin hypercube sampling iterator."""

import logging

import numpy as np
from pyDOE import lhs

from queens.iterators._iterator import Iterator
from queens.utils.logger_settings import log_init_args
from queens.utils.process_outputs import process_outputs, write_results

_logger = logging.getLogger(__name__)


class LatinHypercubeSampling(Iterator):
    """Basic LHS Iterator to enable Latin Hypercube sampling.

    Attributes:
        seed (int): Seed for numpy random number generator.
        num_samples (int):    Number of samples to compute.
        num_iterations (int): Number of optimization iterations of design.
        result_description (dict):  Description of desired results.
        criterion (str): Allowable values are:

            *   *center* or *c*
            *   *maximin* or *m*
            *   *centermaximin* or *cm*
            *   *correlation* or *corr*
        samples (np.array):   Array with all samples.
        output (np.array):   Array with all model outputs.
    """

    @log_init_args
    def __init__(
        self,
        model,
        parameters,
        global_settings,
        seed,
        num_samples,
        result_description=None,
        num_iterations=10,
        criterion="maximin",
    ):
        """Initialise LHSiterator.

        Args:
            model (Model): Model to be evaluated by iterator
            parameters (Parameters): Parameters object
            global_settings (GlobalSettings): settings of the QUEENS experiment including its name
                                              and the output directory
            seed (int): Seed for numpy random number generator
            num_samples (int):    Number of samples to compute
            result_description (dict, opt):  Description of desired results
            num_iterations (int): Number of optimization iterations of design
            criterion (str): Allowable values are "center" or "c", "maximin" or "m",
                             "centermaximin" or "cm", and "correlation" or "corr"
        """
        super().__init__(model, parameters, global_settings)
        self.seed = seed
        self.num_samples = num_samples
        self.num_iterations = num_iterations
        self.result_description = result_description
        self.criterion = criterion
        self.samples = None
        self.output = None

    def pre_run(self):
        """Generate samples for subsequent LHS analysis."""
        np.random.seed(self.seed)

        num_inputs = self.parameters.num_parameters

        _logger.info("Number of inputs: %s", num_inputs)
        _logger.info("Number of samples: %s", self.num_samples)
        _logger.info("Criterion: %s", self.criterion)
        _logger.info("Number of iterations: %s", self.num_iterations)

        # create latin hyper cube samples in unit hyper cube
        hypercube_samples = lhs(
            num_inputs, self.num_samples, criterion=self.criterion, iterations=self.num_iterations
        )
        # scale and transform samples according to the inverse cdf
        self.samples = self.parameters.inverse_cdf_transform(hypercube_samples)

    def core_run(self):
        """Run LHS Analysis on model."""
        self.output = self.model.evaluate(self.samples)

    def post_run(self):
        """Analyze the LHS results."""
        if self.result_description is not None:
            results = process_outputs(self.output, self.result_description, input_data=self.samples)
            if self.result_description["write_results"]:
                write_results(results, self.global_settings.result_file(".pickle"))

        _logger.info("Size of inputs %s", self.samples.shape)
        _logger.debug("Inputs %s", self.samples)
        _logger.info("Size of outputs %s", self.output["result"].shape)
        _logger.debug("Outputs %s", self.output["result"])
