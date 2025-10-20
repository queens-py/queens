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
"""Sobol sequence iterator."""

import logging

from queens.iterators._iterator import Iterator
from queens.utils.logger_settings import log_init_args
from queens.utils.process_outputs import process_outputs, write_results
from queens.utils.sobol_sequence import sample_sobol_sequence

_logger = logging.getLogger(__name__)


class SobolSequence(Iterator):
    """Sobol sequence in multiple dimensions.

    Attributes:
        seed  (int): This is the seed for the scrambling. The seed of the random number generator is
                     set to this, if specified. Otherwise, it uses a random seed.
        number_of_samples (int): Number of samples to compute.
        randomize (bool): Setting this to *True* will produce scrambled Sobol sequences. Scrambling
                          is capable of producing better Sobol sequences.
        result_description (dict):  Description of desired results.
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
        number_of_samples,
        result_description,
        randomize=False,
    ):
        """Initialize Sobol sequence iterator.

        Args:
            model (Model): Model to be evaluated by iterator
            parameters (Parameters): Parameters object
            global_settings (GlobalSettings): settings of the QUEENS experiment including its name
                                              and the output directory
            seed  (int): This is the seed for the scrambling. The seed of the random number
                         generator is set to this, if specified. Otherwise, it uses a random seed.
            number_of_samples (int): Number of samples to compute
            result_description (dict):  Description of desired results
            randomize (bool): Setting this to True will produce scrambled Sobol sequences.
                              Scrambling is capable of producing better Sobol sequences.
        """
        super().__init__(model, parameters, global_settings)

        self.seed = seed
        self.number_of_samples = number_of_samples
        self.randomize = randomize
        self.result_description = result_description
        self.samples = None
        self.output = None

    def pre_run(self):
        """Generate samples for subsequent Sobol sequence analysis."""
        _logger.info("Number of inputs: %s", self.parameters.num_parameters)
        _logger.info("Number of samples: %s", self.number_of_samples)
        _logger.info("Randomize: %s", self.randomize)

        self.samples = sample_sobol_sequence(
            dimension=self.parameters.num_parameters,
            number_of_samples=self.number_of_samples,
            parameters=self.parameters,
            randomize=self.randomize,
            seed=self.seed,
        )

    def core_run(self):
        """Run Sobol sequence analysis on model."""
        self.output = self.model.evaluate(self.samples)

    def post_run(self):
        """Analyze the Sobol sequence results."""
        if self.result_description is not None:  # pylint: disable=duplicate-code
            results = process_outputs(self.output, self.result_description, input_data=self.samples)
            if self.result_description["write_results"]:
                write_results(results, self.global_settings.result_file(".pickle"))
