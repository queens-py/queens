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
"""Monte Carlo iterator."""

import logging

import matplotlib.pyplot as plt
import numpy as np

from queens.iterators.iterator import Iterator
from queens.utils.logger_settings import log_init_args
from queens.utils.process_outputs import process_outputs, write_results

_logger = logging.getLogger(__name__)


class MonteCarloIterator(Iterator):
    """Basic Monte Carlo Iterator to enable MC sampling.

    Attributes:
        seed  (int): Seed for random number generation.
        num_samples (int): Number of samples to compute.
        result_description (dict):  Description of desired results.
        samples (np.array):         Array with all samples.
        output (np.array):          Array with all model outputs.
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
    ):
        """Initialise Monte Carlo iterator.

        Args:
            model (Model): Model to be evaluated by iterator
            parameters (Parameters): Parameters object
            global_settings (GlobalSettings): settings of the QUEENS experiment including its name
                                              and the output directory
            seed  (int):                Seed for random number generation
            num_samples (int):          Number of samples to compute
            result_description (dict, opt):  Description of desired results
        """
        super().__init__(model, parameters, global_settings)
        self.seed = seed
        self.num_samples = num_samples
        self.result_description = result_description
        self.samples = None
        self.output = None

    def pre_run(self):
        """Generate samples for subsequent MC analysis and update model."""
        np.random.seed(self.seed)
        self.samples = self.parameters.draw_samples(self.num_samples)

    def core_run(self):
        """Run Monte Carlo Analysis on model."""
        self.output = self.model.evaluate(self.samples)

    def post_run(self):
        """Analyze the results."""
        if self.result_description is not None:
            results = process_outputs(self.output, self.result_description, self.samples)
            if self.result_description["write_results"]:
                write_results(results, self.global_settings.result_file(".pickle"))

                # ----------------------------- WIP PLOT OPTIONS ----------------------------
                if self.result_description["plot_results"]:
                    _, ax = plt.subplots(figsize=(6, 4))

                    # Check for dimensionality of the results
                    if results["raw_output_data"]["result"].shape[1] == 1:
                        data = results["raw_output_data"]["result"]
                        ax.hist(data, bins="auto")
                        ax.set_xlabel(r"Output")
                        ax.set_ylabel(r"Count [-]")
                        plt.tight_layout()
                        plt.savefig(self.global_settings.result_file(".png"))
                        plt.show()
                    else:
                        _logger.warning(
                            "Plotting is not implemented yet for a multi-dimensional model output"
                        )

        _logger.debug("Size of inputs %s", self.samples.shape)
        _logger.debug("Inputs %s", self.samples)
        _logger.debug("Size of outputs %s", self.output["result"].shape)
        _logger.debug("Outputs %s", self.output["result"])
