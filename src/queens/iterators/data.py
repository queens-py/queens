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
"""Data iterator."""

import logging
import pickle

from queens.iterators._iterator import Iterator
from queens.utils.logger_settings import log_init_args
from queens.utils.process_outputs import process_outputs, write_results

_logger = logging.getLogger(__name__)


class Data(Iterator):
    """Basic Data Iterator to enable restarts from data.

    Attributes:
        samples (np.array): Array with all samples.
        output (np.array): Array with all model outputs.
        eigenfunc (obj): Function for computing eigenfunctions or transformations
                         applied to the data. This attribute is a placeholder and
                         may be updated in future versions (refer to Issue #45).
        path_to_data (string): Path to pickle file containing data.
        result_description (dict): Description of desired results.
    """

    @log_init_args
    def __init__(self, path_to_data, result_description, global_settings, parameters=None):
        """Initialize the data iterator.

        Args:
            path_to_data (string): Path to pickle file containing data.
            result_description (dict): Description of desired results.
            global_settings (GlobalSettings): Settings of the QUEENS experiment including its name
                                              and the output directory.
            parameters (Parameters, optional): Parameters object.
        """
        super().__init__(None, parameters, global_settings)
        self.samples = None
        self.output = None
        self.eigenfunc = (
            None  # TODO this is an intermediate solution--> see Issue #45 # pylint: disable=fixme
        )
        self.path_to_data = path_to_data
        self.result_description = result_description

    def core_run(self):
        """Read data from file."""
        # TODO: We should return a more general data structure in the future # pylint: disable=fixme
        # TODO: including I/O and meta data; for now catch it with a try statement # pylint: disable=fixme
        # TODO: see Issue #45; # pylint: disable=fixme
        try:
            self.samples, self.output, self.eigenfunc = self.read_pickle_file()
        except ValueError:
            self.samples, self.output = self.read_pickle_file()

    def post_run(self):
        """Analyze the results."""
        if self.result_description is not None:
            results = process_outputs(self.output, self.result_description)
            if self.result_description["write_results"]:
                write_results(results, self.global_settings.result_file(".pickle"))
        # else:
        _logger.info("Size of inputs %s", self.samples.shape)
        _logger.info("Inputs %s", self.samples)
        _logger.info("Size of outputs %s", self.output["result"].shape)
        _logger.info("Outputs %s", self.output["result"])

    def read_pickle_file(self):
        """Read in data from a pickle file.

        Main reason for putting this functionality in a method is to make
        mocking reading input easy for testing.

        Returns:
            np.array, np.array: Two arrays, the first contains input samples,
            the second the corresponding output samples
        """
        with open(self.path_to_data, "rb") as file:
            data = pickle.load(file)

        return data
