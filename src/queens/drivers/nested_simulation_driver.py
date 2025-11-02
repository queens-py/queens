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
"""Driver to start adaptive simulations.

Adaptive simulations are inner experiment with (serial) restarts, an
inner iterator, inner serial scheduler and an inner simulation model.
"""

import logging

from queens.drivers.jobscript import JobOptions, Jobscript
from queens.global_settings import GlobalSettings
from queens.main import run_iterator
from queens.models import Simulation
from queens.schedulers import Serial
from queens.utils.metadata import SimulationMetadata

_logger = logging.getLogger(__name__)


class NestedSimulationDriver(Jobscript):
    """A driver for adaptive simulations.

    Runs nested, serial experiments using an inner driver and iterator.
    """

    def __init__(
        self,
        outer_parameters,
        input_templates,
        inner_driver,
        inner_iterator,
        outer_data_processor=None,
        outer_gradient_data_processor=None,
        files_to_copy=None,
        debug=False,
    ):
        """Initialize the adaptive driver.

        Args:
            outer_parameters (Parameter): Parameters of the main (outer) experiment
            input_templates (str, dict): input templates with placeholders for the outer parameters
            inner_driver (Jobscript): the driver for the inner experiment.
            inner_iterator (Iterator): the iterator for the inner experiment.
            outer_data_processor (DataProcessor): extracts the results of the inner
                                                  experiment, i.e. the adaptive simulation.
            outer_gradient_data_processor (Data Processor): extracts the results of the inner
                                                            experiment, i.e. the adaptive
                                                            simulation.
            files_to_copy (list, str): Files to copy. Defaults to None.
            debug (bool): Request debug output. Defaults to False.
        """
        self.inner_driver = inner_driver
        self.inner_iterator = inner_iterator
        self.debug = debug

        super().__init__(
            parameters=outer_parameters,
            input_templates=input_templates,
            jobscript_template="",  # not needed since we never create a jobscript
            executable="",  # not needed since we never create a jobscript
            files_to_copy=files_to_copy,
            data_processor=outer_data_processor,
            gradient_data_processor=outer_gradient_data_processor,
            extra_options=None,
        )

    def run(self, sample, job_id, num_procs, experiment_dir, experiment_name):
        """Override the run method and start the inner experiment instead.

        Args:
            sample (np.array): current (outer) sample
            job_id (_type_): outer job id
            num_procs (int): number of processors for a single job.
            experiment_dir (Path): outer experiment directory
            experiment_name (_type_): outer experiment name

        Returns:
            results (np.array): results for the outer experiment.
        """
        outer_job_dir, outer_output_dir, outer_output_file, outer_input_files, _ = (
            self._manage_paths(job_id, experiment_dir)
        )

        # Broadcast the files_to_copy down to the inner driver.
        # However, the input_templates of the outer driver are
        # no longer needed and should not be copied.
        # (the outer input_templates get added to self.files_to_copy
        # via the _driver constructor)
        inner_files_to_copy = [
            file for file in self.files_to_copy if file not in self.input_templates.values()
        ]

        # initialize the inner driver
        self.inner_driver.initialize(
            input_templates=outer_input_files, files_to_copy=inner_files_to_copy
        )

        outer_sample_dict = self.parameters.sample_as_dict(sample)

        metadata = SimulationMetadata(
            job_id=job_id, inputs=outer_sample_dict, job_dir=outer_job_dir
        )

        with metadata.time_code("prepare_input_files"):
            job_options = JobOptions(
                job_dir=outer_job_dir,
                output_dir=outer_output_dir,
                output_file=outer_output_file,
                job_id=job_id,
                num_procs=num_procs,
                experiment_dir=experiment_dir,
                experiment_name=experiment_name,
                input_files=outer_input_files,
            )

            # Create the input files
            self.prepare_input_files(
                job_options.add_data_and_to_dict(outer_sample_dict),
                experiment_dir,
                outer_input_files,
            )

        with metadata.time_code("run_time_adaptive_simulation"):
            self.run_inner_iterator(
                outer_job_dir=outer_job_dir,
                num_procs=num_procs,
            )

        with metadata.time_code("data_processing"):
            results = self._get_results(outer_output_dir)
            metadata.outputs = results

        return results

    def run_inner_iterator(self, outer_job_dir, num_procs):
        """Method to start the inner experiment.

        Args:
            outer_job_dir (Path): current outer job directory
            num_procs (int): numbers of processes
        """
        # as hardcoded by the jobscript driver.
        output_prefix = "output"

        inner_scheduler = Serial(
            experiment_name=".",  # this is to flatten the directory structure
            num_procs=num_procs,
            verbose=True,
            experiment_base_dir=outer_job_dir / output_prefix,  # the experiment_bas_dir of
            # the inner iterator is the current job_dir of the outer iterator
        )

        inner_model = Simulation(inner_scheduler, self.inner_driver)

        # inner_global_settings should not be a context, as client shutdown
        # is managed by the outer global settings
        inner_global_settings = GlobalSettings(
            experiment_name=self.inner_iterator.__class__.__name__,  # use inner_iterator's name
            output_dir=outer_job_dir,  # again, simply write into the current job_dir of the
            # outer driver directly to flatten the directory structure
            debug=self.debug,
        )

        # initialize the inner iterator
        self.inner_iterator.initialize(inner_model, inner_global_settings)
        #### Analysis ####
        run_iterator(self.inner_iterator, inner_global_settings)
