#
# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (c) 2024, QUEENS contributors.
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
"""Driver to run 4C."""

from queens.drivers.jobscript_driver import JobscriptDriver
from queens.utils.logger_settings import log_init_args

_JOBSCRIPT_TEMPLATE = """
{{ mpi_cmd }} -np {{ num_procs }} {{ executable }} {{ input_file }} {{ output_file }}
if [ ! -z "{{ post_processor|default("") }}" ]
then
  {{ mpi_cmd }} -np {{ num_procs }} {{ post_processor }} --file={{ output_file }} {{ post_options }}
fi
"""


class FourcDriver(JobscriptDriver):
    """Driver to run 4C."""

    @log_init_args
    def __init__(
        self,
        parameters,
        input_templates,
        executable,
        files_to_copy=None,
        data_processor=None,
        gradient_data_processor=None,
        post_processor="",
        post_options="",
        mpi_cmd="/usr/bin/mpirun --bind-to none",
    ):
        """Initialize FourcDriver object.

        Args:
            parameters (Parameters): Parameters object
            input_templates (str, Path, dict): path to simulation input template
            executable (str, Path): path to main executable of respective software
            files_to_copy (list, opt): files or directories to copy to experiment_dir
            data_processor (obj, opt): instance of data processor class
            gradient_data_processor (obj, opt): instance of data processor class for gradient data
            post_processor (path, opt): path to post_processor
            post_options (str, opt): options for post-processing
            mpi_cmd (str, opt): mpi command
        """
        extra_options = {
            "post_processor": post_processor,
            "post_options": post_options,
            "mpi_cmd": mpi_cmd,
        }
        super().__init__(
            parameters=parameters,
            input_templates=input_templates,
            jobscript_template=_JOBSCRIPT_TEMPLATE,
            executable=executable,
            files_to_copy=files_to_copy,
            data_processor=data_processor,
            gradient_data_processor=gradient_data_processor,
            extra_options=extra_options,
        )
