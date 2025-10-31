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
"""Driver to run a jobscript."""


import logging
from dataclasses import dataclass
from pathlib import Path

from queens.drivers._driver import Driver
from queens.utils.exceptions import SubprocessError
from queens.utils.injector import inject, inject_in_template
from queens.utils.io import read_file
from queens.utils.logger_settings import log_init_args
from queens.utils.metadata import SimulationMetadata
from queens.utils.path import create_folder_if_not_existent
from queens.utils.run_subprocess import run_subprocess

_logger = logging.getLogger(__name__)


@dataclass
class JobOptions:
    """Dataclass for job options.

    All the attributes of this dataclass can be injected into input and
    jobscript files. The input files dictionary will be flattened, such
    that the input files are injected by their key.
    """

    job_dir: Path
    output_dir: Path
    output_file: Path
    job_id: int
    num_procs: int
    experiment_dir: Path
    experiment_name: str
    input_files: dict

    def to_dict(self):
        """Create a job options dict.

        Returns:
            dict: Dict containing all the data.
        """
        dictionary = self.__dict__.copy()
        dictionary.update(dictionary.pop("input_files"))
        return dictionary

    def add_data_and_to_dict(self, additional_data):
        """Add additional options to the job options dict.

        Args:
            additional_data (dict): Additional data to combine with the job options.

        Returns:
            dict: Dict combining the job options and the additional data.
        """
        return self.to_dict() | additional_data


class Jobscript(Driver):
    """Driver to run an executable with a jobscript.

    Attributes:
        input_templates (Path): Read in simulation input template as string.
        data_processor (obj): Instance of data processor class.
        gradient_data_processor (obj): Instance of data processor class for gradient data.
        jobscript_template (str): Read-in jobscript template.
        jobscript_options (dict): Dictionary containing jobscript options.
        jobscript_file_name (str): Jobscript file name (default: 'jobscript.sh').
        raise_error_on_jobscript_failure (bool): Whether to raise an error for a non-zero jobscript
                                                 exit code.
    """

    @log_init_args
    def __init__(
        self,
        parameters,
        input_templates,
        jobscript_template,
        executable,
        files_to_copy=None,
        data_processor=None,
        gradient_data_processor=None,
        jobscript_file_name="jobscript.sh",
        extra_options=None,
        raise_error_on_jobscript_failure=True,
    ):
        """Initialize Jobscript object.

        Args:
            parameters (Parameters): Parameters object.
            input_templates (str, Path, dict): Path(s) to simulation input template.
            jobscript_template (str, Path): Path to jobscript template or read-in jobscript
                                            template.
            executable (str, Path): Path to main executable of respective software.
            files_to_copy (list, opt): Files or directories to copy to experiment_dir.
            data_processor (obj, opt): Instance of data processor class.
            gradient_data_processor (obj, opt): Instance of data processor class for gradient data.
            jobscript_file_name (str, opt): Jobscript file name (default: 'jobscript.sh').
            extra_options (dict, opt): Extra options to inject into jobscript template.
            raise_error_on_jobscript_failure (bool, opt): Whether to raise an error for a non-zero
                                                          jobscript exit code.
        """
        super().__init__(parameters=parameters, files_to_copy=files_to_copy)
        self.input_templates = self.create_input_templates_dict(input_templates)
        self.jobscript_template = self.get_read_in_jobscript_template(jobscript_template)
        self.files_to_copy.extend(self.input_templates.values())
        self.data_processor = data_processor
        self.gradient_data_processor = gradient_data_processor

        if extra_options is None:
            extra_options = {}

        self.jobscript_options = extra_options
        self.jobscript_options["executable"] = executable
        self.jobscript_file_name = jobscript_file_name
        self.raise_error_on_jobscript_failure = raise_error_on_jobscript_failure

    @staticmethod
    def create_input_templates_dict(input_templates):
        """Cast input templates into a dict.

        Args:
            input_templates (str, Path, dict): Input template(s).

        Returns:
            dict: Dict containing input file names and template paths.
        """
        if not isinstance(input_templates, dict):
            input_templates = {"input_file": input_templates}

        input_templates_dict = {
            input_template_key: Path(input_template_path)
            for input_template_key, input_template_path in input_templates.items()
        }
        return input_templates_dict

    @staticmethod
    def get_read_in_jobscript_template(jobscript_template):
        """Get the jobscript template contents.

        If the provided jobscript template is a Path or a string of a
        path and a valid file, the corresponding file is read.

        Args:
            jobscript_template (str, Path): Path to jobscript template or read-in jobscript
                                            template.

        Returns:
            str: Read-in jobscript template
        """
        if isinstance(jobscript_template, str):
            # Catch an exception due to a long string
            try:
                if Path(jobscript_template).is_file():
                    jobscript_template = read_file(jobscript_template)
            except OSError:
                _logger.debug(
                    "The provided jobscript template string is not a regular file so we assume "
                    "that it holds the read-in jobscript template. The jobscript template reads:\n"
                    "%s",
                    {jobscript_template},
                )

        elif isinstance(jobscript_template, Path):
            if jobscript_template.is_file():
                jobscript_template = read_file(jobscript_template)
            else:
                raise FileNotFoundError(
                    f"The provided jobscript template path {jobscript_template} is not a file."
                )
        else:
            raise TypeError("The jobscript template needs to be a string or a Path.")

        return jobscript_template

    def run(self, sample, job_id, num_procs, experiment_dir, experiment_name):
        """Run the driver.

        Args:
            sample (dict): Dict containing sample.
            job_id (int): Job ID.
            num_procs (int): Number of processors.
            experiment_dir (Path): Path to QUEENS experiment directory.
            experiment_name (str): Name of QUEENS experiment.

        Returns:
            Result and potentially the gradient.
        """
        job_dir, output_dir, output_file, input_files, log_file = self._manage_paths(
            job_id, experiment_dir
        )

        sample_dict = self.parameters.sample_as_dict(sample)

        metadata = SimulationMetadata(job_id=job_id, inputs=sample_dict, job_dir=job_dir)

        with metadata.time_code("prepare_input_files"):
            job_options = JobOptions(
                job_dir=job_dir,
                output_dir=output_dir,
                output_file=output_file,
                job_id=job_id,
                num_procs=num_procs,
                experiment_dir=experiment_dir,
                experiment_name=experiment_name,
                input_files=input_files,
            )

            # Create the input files
            self.prepare_input_files(
                job_options.add_data_and_to_dict(sample_dict), experiment_dir, input_files
            )

            jobscript_file = job_dir / self.jobscript_file_name

            # Create jobscript
            inject_in_template(
                job_options.add_data_and_to_dict(self.jobscript_options),
                self.jobscript_template,
                str(jobscript_file),
            )

        with metadata.time_code("run_jobscript"):
            execute_cmd = f"bash {jobscript_file} >{log_file} 2>&1"
            self._run_executable(job_id, execute_cmd)

        with metadata.time_code("data_processing"):
            results = self._get_results(output_dir)
            metadata.outputs = results

        return results

    def _manage_paths(self, job_id, experiment_dir):
        """Manage paths for driver run.

        Args:
            job_id (int): Job ID.
            experiment_dir (Path): Path to QUEENS experiment directory.

        Returns:
            job_dir (Path): Path to job directory.
            output_dir (Path): Path to output directory.
            output_file (Path): Path to output file(s).
            input_files (dict): Dict with name and path of the input file(s).
            log_file (Path): Path to log file.
        """
        job_dir = experiment_dir / str(job_id)
        output_dir = job_dir / "output"
        output_dir = create_folder_if_not_existent(output_dir)

        output_prefix = "output"
        output_file = output_dir / output_prefix
        log_file = output_dir / (output_prefix + ".log")

        input_files = {}
        for input_template_name, input_template_path in self.input_templates.items():
            input_file_str = input_template_name + "".join(input_template_path.suffixes)
            input_files[input_template_name] = job_dir / input_file_str

        return job_dir, output_dir, output_file, input_files, log_file

    def _run_executable(self, job_id, execute_cmd):
        """Run executable.

        Args:
            job_id (int): Job ID.
            execute_cmd (str): Executed command.
        """
        process_returncode, _, stdout, stderr = run_subprocess(
            execute_cmd,
            raise_error_on_subprocess_failure=False,
        )
        if self.raise_error_on_jobscript_failure and process_returncode:
            raise SubprocessError.construct_error_from_command(
                command=execute_cmd,
                command_output=stdout,
                error_message=stderr,
                additional_message=f"The jobscript with job ID {job_id} has failed with exit code "
                f"{process_returncode}.",
            )

    def _get_results(self, output_dir):
        """Get results from driver run.

        Args:
            output_dir (Path): Path to output directory.

        Returns:
            result (np.array): Result from the driver run.
            gradient (np.array, None): Gradient from the driver run (potentially None).
        """
        result = None
        if self.data_processor:
            result = self.data_processor.get_data_from_file(output_dir)
            _logger.debug("Got result: %s", result)

        gradient = None
        if self.gradient_data_processor:
            gradient = self.gradient_data_processor.get_data_from_file(output_dir)
            _logger.debug("Got gradient: %s", gradient)
        return result, gradient

    def prepare_input_files(self, sample_dict, experiment_dir, input_files):
        """Prepare and parse data to input files.

        Args:
            sample_dict (dict): Dict containing sample.
            experiment_dir (Path): Path to QUEENS experiment directory.
            input_files (dict): Dict with name and path of the input file(s).
        """
        for input_template_name, input_template_path in self.input_templates.items():
            inject(
                sample_dict,
                experiment_dir / input_template_path.name,
                input_files[input_template_name],
            )
