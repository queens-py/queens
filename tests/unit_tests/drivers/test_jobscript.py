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
"""Unit tests for the jobscript driver."""

import os
from contextlib import nullcontext as does_not_raise

import numpy as np
import pytest
import yaml

from queens.data_processors import NumpyFile, TxtFile
from queens.distributions import FreeVariable
from queens.drivers.jobscript import JobOptions, Jobscript
from queens.parameters import Parameters
from queens.utils.exceptions import SubprocessError


@pytest.fixture(name="parameters")
def fixture_parameters():
    """Parameters for the jobscript driver test."""
    parameters = Parameters(parameter_1=FreeVariable(1), parameter_2=FreeVariable(1))
    return parameters


def create_template(list_of_keys, template_path):
    """Create dict from list of keys and write it to template path."""
    template_dict = {f: "{{ " + f + " }}" for f in list_of_keys}
    # yaml format is used to ease read in of input file
    template_path.write_text(yaml.safe_dump(template_dict))


@pytest.fixture(name="input_template")
def fixture_input_template(tmp_path, parameters):
    """Generate an input template."""
    input_template = tmp_path / "input_template.yaml"
    create_template(parameters.names, input_template)

    return input_template


@pytest.fixture(name="input_templates")
def fixture_input_templates(tmp_path, job_options, parameters):
    """Generate two input templates."""
    # only add the second parameter
    input_template_1 = tmp_path / "input_template_1.yaml"
    create_template(list(job_options.to_dict().keys()) + parameters.names[-1:], input_template_1)

    # add both parameters to the template
    input_template_2 = tmp_path / "input_template_2.yaml"
    create_template(parameters.names + ["input_1", "experiment_name"], input_template_2)

    return input_template_1, input_template_2


@pytest.fixture(name="jobscript_template")
def fixture_jobscript_template():
    """Dummy jobscript template."""
    return 'echo "This is a dummy jobscript"'


@pytest.fixture(name="jobscript_template_path")
def fixture_jobscript_template_path(tmp_path, jobscript_template):
    """Generate a dummy jobscript template."""
    jobscript_template_path = tmp_path / "dummy_jobscript_template.sh"
    jobscript_template_path.write_text(jobscript_template)

    return jobscript_template_path


@pytest.fixture(name="executable")
def fixture_executable(tmp_path):
    """Generate a dummy executable."""
    executable = tmp_path / "dummy_executable"
    executable.write_text("This is a dummy file.")

    # Make the dummy file executable
    os.chmod(executable, 0o755)

    return executable


@pytest.fixture(name="data_processor")
def fixture_data_processor():
    """Dummy data processor."""
    return NumpyFile(
        file_name_identifier="dummy.npy",
        file_options_dict={},
    )


@pytest.fixture(name="gradient_data_processor")
def fixture_gradient_data_processor():
    """Dummy gradient data processor."""
    return TxtFile(file_name_identifier="dummy.txt", file_options_dict={})


@pytest.fixture(name="jobscript_file_name")
def fixture_jobscript_file_name():
    """Jobscript file name."""
    return "dummy_jobscript.sh"


@pytest.fixture(name="extra_options")
def fixture_extra_options():
    """Extra options for JobOptions."""
    return {"option_1": 1, "option_2": "dummy"}


@pytest.fixture(name="job_id")
def fixture_job_id():
    """Fixture for the job id."""
    return 42


@pytest.fixture(name="injected_input_files")
def fixture_injected_input_files(tmp_path, job_id):
    """Fixture for the create input files."""
    input_file_1 = tmp_path / str(job_id) / "input_1.yaml"
    input_file_2 = tmp_path / str(job_id) / "input_2.yaml"
    injected_input_files = {"input_1": input_file_1, "input_2": input_file_2}

    return injected_input_files


@pytest.fixture(name="job_options")
def fixture_job_options(tmp_path, job_id, test_name, injected_input_files):
    """Job options to be injected."""
    num_procs = 4
    experiment_dir = tmp_path

    job_options = JobOptions(
        job_dir=tmp_path / str(job_id),
        output_dir=tmp_path / str(job_id) / "output",
        output_file=tmp_path / str(job_id) / "output/output",
        job_id=job_id,
        num_procs=num_procs,
        experiment_dir=experiment_dir,
        experiment_name=test_name,
        input_files=injected_input_files,
    )

    return job_options


@pytest.fixture(name="jobscript_driver")
def fixture_jobscript_driver(parameters, input_templates, executable):
    """Jobscript driver object."""
    input_template_1, input_template_2 = input_templates

    driver = Jobscript(
        parameters=parameters,
        jobscript_template="",
        executable=executable,
        input_templates={"input_1": input_template_1, "input_2": input_template_2},
    )

    return driver


@pytest.fixture(name="args_init")
def fixture_args_init(
    parameters,
    jobscript_template,
    executable,
    input_template,
    files_to_copy,
    data_processor,
    gradient_data_processor,
    jobscript_file_name,
    extra_options,
):
    """Arguments to initialize a Jobscript.

    These arguments are meant for initialization with the default
    constructor.
    """
    args_init = {
        "parameters": parameters,
        "jobscript_template": jobscript_template,
        "executable": executable,
        "input_templates": input_template,
        "files_to_copy": files_to_copy,
        "data_processor": data_processor,
        "gradient_data_processor": gradient_data_processor,
        "jobscript_file_name": jobscript_file_name,
        "extra_options": extra_options.copy(),
    }
    return args_init


def assert_jobscript_driver_attributes(jobscript_driver, args_init, extra_options):
    """Assert that the jobscript driver attributes are set correctly."""
    extra_options.update({"executable": args_init["executable"]})

    assert jobscript_driver.parameters == args_init["parameters"]
    assert jobscript_driver.input_templates == {"input_file": args_init["input_templates"]}
    assert jobscript_driver.jobscript_template == args_init["jobscript_template"]
    assert jobscript_driver.files_to_copy == args_init["files_to_copy"]
    assert jobscript_driver.data_processor == args_init["data_processor"]
    assert jobscript_driver.gradient_data_processor == args_init["gradient_data_processor"]
    assert jobscript_driver.jobscript_file_name == args_init["jobscript_file_name"]
    assert jobscript_driver.jobscript_options == extra_options


def test_init_from_jobscript_template_str(args_init, extra_options):
    """Test initialization of the Jobscript.

    For this initialization, the jobscript template is provided in the
    form of a string describing the jobscript template contents.
    """
    driver = Jobscript(**args_init)
    assert_jobscript_driver_attributes(driver, args_init, extra_options)


def test_init_from_jobscript_template_path(args_init, jobscript_template_path, extra_options):
    """Test initialization of the Jobscript.

    For this initialization, the jobscript template is provided in the
    form of a string describing the path to a file.
    """
    args_init_from_jobscript_template_path = args_init.copy()
    args_init_from_jobscript_template_path["jobscript_template"] = jobscript_template_path
    driver = Jobscript(**args_init_from_jobscript_template_path)
    assert_jobscript_driver_attributes(driver, args_init, extra_options)


def test_multiple_input_files(jobscript_driver, job_options, injected_input_files, parameters):
    """Test if multiple input files are correctly generated."""
    # Samples to be injected
    sample_dict = parameters.sample_as_dict(np.array([1, 2]))
    sample = np.array(list(sample_dict.values()))

    # Run the driver
    jobscript_driver.run_from_parameters(
        sample=sample,
        job_id=job_options.job_id,
        num_procs=job_options.num_procs,
        experiment_dir=job_options.experiment_dir,
        experiment_name=job_options.experiment_name,
    )

    # Join all options
    injectable_options = job_options.add_data_and_to_dict(sample_dict)

    # Check if all the data was injected correctly in all input files
    for input_file in injected_input_files.values():
        for key, value in yaml.safe_load(input_file.read_text()).items():
            assert value == str(injectable_options[key])


@pytest.mark.parametrize(
    "raise_error_on_jobscript_failure, expectation",
    [
        (False, does_not_raise()),
        (True, pytest.raises(SubprocessError)),
    ],
)
def test_error_in_jobscript_template(
    parameters, input_template, job_options, raise_error_on_jobscript_failure, expectation
):
    """Test for an error when the jobscript template has an error."""
    jobscript_driver = Jobscript(
        parameters=parameters,
        input_templates=input_template,
        jobscript_template="This jobscript should fail.",
        executable="",
        raise_error_on_jobscript_failure=raise_error_on_jobscript_failure,
    )
    sample_dict = parameters.sample_as_dict(np.array([1, 2]))
    sample = np.array(list(sample_dict.values()))

    with expectation:
        jobscript_driver.run_from_parameters(
            sample=sample,
            job_id=job_options.job_id,
            num_procs=job_options.num_procs,
            experiment_dir=job_options.experiment_dir,
            experiment_name=job_options.experiment_name,
        )


@pytest.mark.parametrize(
    "raise_error_on_jobscript_failure, expectation",
    [
        (False, does_not_raise()),
        (True, pytest.raises(SubprocessError)),
    ],
)
def test_nonzero_exit_code(
    parameters, input_template, job_options, raise_error_on_jobscript_failure, expectation
):
    """Test for an error when the jobscript exits with a code other than 0."""
    jobscript_driver = Jobscript(
        parameters=parameters,
        input_templates=input_template,
        jobscript_template="exit 1",
        executable="",
        raise_error_on_jobscript_failure=raise_error_on_jobscript_failure,
    )
    sample_dict = parameters.sample_as_dict(np.array([1, 2]))
    sample = np.array(list(sample_dict.values()))

    with expectation:
        jobscript_driver.run_from_parameters(
            sample=sample,
            job_id=job_options.job_id,
            num_procs=job_options.num_procs,
            experiment_dir=job_options.experiment_dir,
            experiment_name=job_options.experiment_name,
        )


def test_long_jobscript_template_str(parameters, input_template):
    """Test that a long jobscript template string does not raise an error."""
    long_str = "dummy" * 100
    jobscript_driver = Jobscript(
        parameters=parameters,
        input_templates=input_template,
        jobscript_template=long_str,
        executable="",
    )
    assert jobscript_driver.jobscript_template == long_str
