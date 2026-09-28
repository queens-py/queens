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
"""Unit tests for the function driver."""

from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

from queens.drivers.function import Function


@pytest.mark.parametrize("function_type", ["explicit", "kwargs"])
def test_pass_run_arguments(function_type):
    """Test that the function receives the correct run arguments."""
    reference_parameter = 1.5
    reference_job_id = 2
    reference_num_procs = 4
    reference_experiment_dir = Path("/function_test")
    reference_experiment_name = "function_test"

    if function_type == "explicit":

        def function(parameter, job_id, num_procs, experiment_dir, experiment_name):
            assert parameter == reference_parameter
            assert job_id == reference_job_id
            assert num_procs == reference_num_procs
            assert experiment_dir == reference_experiment_dir
            assert experiment_name == reference_experiment_name
            return parameter

    elif function_type == "kwargs":

        def function(parameter, **kwargs):
            assert parameter == reference_parameter
            assert kwargs["job_id"] == reference_job_id
            assert kwargs["num_procs"] == reference_num_procs
            assert kwargs["experiment_dir"] == reference_experiment_dir
            assert kwargs["experiment_name"] == reference_experiment_name
            return parameter

    else:
        raise ValueError(f"Unknown function type: {function_type}")

    parameters = Mock()
    parameters.sample_as_dict.return_value = {"parameter": reference_parameter}
    driver = Function(parameters=parameters, function=function)

    result = driver.run(
        sample=np.array([reference_parameter]),
        job_id=reference_job_id,
        num_procs=reference_num_procs,
        experiment_dir=reference_experiment_dir,
        experiment_name=reference_experiment_name,
    )
    assert result["result"] == reference_parameter


def test_pass_run_arguments_overwrite_parameter():
    """Test error if a run argument overwrites a parameter."""
    reference_parameter = 1.5
    reference_job_id = 2
    reference_num_procs = 4
    reference_experiment_dir = Path("/function_test")
    reference_experiment_name = "function_test"

    def function(num_procs):
        return num_procs

    parameters = Mock()
    parameters.sample_as_dict.return_value = {"num_procs": reference_parameter}
    driver = Function(parameters=parameters, function=function)

    with pytest.raises(KeyError):
        driver.run(
            sample=np.array([reference_parameter]),
            job_id=reference_job_id,
            num_procs=reference_num_procs,
            experiment_dir=reference_experiment_dir,
            experiment_name=reference_experiment_name,
        )
