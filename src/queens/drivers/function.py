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
"""Function Driver."""

import inspect
from pathlib import Path
from typing import Any

import numpy as np

from example_simulator_functions import example_simulator_function_by_name
from queens.drivers._driver import Driver
from queens.utils.imports import get_module_attribute
from queens.utils.logger_settings import log_init_args


class Function(Driver):
    """Driver to run an python function.

    Attributes:
        function (function): Function to evaluate.
        function_requires_job_id (bool): True if function requires job_id
        function_requires_num_procs (bool): True if function requires num_procs
        function_requires_experiment_dir (bool): True if function requires experiment_dir
        function_requires_experiment_name (bool): True if function requires experiment_name
    """

    @log_init_args
    def __init__(
        self,
        parameters,
        function,
        external_python_module_function=None,
        worker_log_level=None,
    ):
        """Initialize Function object.

        Args:
            parameters (Parameters): Parameters object
            function (callable, str): Function or name of example function provided by QUEENS.
                If the function contains the arguments `job_id`, `num_procs`, `experiment_dir`,
                or `experiment_name`, they will be passed to the function when calling the `run`
                method. If the function contains a `**kwargs` parameter, all the arguments will
                be passed. Thus, these argument names should be avoided for non-related arguments.
            external_python_module_function (Path | str): Path to external module with function
            worker_log_level (int | str | None): Logging level of the job log files (default:
                                                 None, i.e., no log files are written)
        """
        super().__init__(
            parameters=parameters,
            worker_log_level=worker_log_level,
        )
        if external_python_module_function is None:
            if isinstance(function, str):
                # Try to load existing simulator functions
                my_function = example_simulator_function_by_name(function)
            else:
                my_function = function
        else:
            # Try to load external simulator functions
            my_function = get_module_attribute(external_python_module_function, function)

        def _function_signature_contains(argument: str) -> bool:
            """Check if the function signature contains a specific argument.

            Args:
                argument: Argument to check for in the function signature.

            Returns:
                True if the argument is in the function signature or if the function
                has a **kwargs parameter, False otherwise.
            """
            spec = inspect.getfullargspec(my_function)
            return bool(spec.varkw or argument in spec.args or argument in spec.kwonlyargs)

        # check which arguments we need to pass to the function
        self.function_requires_job_id = _function_signature_contains("job_id")
        self.function_requires_num_procs = _function_signature_contains("num_procs")
        self.function_requires_experiment_dir = _function_signature_contains("experiment_dir")
        self.function_requires_experiment_name = _function_signature_contains("experiment_name")

        # Wrap function to clean the output
        self.function = self.function_wrapper(my_function)

    @staticmethod
    def function_wrapper(function):
        """Wrap the function to be used.

        This wrapper calls the function by a kwargs dict only and reshapes the output as needed.
        This way if called in a pool, the reshaping is also done by the workers.

        Args:
            function (function): Function to be wrapped

        Returns:
            reshaped_output_function (function): Wrapped function
        """

        def reshaped_output_function(sample_dict):
            """Call function and reshape output.

            Args:
                sample_dict (dict): Dictionary containing parameters and `job_id`

            Returns:
                (np.ndarray): Result of the function call
            """
            result_array = function(**sample_dict)
            if isinstance(result_array, tuple):
                # here we expect a gradient return
                result = result_array[0]
                gradient = np.array(result_array[1])
                if not result.shape:
                    result = np.expand_dims(result, axis=0)
                    gradient = np.expand_dims(gradient, axis=0)
                return {"result": result, "gradient": gradient}

            # here no gradient return
            # take scalars and convert them to numpy floats
            if not isinstance(result_array, np.floating):
                result_array = np.float64(result_array)

            if not result_array.shape:
                result_array = np.expand_dims(result_array, axis=0)
            return {"result": result_array}

        return reshaped_output_function

    def run(
        self,
        sample: np.ndarray,
        job_id: int,
        num_procs: int,
        experiment_dir: Path,
        experiment_name: str,
    ) -> dict:
        """Run the driver.

        Args:
            sample (np.ndarray): Input sample
            job_id (int): Job ID
            num_procs (int): number of processors
            experiment_dir (Path): Path to QUEENS experiment directory.
            experiment_name (str): name of QUEENS experiment.

        Returns:
            Result and potentially the gradient
        """

        def _pass_run_arguments(sample_dict: dict, name: str, value: Any) -> None:
            """Pass arguments to the sample dictionary.

            Detect overwriting of keys and raise an error if found.
            """
            if name in sample_dict:
                raise KeyError(f"The key {name} is already in the sample dictionary.")
            sample_dict[name] = value

        sample_dict = self.parameters.sample_as_dict(sample)
        if self.function_requires_job_id:
            _pass_run_arguments(sample_dict, "job_id", job_id)
        if self.function_requires_num_procs:
            _pass_run_arguments(sample_dict, "num_procs", num_procs)
        if self.function_requires_experiment_dir:
            _pass_run_arguments(sample_dict, "experiment_dir", experiment_dir)
        if self.function_requires_experiment_name:
            _pass_run_arguments(sample_dict, "experiment_name", experiment_name)
        results = self.function(sample_dict)
        return results
