"""Function Driver."""

import inspect
import logging

import numpy as np

from queens.drivers.driver import Driver
from queens.example_simulator_functions import example_simulator_function_by_name
from queens.utils.import_utils import get_module_attribute
from queens.utils.logger_settings import log_init_args

_logger = logging.getLogger(__name__)


class FunctionDriver(Driver):
    """Driver to run an python function.

    Attributes:
        function (function): Function to evaluate.
        function_requires_job_id (bool): True if function requires job_id
    """

    @log_init_args
    def __init__(
        self,
        function,
        external_python_module_function=None,
    ):
        """Initialize FunctionDriver object.

        Args:
            function (callable, str): Function or name of example function provided by QUEENS
            external_python_module_function (Path | str): Path to external module with function
        """
        super().__init__()
        if external_python_module_function is None:
            if isinstance(function, str):
                # Try to load existing simulator functions
                my_function = example_simulator_function_by_name(function)
            else:
                my_function = function
        else:
            # Try to load external simulator functions
            my_function = get_module_attribute(external_python_module_function, function)

        # if keywords or job_id in the function's signature pass the job_id
        self.function_requires_job_id = bool(
            inspect.getfullargspec(my_function).varkw
            or "job_id" in inspect.getfullargspec(my_function).args
        )

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
                return result, gradient
            # here no gradient return
            if not result_array.shape:
                result_array = np.expand_dims(result_array, axis=0)
            return result_array, None

        return reshaped_output_function

    def run(self, sample_dict, num_procs, experiment_dir, experiment_name):
        """Run the driver.

        Args:
            sample_dict (dict): Dict containing sample and job id
            num_procs (int): number of processors
            experiment_name (str): name of QUEENS experiment.
            experiment_dir (Path): Path to QUEENS experiment directory.

        Returns:
            Result and potentially the gradient
        """
        if not self.function_requires_job_id:
            sample_dict.pop("job_id")
        results = self.function(sample_dict)
        return results