import logging

import numpy as np

from queens.iterators.iterator import Iterator
from queens.models.simulation_model import SimulationModel
from queens.schedulers.pool_scheduler import PoolScheduler
from queens.utils.logger_settings import log_init_args

_logger = logging.getLogger(__name__)


class MLMCIterator(Iterator):
    """Multi Level Monte Carlo Iterator for two levels.

    Attributes:
        seed (int): Seed for random number generation.
        models (list): Models of different fidelity to use for evaluation.
        num_samples (list): Number of samples to compute on each level.
        result_description (dict):  Description of desired results.
        samples (np.array):         Array with all samples on each level. Samples on level 0 from index 0 to num_samples[0]-1. Samples on level 1 from index num_samples[0] to num_samples[1]-1
        output (np.array):          Array with all model outputs.
    """

    @log_init_args
    def __init__(
        self,
        drivers,
        parameters,
        global_settings,
        seed,
        num_samples,
        result_description=None,
    ):
        """_summary_

        Args:
            drivers (_type_): _description_
            parameters (_type_): _description_
            global_settings (_type_): _description_
            seed (_type_): _description_
            num_samples (_type_): _description_
            result_description (_type_, optional): _description_. Defaults to None.

        Raises:
            ValueError: _description_
        """
        if (
            isinstance(drivers, list)
            and isinstance(num_samples, list)
            and len(num_samples) == len(drivers)
        ):
            super().__init__(
                None, parameters, global_settings
            )  # init parent iterator with no model
            self.seed = seed
            self.drivers = drivers
            self.num_samples = num_samples
            self.result_description = result_description
            self.samples = None
            self.output = None
            models = []

            scheduler = PoolScheduler(experiment_name=global_settings.experiment_name)

            # creating a model object for each driver object in drivers list and adding the model object to the models list
            for driver in drivers:
                models.append(SimulationModel(scheduler=scheduler, driver=driver))

            # add models list as attribute
            self.models = models

        else:
            raise ValueError("models and num_samples have to lists and need to be of same size")

    def pre_run(self):
        """Generate samples for subsequent MLMC analysis and update model."""
        np.random.seed(self.seed)
        self.samples = self.parameters.draw_samples(sum(self.num_samples))

    def core_run(self):
        # check if number of smaples matches the number of samples given
        if len(self.samples) != sum(self.num_samples):
            _logger.warning("number of sampels does not match number of samples given")

        # ------------------ level 0 -------------------
        mean = self.models[0].evaluate(self.samples[: self.num_samples[0]])["result"].mean()
        pivot = self.num_samples[
            0
        ]  # index to select the right samples out of sample list for each level, gets updated throughout for-loop

        breakpoint()

        # ---------------- levels 1,2,3 ... n ------------------

        for i in range(1, len(self.models)):
            samples = self.samples[
                pivot : pivot + self.num_samples[i]
            ]  # samples index by number of samples on each level and extracted from samples array

            mean += (
                self.models[i].evaluate(samples)["result"]
                - self.models[i - 1].evaluate(samples)["result"]
            ).mean()  # Difference between levels is added to the mean
            pivot += self.num_samples[i]

        self.output = {
            "result": np.array(mean),
        }

    def post_run(self):
        _logger.debug("Size of inputs %s", self.samples.shape)
        _logger.debug("Inputs %s", self.samples)
        _logger.debug("Size of outputs %s", self.output["result"].shape)
        _logger.debug("Outputs %s", self.output["result"])


"""
- scheduler als Parameter
- rais error switch if else
- check num samples decreasing (warning)
- docstring methoden
- samples
- visualization  mc vs mlmc
- commiten
- python debugger
"""
