"""Multilevel Monte Carlo implementation to use for forward uncertainty
quantification."""

import logging

import numpy as np

from queens.iterators.iterator import Iterator
from queens.models.simulation_model import SimulationModel
from queens.utils.logger_settings import log_init_args

_logger = logging.getLogger(__name__)


class MLMCIterator(Iterator):
    """Multi Level Monte Carlo Iterator.

    Attributes:
        seed (int): Seed for random number generation.
        models (list): Models of different fidelity to use for evaluation.
        num_samples (list): Number of samples to compute on each level.
        result_description (dict):  Description of desired results.
        samples (list(np.array)):   List of samples arrays for each level
        output (dict):          Dict with all iterator outputs, e.g. mean,
            standard deviation of estimator
    """

    @log_init_args
    def __init__(
        self,
        drivers,
        parameters,
        global_settings,
        seed,
        num_samples,
        scheduler,
        result_description=None,
    ):
        """Constructor MLMCIterator.

        Args:
            drivers (list(Driver)): Pass list of Drivers to evaluate
            parameters (Parameters): Pass Parameters to use to evalute the SimulationModels with
            global_settings (GlobalSettings): Pass GlobalSettings
            seed (int): Seed to use for samples generation
            num_samples (np.array(int)): Number of samples to evalute on each level
            scheduler (Scheduler): Pass scheduler to use to evaluate models
            result_description (dict, optional): Pass result description if needed.
                Defaults to None.


        Raises:
            ValueError: If num_samples and drivers are not of same length,
                drivers is not a list or num_samples is not a list
        """

        if (
            not isinstance(drivers, list)
            or not isinstance(num_samples, list)
            or not len(num_samples) == len(drivers)
        ):
            raise ValueError("models and num_samples have to lists and need to be of same size")

        super().__init__(None, parameters, global_settings)  # init parent iterator with no model
        self.seed = seed
        self.drivers = drivers
        self.num_samples = num_samples
        self.result_description = result_description
        self.samples = None
        self.output = None
        models = []

        # creating a model object for each driver object in drivers list
        #   and adding the model object to the models list
        for driver in drivers:
            models.append(SimulationModel(scheduler=scheduler, driver=driver))

            # add models list as attribute
            self.models = models

    def pre_run(self):
        """Generate samples for subsequent MLMC analysis and update model."""
        np.random.seed(self.seed)

        self.samples = []

        for i, num in enumerate(self.num_samples):
            np.random.seed(self.seed + i)
            if num == 0:
                raise ValueError("entry in num_samples cannot be 0")

            self.samples.append(self.parameters.draw_samples(num))

    def core_run(self):
        """Multi-level monte-carlo logic."""

        for i in range(1, len(self.num_samples)):
            if self.num_samples[i] - self.num_samples[i - 1] > 0:
                _logger.warning(
                    "WARNING: Number of samples not decreasing does not fit purpose of mlmc"
                )
                break

        # result list of computation on all levels
        result = []

        # ------------------ level 0 -------------------
        result.append(self.models[0].evaluate(self.samples[0])["result"])

        # ---------------- levels 1,2,3 ... n ------------------
        for i, samples in enumerate(self.samples[1:], 1):
            result.append(
                self.models[i].evaluate(samples)["result"]
                - self.models[i - 1].evaluate(samples)["result"]
            )

        self.output = {
            "result": result,
        }

    def post_run(self):
        """Data processing of raw results of model evaluations."""
        # _logger.debug("Size of inputs %s", self.samples.shape)
        # _logger.debug("Inputs %s", self.samples)
        # _logger.debug("Size of outputs %s", self.output["result"].shape)
        # _logger.debug("Outputs %s", self.output["result"])

        level_mean = []
        level_var = []

        for l, np_arr in enumerate(self.output["result"]):
            level_mean.append(np_arr.mean())

            # variance estimation via variance of sample mean estimator
            level_var.append(np_arr.var() / self.num_samples[l])

        mean = np.array(level_mean).sum()

        # total variance of estimator is the sum of the variance of each level,
        #    due to statistical independence between levels
        var = np.array(level_var).sum()

        self.output["mean"] = mean
        self.output["level_mean"] = level_mean
        self.output["var"] = var
        self.output["std"] = var**0.5
        self.output["level_var"] = level_var
