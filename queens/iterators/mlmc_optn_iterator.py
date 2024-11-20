"""Implementation of multilevel Monte Carlo iterator with user prescriped
standard deviation tolerance and optimal number of evaluations of each
level."""

import logging

import numpy as np

from queens.iterators.iterator import Iterator
from queens.models.simulation_model import SimulationModel
from queens.utils.logger_settings import log_init_args

_logger = logging.getLogger(__name__)


class MLMCOptNIterator(Iterator):
    """Multilevel monte carlo iterator with user prescribed standard deviation
    tolerance and optimal number of evaluated samples on each level.

    Attributes:
        seed (int): Seed for random number generation.
        models (list): Models of different fidelity to use for evaluation.
        num_samples (list): Initial number of samples to compute on each level.
        result_description (dict):  Description of desired results.
        samples (list(np.array)):   List of samples arrays for each level
        std_tolerance (float): Prescribed tolerance for computed standard deviation
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
        num_samples_initial,
        scheduler,
        std_tolerance,
        cost_models,
        result_description=None,
    ):

        if (
            not isinstance(drivers, list)
            or not isinstance(num_samples_initial, list)
            or not len(num_samples_initial) == len(drivers)
        ):
            raise ValueError("Models and num_samples have to lists and need to be of same size!")

        for num in num_samples_initial:
            if num <= 0:
                raise ValueError("Number of samples for each model has to be greater than zero!")

        super().__init__(None, parameters, global_settings)  # init parent iterator with no model
        self.seed = seed
        self.drivers = drivers
        self.num_samples = num_samples_initial
        self.result_description = result_description
        self.samples = None
        self.output = None
        self.std_tolerance = std_tolerance

        # The cost of level i is the sum of the cost of model i and model i-1,
        #   since both have to be evaluated to compute the expecation of level i
        self.level_cost = [cost_models[0]] + [
            cost_models[i] + cost_models[i - 1] for i in range(1, len(cost_models))
        ]

        models = []

        # creating a model object for each driver object in drivers list and
        #   adding the model object to the models list
        for driver in drivers:
            models.append(SimulationModel(scheduler=scheduler, driver=driver))

            # add models list as attribute
            self.models = models

    # computes mean, var and std for each level to check convergence
    def __compute_level_statistics(self, level_data):
        """Computes mean and variance for each level.

        Args:
            level_data (list(np.array)): list of np.array results for each level

        Returns:
            (level_mean (np.array), level_var (np.array)), mean, std): A tupel of
                four entries, the first one being the mean of each level and the
                second one being the variance of each level, the third one being
                the mean of the mlmc estimtor and the fourth one being the standard
                deviation of the mlmc estimator
        """
        level_mean = []
        level_var = []
        var = 0
        mean = 0

        for np_arr in level_data:
            # mean estimation of level via sample mean estimator
            level_mean.append(np_arr.mean())

            # variance estimation of level via sample variance estimator
            level_var.append(np_arr.var())

            # update mean
            mean += np_arr.mean()
            var += np_arr.var() / np_arr.size

        std = var**0.5

        return (np.array(level_mean), np.array(level_var), mean, std)

    def __draw_samples(self, num_samples):
        """Draws samples from paramter space.

        Args:
            num_samples (list(int)): number of samples to draw on each level

        Returns:
            samples (list(np.array)): Drawn samples on each level
        """

        samples = []

        for num in num_samples:
            samples.append(self.parameters.draw_samples(num))

        return samples

    def pre_run(self):
        np.random.seed(self.seed)

        self.samples = self.__draw_samples(self.num_samples)

    def core_run(self):

        for i in range(1, len(self.num_samples)):
            if self.num_samples[i] - self.num_samples[i - 1] > 0:
                _logger.warning(
                    """WARNING: Number of samples not decreasing does not
                                fit purpose of mlmc"""
                )
                break

        # result list of computation on all levels
        result = []

        # ------------------ first iteration of level 0 -------------------
        result.append(self.models[0].evaluate(self.samples[0])["result"])

        # ---------------- first iteration of levels 1,2,3 ... n ------------------
        for i, samples in enumerate(self.samples[1:], 1):
            result.append(
                self.models[i].evaluate(samples)["result"]
                - self.models[i - 1].evaluate(samples)["result"]
            )

        # ---more itereations on all levels, until std_tolerance requirement is met ---
        (level_mean, level_var, mean, std) = self.__compute_level_statistics(level_data=result)

        while std > self.std_tolerance:
            # using the formula proposed by Giles in Multilevel Monte Carlo methods (2018)
            #    to compute optimal num_levels
            mu = (self.std_tolerance * 0.9) ** (-2) * np.dot(
                level_var**0.5, np.array(self.level_cost) ** 0.5
            )
            ideal_num_samples = np.ceil(
                [mu * (level_var[i] / self.level_cost[i]) ** 0.5 for i in range(len(self.models))]
            ).astype(int)

            num_samples_additional = np.maximum(
                np.array(ideal_num_samples - self.num_samples), np.zeros(len(self.num_samples))
            ).astype(int)

            print(num_samples_additional)

            self.samples = self.__draw_samples(num_samples_additional)

            # ------------------ iteration on level 0 -------------------
            result[0] = np.concatenate(
                (result[0], self.models[0].evaluate(self.samples[0])["result"])
            )
            # result[0].append(self.models[0].evaluate(self.samples[0])['result'])

            # ---------------- iteration on levels 1,2,3 ... n ------------------
            for i, samples in enumerate(self.samples[1:], 1):
                result[i] = np.concatenate(
                    (
                        result[i],
                        self.models[i].evaluate(samples)["result"]
                        - self.models[i - 1].evaluate(samples)["result"],
                    )
                )
                # result.append(self.models[i].evaluate(samples)['result'] - self.models[i-1].
                #   evaluate(samples)['result'])

            self.num_samples = self.num_samples + num_samples_additional
            level_mean, level_var, mean, std = self.__compute_level_statistics(level_data=result)

        print(self.num_samples)

        self.output = {
            "result": result,
            "mean": mean,
            "std": std,
            "level_mean": level_mean,
            "level_var": level_var,
        }

    # def post_run(self):
    # _logger.debug("Size of inputs %s", self.samples.shape)
    # _logger.debug("Inputs %s", self.samples)
    # _logger.debug("Size of outputs %s", self.output["result"].shape)
    # _logger.debug("Outputs %s", self.output["result"])
