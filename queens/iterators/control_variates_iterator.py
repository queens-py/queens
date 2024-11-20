"""Implementation of Control Variates Iterator."""

import logging

import numpy as np

from queens.drivers.driver import Driver
from queens.iterators.iterator import Iterator
from queens.models.simulation_model import SimulationModel
from queens.utils.logger_settings import log_init_args

_logger = logging.getLogger(__name__)


class ControlVariatesIterator(Iterator):
    """Control varaites iterator class.

    Attributes:
        driver_main (Driver): driver for the function which you want to estimate
        driver_cvs (list(Driver)): drivers for the functions of the control variates
        parameters (Parameters): parameters to use for evaluation
        global_settings (GlobalSettings): global settings to use for evaluation
        seed (int): seed for random samples
        num_samples (int): number of samples to evaluate for estimation
        expecation_cvs (list(int)): known expected values of all control
            variates in same order as driver of control variates
        num_cvs (int): number of control variates
        output (dict): output of iterator
        result_description (dict, opt): desctiption of results of iterator
    """

    @log_init_args
    def __init__(
        self,
        driver_main,
        drivers_cvs,
        parameters,
        global_settings,
        seed,
        num_samples,
        expectation_cvs,
        scheduler,
        result_description=None,
    ):
        """Control variates iterator constructor.

        Args:
            driver_main (Driver): driver of function to be estimated
            drivers_cvs (list(Driver)): drivers of all control variates
            parameters (Parameters): parameters to be used for evaluation
            global_settings (GlobalSettings): global settings to use for evaluation
            seed (int): seed to use for samples generation
            num_samples (int): number of samples to evaluate
            expectation_cvs (list(int)): known expected value of control variates
            scheduler (Scheduler): scheduler to use for model evaluation
            result_description (dict, optional): result description for output
                of iterator. Defaults to None.

        Raises:
            ValueError: if number of expected values does not match number of
                passed control variates, or driver_cvs is not a list
        """

        if isinstance(expectation_cvs, int):
            expectation_cvs = [expectation_cvs]

        if isinstance(drivers_cvs, Driver):
            drivers_cvs = [drivers_cvs]

        if not isinstance(drivers_cvs, list):
            raise ValueError("drivers_cvs is not a Driver or list of Drivers")

        if not len(drivers_cvs) == len(expectation_cvs):
            raise ValueError(
                """number of control variates and number of expected
                              values does not match"""
            )

        super().__init__(None, parameters, global_settings)  # init parent iterator with no model
        self.seed = seed
        self.driver_main = driver_main
        self.drivers_cvs = drivers_cvs
        self.num_samples = num_samples
        self.result_description = result_description
        self.samples = None
        self.output = None
        self.num_cvs = len(drivers_cvs)
        self.expectation_cvs = expectation_cvs

        models = []

        # creating a model object for each driver object in drivers list
        #   and adding the model object to the models list
        for driver in self.drivers_cvs:
            models.append(SimulationModel(scheduler=scheduler, driver=driver))

            # add models list as attribute
            self.models_cvs = models

        self.model_main = SimulationModel(scheduler=scheduler, driver=self.driver_main)

    def __draw_samples(self, num_samples):
        """Draws samples from paramter space.

        Args:
            num_samples (list(int)): number of samples to draw on each level

        Returns:
            samples (list(np.array)): drawn samples
        """

        samples = self.parameters.draw_samples(num_samples)

        return samples

    def pre_run(self):
        """Draws samples for core_run()"""
        np.random.seed(self.seed)

        self.samples = self.__draw_samples(self.num_samples)

    def core_run(self):
        """Computes expectation estimate and variance, standard deviation of
        expectation estimate."""

        # compute main model
        computed_samples_main = np.concatenate(self.model_main.evaluate(self.samples)["result"])

        # compute control variables
        # computed_samples_cvs = np.zeros((self.num_cvs, self.num_samples))
        computed_samples_cvs = []
        for i, model in enumerate(self.models_cvs):
            # computed_samples_cvs[i, :] = model.evaluate(self.samples)['result']
            computed_samples_cvs.append(np.concatenate(model.evaluate(self.samples)["result"]))

        computed_samples_cvs = np.array(computed_samples_cvs)

        # compute covariance matrix between control variables
        cov_cvs = np.cov(computed_samples_cvs)

        breakpoint()

        cov_main = [
            np.cov(computed_samples_main, computed_samples_cvs[i])[0, 1]
            for i in range(self.num_cvs)
        ]

        breakpoint()
        alpha = np.linalg.solve(cov_cvs, cov_main)

        breakpoint()
        mean = (
            computed_samples_main
            - np.dot(
                computed_samples_cvs
                - np.tensordot(np.array(self.expectation_cvs), np.ones(self.num_samples)),
                alpha,
            )
        ).mean()

        var = computed_samples_main.var() - np.dot(cov_main, np.dot(cov_cvs, cov_main))

        self.output = {"mean": mean, "std": var**0.5}
