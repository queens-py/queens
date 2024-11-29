"""Implementation of Control Variates Iterator."""

import logging

import numpy as np

from queens.iterators.iterator import Iterator
from queens.utils.logger_settings import log_init_args

_logger = logging.getLogger(__name__)


class ControlVariatesIterator(Iterator):
    """Control varaites iterator class.

    The implementation of control variates is based on chapter
    9.3 of "Handbook of Monte Carlo Methods" written by Kroese, Taimre and Botev.

    Attributes:
        models (list(Model)):           Models to be used for evaluation. With models[0] being the
                                        function that you want to approximate and models[1:] being
                                        the control variates. Pass at least two models.
        parameters (Parameters):        Parameters to use for evaluation.
        global_settings (GlobalSettings):   Global settings to use for evaluation.
        seed (int):                     Seed for random samples.
        num_samples (int):              Number of samples to evaluate for estimation.
        expecation_cvs (list(int)):     Here you can pass expected values of control variates with
                                        expectation[i] corresponding to models[i+1] (the i-th
                                        control variate). If the expectation value of a control
                                        variate is not known, pass None as it's expected value.
        num_cvs (int):                  Number of control variates.
        output (dict):                  Iterator outputs a dict with following entries:
            - mean (float): Estimated mean of main model
            - std (float): Calculated standard error of estimator, ASSUMING expected values to be
            exact, even if computed. This value can only trusted, if given expectation values are
            exact and none are estimated by control variates iterator!
        result_description (dict, opt): Desctiption of results of iterator.
        num_samples_cvs (list(int)):    Number of samples to use for computing the expectation
                                        value of each control variable. With num_samples_cvs[i]
                                        corresponding to models[i+1] (the i-th control
                                        varaible). If expectation of a control variable is known,
                                        you can pass None as it's number of samples.
        samples (np.array):             Samples, on which all models are evaluated on.
    """

    @log_init_args
    def __init__(
        self,
        models,
        parameters,
        global_settings,
        seed,
        num_samples,
        expectation_cvs=None,
        result_description=None,
        num_samples_cvs=None,
    ):
        """Control variates iterator constructor.

        Args:
            models (list(Model)):           Models to be used for evaluation. With models[0] being
                                            the function that you want to approximate and models
                                            [1:] being the control variates. Pass at least two
                                            models.
            parameters (Parameters):        Parameters to use for evaluation.
            global_settings (GlobalSettings):   Global settings to use for evaluation.
            seed (int):                     Seed for random samples.
            num_samples (int):              Number of samples to evaluate for estimation.
            expecation_cvs (list(int), opt):    Here you can pass expected values of control
                                                variates
                                            with expectation[i] corresponding to models[i+1] (the
                                            i-th control variate). If the expectation value of a
                                            control variate is not known, pass None as it's
                                            expected value.
            result_description (dict, opt): Desctiption of results of iterator.
            num_samples_cvs (list(int), optional):    Number of samples to use for computing the
                                            expectation value of each control variable. With
                                            num_samples_cvs[i] corresponding to models[i+1] (the
                                            i-th control varaible). If expectation of a control
                                            variable is known, you can pass None as it's number of
                                            samples.

        Raises:
            ValueError: if number of expected values does not match number of
                passed control variates, or driver_cvs is not a list.
        """
        if expectation_cvs is None:
            expectation_cvs = [None for i in range(len(models) - 1)]
        if isinstance(expectation_cvs, int):
            expectation_cvs = [expectation_cvs]

        if num_samples_cvs is None:
            num_samples_cvs = [None for i in range(len(models) - 1)]

        if not isinstance(models, list):
            raise ValueError("Models have to be given in the form of a list!")
        if len(models) < 2:
            raise ValueError("At least two models have to be given!")

        if not len(models) - 1 == len(expectation_cvs):
            raise ValueError(
                """number of control variates and number of expected
                              values does not match"""
            )

        if not len(models) - 1 == len(num_samples_cvs):
            raise ValueError(
                """
                Length of num_samples_cvs has to be equal to len(models)-1
            """
            )

        for i in range(len(models) - 1):
            if expectation_cvs[i] is None and num_samples_cvs[i] is None:
                raise ValueError(
                    """
                    For each control variate either a expectation value has to be given
                    or a number of samples, for the computation of it's expectation value
                """
                )

        super().__init__(None, parameters, global_settings)  # init parent iterator with no model
        self.models = models
        self.seed = seed
        self.num_samples = num_samples
        self.result_description = result_description
        self.samples = None
        self.output = None
        self.num_cvs = len(models) - 1
        self.expectation_cvs = expectation_cvs
        self.num_samples_cvs = num_samples_cvs

    def __draw_samples(self, num_samples):
        """Draws samples from paramter space.

        Args:
            num_samples (list(int)): Number of samples to draw on each level.

        Returns:
            samples (list(np.array)): Drawn samples.
        """
        samples = self.parameters.draw_samples(num_samples)

        return samples

    def pre_run(self):
        """Draws samples for core_run()."""
        np.random.seed(self.seed)

        self.samples = self.__draw_samples(self.num_samples)

    def core_run(self):
        """Core run of iterator.

        Computes expectation estimate and variance, standard deviation
        of expectation estimate. The implementation is based on chapter
        9.3 of "Handbook of Monte Carlo Methods" written by Kroese,
        Taimre and Botev.
        """
        # Compute expecation of control variables, if none are given
        # Using simple monte carlo simulation
        for i, expectation in enumerate(self.expectation_cvs):
            if expectation is None:

                samples = self.__draw_samples(self.num_samples_cvs[i])

                self.expectation_cvs[i] = self.models[i + 1].evaluate(samples)["result"].mean()

        # Check if expectation values for all control variables are now present
        for expectation in self.expectation_cvs:
            if expectation is None:
                raise ValueError(
                    """
                    Expectation values of control variables could be successfully
                    computed"""
                )

        # compute models
        computed_samples = []
        for i, model in enumerate(self.models):
            computed_samples.append(np.concatenate(model.evaluate(self.samples)["result"]))

        computed_samples = np.array(computed_samples)

        models_cov = np.cov(computed_samples)

        # compute covariance matrix between control variables
        cvs_cov = models_cov[1:, 1:]

        main_cov = models_cov[0, 1:]

        alpha = np.linalg.solve(cvs_cov, main_cov)

        expectation_tensor = np.tensordot(
            np.array(self.expectation_cvs), np.ones(self.num_samples), axes=0
        )
        correction = np.dot(alpha, computed_samples[1:] - expectation_tensor)
        mean = (computed_samples[0] - correction).mean()

        var_main_samples = models_cov[0, 0]

        coefficient_of_multiple_correlation = (
            np.dot(np.array(main_cov).T, np.dot(np.linalg.inv(cvs_cov), np.array(main_cov)))
            / var_main_samples
        )

        var_estimator = (
            1 / self.num_samples * (1 - coefficient_of_multiple_correlation) * var_main_samples
        )

        self.output = {"mean": mean, "std": var_estimator**0.5}
