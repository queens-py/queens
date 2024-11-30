"""Implementation of Control Variates Iterator."""

import logging

import numpy as np

from queens.iterators.iterator import Iterator
from queens.utils.logger_settings import log_init_args
from queens.utils.process_outputs import write_results

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
        use_optn=False,
        models_cost=None,
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
        if not isinstance(models, list):
            raise ValueError("Models have to be given in the form of a list!")
        if len(models) < 2:
            raise ValueError("At least two models have to be given!")

        if expectation_cvs is None and num_samples_cvs is None and use_optn is False:
            raise ValueError(
                "expectation_cvs or num_samples_cvs has to be given, when not using optn"
            )

        if use_optn is True and models_cost is None:
            raise ValueError("model cost has to be given, you want to use optn")

        super().__init__(None, parameters, global_settings)  # init parent iterator with no model
        self.models = models
        self.seed = seed
        self.num_samples = num_samples
        self.result_description = result_description
        self.samples = None
        self.output = None
        self.expectation_cvs = expectation_cvs
        self.num_samples_cvs = num_samples_cvs
        self.use_optn = use_optn
        self.models_cost = models_cost

        if expectation_cvs is not None:
            self.variance_cvs = 0
        else:
            self.variance_cvs = None

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
        # compute models
        computed_samples = []
        for model in self.models:
            computed_samples.append(np.concatenate(model.evaluate(self.samples)["result"]))

        computed_samples = np.array(computed_samples)

        models_cov = np.cov(computed_samples)

        cov = models_cov[0, 1]
        var0 = models_cov[0, 0]
        var1 = models_cov[1, 1]
        rho = cov / np.sqrt(var0 * var1)

        # Compute expecation of control variable, if it is not known
        # Using simple monte carlo simulation
        if self.expectation_cvs is None:
            if self.use_optn is True:
                # calculate optimal factor relating number of samples on main estimator and
                # monte carlo estimator
                beta = np.sqrt(
                    rho**2
                    * (self.models_cost[0] / self.models_cost[1] + 1)
                    / (self.num_samples * (1 - rho**2))
                )
                self.num_samples_cvs = int(beta * self.num_samples)

            samples = self.__draw_samples(self.num_samples_cvs)
            results = self.models[1].evaluate(samples)["result"]
            self.expectation_cvs = results.mean()
            self.variance_cvs = results.var() / self.num_samples_cvs

        alpha = cov / (models_cov[1, 1] + self.variance_cvs)
        mean = (
            computed_samples[0] - alpha * computed_samples[1]
        ).mean() + alpha * self.expectation_cvs

        var_estimator = models_cov[0, 0] - cov**2 / (models_cov[1, 1] + self.variance_cvs)
        var_estimator *= 1 / self.num_samples

        self.output = {
            "mean": mean,
            "std": var_estimator**0.5,
            "num_samples_cvs": self.num_samples_cvs,
            "std_cvs": self.variance_cvs**0.5,
            "alpha": alpha,
        }

    def post_run(self):
        """Writes results to result file."""
        write_results(
            processed_results=self.output, file_path=self.global_settings.result_file(".pickle")
        )
