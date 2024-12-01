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
    This iterator uses one control variable.

    Attributes:
        models (list(Model)):           Models to be used for evaluation. With models[0] being the
                                        function that you want to approximate and models[1] being
                                        the control variable. Pass at two models.
        parameters (Parameters):        Parameters to use for evaluation.
        global_settings (GlobalSettings):   Global settings to use for evaluation.
        seed (int):                     Seed for random samples.
        num_samples (int):              Number of samples to evaluate for estimation.
        expecation_cvs (int)):     Here you can pass expected values of the control variate.
                                        If the expectation value of a control
                                        variate is not known, pass None as it's expected value.
        output (dict):                  Iterator outputs a dict with following entries:
            - mean (float): Estimated mean of main model
            - std (float): Calculated standard deviation of control variates estimator
        result_description (dict, opt): Desctiption of results of iterator.
        num_samples_cvs (list(int)):    Number of samples to use for computing the expectation
                                        value of control variable. If expectation of a control
                                        variableis known, you can pass None as it's number of
                                        samples.
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
            models (list(Model)):       Models to be used for evaluation. With models[0] being the
                                        function that you want to approximate and models[1] being
                                        the control variable. Pass at two models.
            parameters (Parameters):        Parameters to use for evaluation.
            global_settings (GlobalSettings):   Global settings to use for evaluation.
            seed (int):                     Seed for random samples.
            num_samples (int):              Number of samples to evaluate for estimation.
            expecation_cvs (int)):      Here you can pass expected values of the control variate.
                                        If the expectation value of a control
                                        variate is not known, pass None as it's expected value.
            result_description (dict, opt): Desctiption of results of iterator.
            num_samples_cvs (list(int)):    Number of samples to use for computing the expectation
                                        value of control variable. If expectation of a control
                                        variable is known, you can pass None as it's number of
                                        samples.

        Raises:
            ValueError:     If models is not a list of length two.
                            If the expectation or num_samples_cvs is not given when
                            not using optn.
        """
        if not isinstance(models, list):
            raise ValueError("Models have to be given in the form of a list!")
        if len(models) != 2:
            raise ValueError("Two models have to be given!")

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
        of expectation estimate.
        """
        # compute models on num_samples number of samples
        computed_samples = []
        for model in self.models:
            computed_samples.append(np.concatenate(model.evaluate(self.samples)["result"]))

        # convert list to np.array
        computed_samples = np.array(computed_samples)

        # compute the covariance matrix between the two models
        models_cov = np.cov(computed_samples)

        cov = models_cov[0, 1]  # covariance between main function and control variable
        var_model0 = models_cov[0, 0]  # variance of main function
        var_model1 = models_cov[1, 1]  # variance of control variable
        rho = cov / np.sqrt(var_model0 * var_model1)  # correlation coefficient between main and cv

        # Compute expecation of control variable, if it is not known
        # Using simple monte carlo simulation
        if self.expectation_cvs is None:

            # if optn is true, calculate the best ratio of num_samples to num_samples_cvs
            if self.use_optn is True:
                if rho >= 0.99999:
                    raise ValueError(
                        """The correlation between input models is perfect, do not use
                        control variates!"""
                    )
                # calculate optimal factor relating number of samples on main estimator and
                # monte carlo estimator
                beta = (
                    np.sqrt(rho**2 / (1 - rho**2) * (self.models_cost[0] / self.models_cost[1])) - 1
                )

                if beta <= 0:
                    raise ValueError(
                        """Optimal n for input models not possible, due to control
                        variates approach not being sensible for these input models"""
                    )
                self.num_samples_cvs = int(beta * self.num_samples)

            # draw samples and estimate the mean of the control variable via naive monte carlo
            samples = self.__draw_samples(self.num_samples_cvs)
            results = self.models[1].evaluate(samples)["result"]
            self.expectation_cvs = results.mean()
            self.variance_cvs = (
                results.var() / self.num_samples_cvs
            )  # sample mean estimator variance

        # calculate coefficient that determines how much the control variable influences
        # the control variate mean estimator
        alpha = cov / (var_model1 + self.num_samples * self.variance_cvs)

        # calculate estimated mean of mean function with control variates estimator
        mean = (
            computed_samples[0] - alpha * computed_samples[1]
        ).mean() + alpha * self.expectation_cvs

        # calculate the variance of control variates estimator
        var_estimator = 1 / self.num_samples * (var_model0 - alpha * cov)

        self.output = {
            "mean": mean,
            "std": var_estimator**0.5,
            "num_samples_cvs": self.num_samples_cvs,
            "mean_cv": self.expectation_cvs,
            "std_cv": self.variance_cvs**0.5,
            "alpha": alpha,
        }

        if self.use_optn is True:
            self.output["beta"] = beta

    def post_run(self):
        """Writes results to result file."""
        write_results(
            processed_results=self.output, file_path=self.global_settings.result_file(".pickle")
        )
