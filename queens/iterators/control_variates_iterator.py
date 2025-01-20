#
# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (c) 2025, QUEENS contributors.
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
"""Monte Carlo Control Variates Iterator."""

import logging

import numpy as np

from queens.iterators.iterator import Iterator
from queens.utils.logger_settings import log_init_args
from queens.utils.process_outputs import write_results

_logger = logging.getLogger(__name__)


class ControlVariatesIterator(Iterator):
    r"""Monte Carlo control variates iterator.

    The control variates method in the context of Monte Carlo is used to quantify uncertainty in a
    model when input parameters are uncertain and only expressed as probability distributions. The
    Monte Carlo control variates method uses so-called low-fidelity models as control variates to
    make the quantification more precise. In the context of Monte Carlo, the control variate method
    is sometimes also called control variable method.

    The estimator for the Monte Carlo control variates method with a single control variate is
    given by
    :math:`\hat{\mu}_{f}= \underbrace{\frac{1}{N} \sum\limits_{i=1}^{N} \Big [ f(x^{(i)}) - \alpha
    \Big(g(x^{(i)}) - \hat\mu_{g} \Big) \Big]}_\textrm{cross-model estimator}`
    where :math:`f` represents the model, :math:`g` the control variate, and :math:`\hat\mu_{g}`
    the expectation of the control variate. :math:`N` represents the number of samples on the
    cross-model estimator and :math:`x^{(i)}` are random parameter samples.

    In case the mean of the control variate is known, :math:`\hat\mu_{g}` can be passed to the
    iterator as expectation_cv. Otherwise, :math:`\hat\mu_{g}` is estimated with the Monte
    Carlo method.

    The implementation is based on chapter 9.3 in [1] and uses one control variate.

    References:
        [1] D. P. Kroese, Z. I. Botev, and T. Taimre. "Handbook of Monte Carlo Methods". Wiley,
            2011.

    Attributes:
        model (Model): Main model. The uncertainties are quantified for this model.
        control_variate (Model): Control variate model.
        seed (int): Seed for random samples.
        num_samples (int): Number of samples on the cross-model estimator.
        expectation_cv (float): Expectation of the control variate. If the expectation is None,
                                it will be estimated via MC sampling.
        output (dict): Output dict with the following entries:

                       * ``mean`` (float): Cross-model estimator.
                       * ``std`` (float): Estimated standard deviation of the cross-model estimator.
                       * ``num_samples_cv`` (int): Number of samples to estimate the control variate
                                                   mean.
                       * ``mean_cv`` (float): Mean of control variate.
                       * ``std_cv_mean_estimator`` (float): Standard deviation of control variate
                                                            mean estimation.
                       * ``cv_influence_coeff`` (float): Method specific parameter that determines
                                                         the influence of the control variate.
                       * ``sample_ratio`` (float): Ratio of number of samples on control variate to
                                                   number of samples on main model. Is only part of
                                                   output if use_optimal_num_samples is True.

        num_samples_cv (int): Number of samples to use for computing the expectation
                              of the control variate if this expectation is unknown.
        samples (np.array): Samples for the control variates estimator.
        use_optimal_num_samples (bool): Determines wether the iterator calculates and uses the
                                        optimal number of samples to estimate the control variate
                                        mean such that the variance of the control variates
                                        estimator is minimized.
        cost_model (float): Cost of evaluating the model.
        cost_cv (float): Cost of evaluating the control variate.
        variance_cv_mean_estimator (float): Variance of the control variate mean estimator.
    """

    @log_init_args
    def __init__(
        self,
        model,
        control_variate,
        parameters,
        global_settings,
        seed,
        num_samples,
        expectation_cv=None,
        num_samples_cv=None,
        use_optimal_num_samples=False,
        cost_model=None,
        cost_cv=None,
    ):
        """Initialize the control variates iterator.

        Args:
            model (Model): Main model. The uncertainties are quantified for this model.
            control_variate (Model): Control variate model.
            parameters (Parameters): Parameters to use for evaluation.
            global_settings (GlobalSettings): Global settings to use for evaluation.
            seed (int): Seed for random samples.
            num_samples (int): Number of samples on the cross-model estimator.
            expectation_cv (float, opt): Expectation of the control variate. If the expectation is
                                         None, it will be estimated via MC sampling.
            num_samples_cv (int, opt): Number of samples to use for computing the expectation
                                       of the control variate if this expectation is unknown.
            use_optimal_num_samples (bool, opt): Determines wether the iterator calculates and uses
                                                 the optimal number of samples to estimate the
                                                 control variate mean such that the variance of the
                                                 control variates estimator is minimized.
            cost_model (float, opt): Cost of evaluating the model.
            cost_cv (float, opt): Cost of evaluating the control variate.

        Raises:
            ValueError: If model is None.
                        If control_variable is None.
                        If neither the expectation nor the number of samples of the control
                        variate are given and the optimal number of samples is not used.
                        If the optimal number of samples are used and the costs of the models are
                        not provided.
        """
        # Initialize parent iterator with main model.
        super().__init__(model, parameters, global_settings)

        if model is None or control_variate is None:
            raise ValueError("A model and a control variate have to be given!")

        if expectation_cv is None and num_samples_cv is None and not use_optimal_num_samples:
            raise ValueError(
                "expectation_cv or num_samples_cv has to be given when not using "
                "the optimal number of samples."
            )

        if use_optimal_num_samples and (cost_model is None or cost_cv is None):
            raise ValueError(
                "Model and control variate costs have to be given if you want to use the optimal "
                "number of samples"
            )

        self.control_variate = control_variate
        self.seed = seed
        self.num_samples = num_samples
        self.samples = None
        self.output = None
        self.expectation_cv = expectation_cv
        self.num_samples_cv = num_samples_cv
        self.use_optimal_num_samples = use_optimal_num_samples
        self.cost_model = cost_model
        self.cost_cv = cost_cv
        self.variance_cv_mean_estimator = 0

    def pre_run(self):
        """Draw samples for the core run."""
        np.random.seed(self.seed)

        self.samples = self.parameters.draw_samples(self.num_samples)

    def core_run(self):
        """Core run of iterator.

        Computes the cross-model estimator and its standard deviation.
        """
        # Evaluate models for the drawn samples.
        output_model = np.concatenate(self.model.evaluate(self.samples)["result"])
        output_cv = np.concatenate(self.control_variate.evaluate(self.samples)["result"])

        # Compute the covariance matrix between the two models.
        models_cov = np.cov(output_model, output_cv)

        cov = models_cov[0, 1]  # Covariance between the main model and the control variate
        var_model = models_cov[0, 0]  # Variance of main model
        var_cv = models_cov[1, 1]  # Variance of control variate

        # Compute expectation of control variate if it is not known.
        if self.expectation_cv is None:

            # If using the optimal number of samples, calculate the best ratio of
            # num_samples to num_samples_cv.
            if self.use_optimal_num_samples:
                # Correlation coefficient between the main model and the control variate.
                correlation_coefficient = cov / np.sqrt(var_model * var_cv)
                if correlation_coefficient >= 0.99999:
                    _logger.warning(
                        "The correlation between input models is perfect, do not use "
                        "control variates!"
                    )
                # Calculate optimal factor relating number of samples on the cross-model
                # estimator and the control variate estimator.
                sample_ratio = (
                    np.sqrt(
                        correlation_coefficient**2
                        / (1 - correlation_coefficient**2)
                        * (self.cost_model / self.cost_cv)
                    )
                    - 1
                )

                if sample_ratio <= 0:
                    raise ValueError(
                        "Optimal number of samples not possible for the chosen input models."
                    )
                self.num_samples_cv = int(sample_ratio * self.num_samples)

            # Draw samples and estimate the mean of the control variate via naive monte carlo.
            samples = self.parameters.draw_samples(self.num_samples_cv)
            results = self.control_variate.evaluate(samples)["result"]
            self.expectation_cv = results.mean()
            self.variance_cv_mean_estimator = results.var() / self.num_samples_cv

        # Calculate coefficient that determines how much the control variate influences
        # the control variate mean estimator.
        cv_influence_coeff = cov / (var_cv + self.num_samples * self.variance_cv_mean_estimator)

        # Calculate estimated mean of mean function with control variates estimator.
        mean = (
            output_model - cv_influence_coeff * output_cv
        ).mean() + cv_influence_coeff * self.expectation_cv

        # Calculate the variance of control variates estimator.
        var_estimator = 1 / self.num_samples * (var_model - cv_influence_coeff * cov)

        self.output = {
            "mean": mean,
            "std": var_estimator**0.5,
            "num_samples_cv": self.num_samples_cv,
            "mean_cv": self.expectation_cv,
            "std_cv_mean_estimator": self.variance_cv_mean_estimator**0.5,
            "cv_influence_coeff": cv_influence_coeff,
        }

        if self.use_optimal_num_samples:
            self.output["sample_ratio"] = sample_ratio

    def post_run(self):
        """Write results to result file."""
        write_results(
            processed_results=self.output, file_path=self.global_settings.result_file(".pickle")
        )
