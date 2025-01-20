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
"""Multilevel Monte Carlo Iterator."""

import logging

import numpy as np

from queens.distributions.uniform_discrete import UniformDiscreteDistribution
from queens.iterators.iterator import Iterator
from queens.utils.logger_settings import log_init_args
from queens.utils.process_outputs import write_results

_logger = logging.getLogger(__name__)


class MLMCIterator(Iterator):
    r"""Multilevel Monte Carlo Iterator.

    The equations were taken from [1]. This iterator can be used in two different modes by setting
    the truth value of the parameter use_optimal_num_samples. When set to false, the iterator uses
    the number of samples provided by the user. When set to true, the iterator calculates and uses
    the optimal number of samples on each estimator. The iterator does this by calculating the
    optimal ratio of samples between the estimators. The number of samples on the highest-fidelity
    model is set by the user.

    The multilevel Monte Carlo (MLMC) estimator is given by
    :math:`\hat{\mu}_\mathrm{MLMC} = \underbrace{\frac{1}{N_{0}} \sum_{i=1}^{N_{0}} f_{0}(x^{(0,
    i)})}_\textrm{estimator 0} + \sum_{l=1}^{L} \underbrace{\bigg \{ \frac{1}{N_{l}} \sum_{i=1}^{N_
    {l}} \Big (  f_{l}(x^{(l, i)}) - f_{l-1}(x^{(l, i)})  \Big ) \bigg \}}_{\textrm{estimator }l}`
    where :math:`f_{l}` are the models with increasing fidelity as :math:`l` increases.
    :math:`N_{l}` are the number of samples on the :math:`l`-th estimator and :math:`x^{(l,i)}` is
    the :math:`i`-th sample on the :math:`l`-th estimator.

    References:
        [1] M. B. Giles. "Multilevel Monte Carlo methods". Acta Numerica, 2018.

    Attributes:
        seed (int): Seed for random number generation.
        models (list(Model)): Models of different fidelity to use for evaluation. The model
                              fidelity and model cost increases with increasing index.
        num_samples (list(int)): The number of samples to evaluate each estimator with. If
                                 use_optimal_num_samples is False (default), the values represent
                                 the final number of model evaluations on each estimator. If
                                 use_optimal_num_samples is True, the values represent the
                                 initial number of model evaluations on each estimator needed to
                                 estimate the variance of each estimator, after which the optimal
                                 number of samples of each estimator is computed. The i-th entry of
                                 the list corresponds to the i-th estimator.
        samples (list(np.array)): List of samples for each estimator.
        output (dict): Output dict with the following entries:

                        * ``mean`` (float): MLMC estimator.
                        * ``var`` (float): Variance of the MLMC estimator.
                        * ``std`` (float): Standard deviation of the MLMC estimator.
                        * ``result`` (np.array): Evaluated samples of each estimator.
                        * ``mean_estimators`` (list): Estimated mean of each estimator.
                        * ``var_estimators`` (list): Variance of each estimator.
                        * ``num_samples`` (list): Number of evaluated samples of each estimator.
                        * ``std_bootstrap`` (float): Bootstrap approximation of the calculated MLMC
                                                     estimator standard deviation. This value is not
                                                     computed if num_bootstrap_samples is 0.

        cost_estimators (list(float)): The relative cost of each estimator. The i-th
                                       entry of the list corresponds to the i-th estimator.
        use_optimal_num_samples (bool): Sets the mode of the iterator to either using num_samples as
                                        the number of model evaluations on each estimator or using
                                        num_samples as initial samples to calculate the optimal
                                        number of samples from.
        num_bootstrap_samples (int): Number of resamples to use for bootstrap estimate of
                                     standard deviation of this estimator. If set to 0, the
                                     iterator won't compute a bootstrap estimate.
    """

    @log_init_args
    def __init__(
        self,
        models,
        parameters,
        global_settings,
        seed,
        num_samples,
        cost_models=None,
        use_optimal_num_samples=False,
        num_bootstrap_samples=0,
    ):
        """Initialize the multilevel Monte Carlo iterator.

        Args:
            models (list(Model)): Models of different fidelity to use for evaluation. The model
                                  fidelity and model cost increases with increasing index.
            parameters (Parameters): Parameters with which to evaluate the Models.
            global_settings (GlobalSettings): Global settings of the QUEENS experiment.
            seed (int): Seed to use for samples generation
            num_samples (list(int)): Number of samples to evaluate on each estimator or initial
                                     number of model evaluations if use_optimal_num_samples is True.
            cost_models (list(float), optional): List with the relative cost of each model. Only
                                                 needed if use_optimal_num_samples is True.
            use_optimal_num_samples (bool, optional): Sets the mode of the iterator to either use
                                                      num_samples as the number of model
                                                      evaluations on each estimator or use
                                                      num_samples as initial samples to calculate
                                                      the optimal number of samples from.
            num_bootstrap_samples (int, optional): Number of resamples to use for bootstrap
                                                   estimate of standard deviation of this
                                                   estimator. If set to 0, the iterator won't
                                                   compute a bootstrap estimate.

        Raises:
            ValueError: If num_samples and models are not of same length.
                        If models is not a list or num_samples is not a list.
        """
        # Initialize parent iterator with no model.
        super().__init__(None, parameters, global_settings)

        if not isinstance(models, list):
            raise ValueError("Models have to be passed in form of a list!")

        if not isinstance(num_samples, list):
            raise ValueError("Num_samples have to be passed in form of a list!")

        if not len(num_samples) == len(models):
            raise ValueError("models and num_samples have to be lists of same size!")

        if use_optimal_num_samples:
            if cost_models is None:
                raise ValueError(
                    "cost_models needs to be specified to use optimal number of samples"
                )

            # The cost of one sample evaluation of estimator i is the sum of the cost of model i
            # and model i-1, since both have to be evaluated to compute the expectation of the i-th
            # estimator.
            self.cost_estimators = [cost_models[0]] + [
                cost_models[i] + cost_models[i - 1] for i in range(1, len(cost_models))
            ]

        self.seed = seed
        self.models = models
        self.num_samples = num_samples
        self.samples = None
        self.output = None
        self.use_optimal_num_samples = use_optimal_num_samples
        self.num_bootstrap_samples = num_bootstrap_samples

        # Test if number of samples is decreasing with increasing index.
        for i in range(1, len(self.num_samples)):
            if self.num_samples[i] - self.num_samples[i - 1] > 0:
                _logger.warning(
                    "WARNING: Number of samples does not decrease with increasing index. This does "
                    "not fulfill the purpose of multilevel Monte Carlo."
                )
                break

    def _draw_samples(self, num_samples):
        """Draw samples from the parameter space.

        Args:
            num_samples (list(int)): Number of samples to draw for each estimator.

        Returns:
            samples (list(np.array)): Drawn samples for each estimator.
        """
        samples = []

        for num in num_samples:
            samples.append(self.parameters.draw_samples(num))

        return samples

    def _compute_estimator_statistics(self, results_estimators):
        """Computes mean and variance for each estimator.

        Args:
            results_estimators (list(np.array)): List of results for each estimator.

        Returns:
            np.array: Mean of each estimator.
            np.array: Variance of each estimator.
            float: Mean of the MLMC estimator.
            float: Standard deviation of the MLMC estimator.
        """
        mean_estimators = []
        var_estimators = []

        # Initialize mean and variance of the MLMC estimator.
        mean = 0
        var = 0

        for result in results_estimators:
            # Calculate mean of estimators via samples mean.
            mean_estimators.append(result.mean())

            # Calculate variance of estimators via sample mean variance.
            var_estimators.append(result.var())

            # Update MLMC mean.
            mean += mean_estimators[-1]
            # Update MLMC variance.
            var += var_estimators[-1] / result.size

        std = var**0.5

        return np.array(mean_estimators), np.array(var_estimators), mean, std

    def pre_run(self):
        """Generate samples for subsequent MLMC analysis."""
        np.random.seed(self.seed)

        self.samples = self._draw_samples(self.num_samples)

    def core_run(self):
        """Perform multilevel Monte Carlo analysis."""
        # List of all estimator results.
        results_estimators = []

        # Estimator 0
        results_estimators.append(self.models[0].evaluate(self.samples[0])["result"])

        # Estimator 1,2,3 ... n
        for i, samples in enumerate(self.samples[1:], 1):
            results_estimators.append(
                self.models[i].evaluate(samples)["result"]
                - self.models[i - 1].evaluate(samples)["result"]
            )

        mean_estimators, var_estimators, mean, std = self._compute_estimator_statistics(
            results_estimators
        )

        if self.use_optimal_num_samples:

            ideal_num_samples = np.array(self.num_samples)

            # Iterate over all estimators except for the last one in reversed order
            # to update the optimal number of samples on each estimator using
            # n_{l-1} = n_{l} * \sqrt{\frac{Var_{l-1}*Cost_{l}}{Var_{l}*Cost_{l-1}}}.
            for i in reversed(range(len(self.num_samples) - 1)):

                # The samples ratio relates the number of samples for estimators via
                # n_l = sample_ratio * n_{l+1}
                sample_ratio = np.sqrt(
                    (var_estimators[i] * self.cost_estimators[i + 1])
                    / (var_estimators[i + 1] * self.cost_estimators[i])
                )

                ideal_num_samples[i] = int(ideal_num_samples[i + 1] * sample_ratio)

            # Calculate the difference between the current number of samples and the ideal
            # number of samples. These are the additional samples that have to be computed.
            # If this value is negative, choose 0.
            num_samples_additional = np.maximum(
                np.array(ideal_num_samples - self.num_samples), np.zeros(len(self.num_samples))
            ).astype(int)

            additional_samples = self._draw_samples(num_samples_additional)

            # Iteration on estimator 0
            results_estimators[0] = np.concatenate(
                (results_estimators[0], self.models[0].evaluate(additional_samples[0])["result"])
            )

            # Iteration on estimators 1,2,3 ... n
            for i, samples in enumerate(additional_samples[1:], 1):
                if num_samples_additional[i] > 0:
                    results_estimators[i] = np.concatenate(
                        (
                            results_estimators[i],
                            self.models[i].evaluate(samples)["result"]
                            - self.models[i - 1].evaluate(samples)["result"],
                        )
                    )

            # Update num_samples with additional samples and update estimator statistics.
            self.num_samples = self.num_samples + num_samples_additional
            mean_estimators, var_estimators, mean, std = self._compute_estimator_statistics(
                results_estimators
            )

        self.output = {
            "mean": mean,
            "var": std**2,
            "std": std,
            "result": results_estimators,
            "mean_estimators": mean_estimators,
            "var_estimators": var_estimators,
            "num_samples": self.num_samples,
        }

        if self.num_bootstrap_samples > 0:
            self.output["std_bootstrap"] = self._bootstrap(results_estimators)

    def post_run(self):
        """Write results to result file."""
        write_results(
            processed_results=self.output, file_path=self.global_settings.result_file(".pickle")
        )

    def _bootstrap(self, results_estimators):
        """Bootstrapping standard deviation estimate.

        The bootstrap estimate approximates the standard deviation of the MLMC estimator calculated
        by the iterator. The accuracy of the bootstrap approximation increases for an increasing
        number of bootstrap samples.

        Args:
            results_estimators (list(np.array)): Results of the core run for each estimator.

        Returns:
            float: Standard deviation of the MLMC estimator using bootstrapping.
        """
        var_estimate_bootstrap = 0
        for result in results_estimators:
            dist = UniformDiscreteDistribution(
                np.arange(stop=result.size, dtype=int).reshape(-1, 1)
            )

            bootstrap_sample_mean = np.zeros(self.num_bootstrap_samples)
            for i in range(self.num_bootstrap_samples):
                bootstrap_sample_mean[i] = result[dist.draw(result.size)].mean()

            var_estimate_bootstrap += bootstrap_sample_mean.var()

        return var_estimate_bootstrap**0.5
