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
"""Iterator for Sobol indices with GP uncertainty."""

import logging
import multiprocessing as mp
import time

from queens.iterators.sobol_index_gp_uncertainty.estimator import (
    SobolIndexEstimator,
    SobolIndexEstimatorThirdOrder,
)
from queens.iterators.sobol_index_gp_uncertainty.predictor import Predictor
from queens.iterators.sobol_index_gp_uncertainty.sampler import Sampler, ThirdOrderSampler
from queens.iterators.sobol_index_gp_uncertainty.statistics import (
    StatisticsSecondOrderEstimates,
    StatisticsSobolIndexEstimates,
    StatisticsThirdOrderSobolIndexEstimates,
)
from queens.utils.logger_settings import log_init_args
from queens.utils.process_outputs import write_results

from .iterator import Iterator

logging.getLogger("matplotlib").setLevel(logging.CRITICAL)
_logger = logging.getLogger(__name__)


class SobolIndexGPUncertaintyIterator(Iterator):
    """Iterator for Sobol indices with metamodel uncertainty.

    This iterator estimates first- and total-order Sobol indices based on Monte-Carlo integration
    and the use of Gaussian process as surrogate model. Additionally, uncertainty estimates for the
    Sobol index estimates are calculated: total uncertainty and separate uncertainty due to
    Monte-Carlo integration and due to the use of the Gaussian process as a surrogate model.
    Second-order indices can optionally be estimated.

    Alternatively, one specific third-order Sobol index can be estimated for one specific
    combination of three parameters (specified as *third_order_parameters* in the input file).

    The approach is based on:

    Le Gratiet, Loic, Claire Cannamela, and Bertrand Iooss. ‘A Bayesian Approach
    for Global Sensitivity Analysis of (Multifidelity) Computer Codes’. SIAM/ASA Journal on
    Uncertainty Quantification 2, no. 1 (1 January 2014): 336–63.
    https://doi.org/10.1137/130926869.

    Further details can be found in:

    Wirthl, B., Brandstaeter, S., Nitzler, J., Schrefler, B. A., & Wall, W. A. (2023). Global
    sensitivity analysis based on Gaussian-process metamodelling for complex biomechanical problems.
    International Journal for Numerical Methods in Biomedical Engineering, 39(3), e3675.
    https://doi.org/10.1002/cnm.3675


    Attributes:
        result_description (dict): Dictionary with desired result description.
        num_procs (int): Number of processors.
        sampler (Sampler object): Sampler object.
        predictor (Predictor object): Metamodel predictor object.
        index_estimator (SobolIndexEstimator object): Estimator object.
        statistics (list): List of statistics objects.
        calculate_second_order (bool): *True* if second-order indices are calculated.
        calculate_third_order (bool): *True* if third-order indices only are calculated.
        results (dict): Dictionary for results.
    """

    @log_init_args
    def __init__(
        self,
        model,
        parameters,
        global_settings,
        result_description,
        num_procs=mp.cpu_count() - 2,
        second_order=False,
        third_order=False,
        **additional_options,  # Using kwargs should only be an intermediate solution.
    ):
        """Initialize Sobol index iterator with GP uncertainty.

        Args:
            model (Model): Model to be evaluated by iterator
            parameters (Parameters): Parameters object
            global_settings (GlobalSettings): settings of the QUEENS experiment including its name
                                              and the output directory
            result_description (dict): dictionary with desired result description
            num_procs (int, opt): number of processors
            second_order (bool, opt): true if second-order indices are calculated
            third_order (bool, opt): true if third-order indices only are calculated
            additional_options: Additional keyword arguments.
        """
        super().__init__(model, parameters, global_settings)

        additional_options["second_order"] = second_order
        additional_options["third_order"] = third_order

        sampler_method, estimator_method = self._choose_helpers(third_order)
        sampler = sampler_method.from_config_create(
            additional_options, self.parameters.names, self.parameters
        )
        index_estimator = estimator_method.from_config_create(
            additional_options, self.parameters.names
        )
        predictor = Predictor.from_config_create(additional_options, model)

        statistics = []
        if third_order:
            statistics.append(
                StatisticsThirdOrderSobolIndexEstimates.from_config_create(
                    additional_options, self.parameters.names
                )
            )
        else:
            statistics.append(
                StatisticsSobolIndexEstimates.from_config_create(
                    additional_options, self.parameters.names
                )
            )
            if second_order:
                statistics.append(
                    StatisticsSecondOrderEstimates.from_config_create(
                        additional_options, self.parameters.names
                    )
                )

        _logger.info("Calculate second-order indices is %s", second_order)

        self.result_description = result_description
        self.num_procs = num_procs
        self.sampler = sampler
        self.predictor = predictor
        self.index_estimator = index_estimator
        self.statistics = statistics
        self.calculate_second_order = second_order
        self.calculate_third_order = third_order
        self.results = {}

    def pre_run(self):
        """Pre-run."""

    def core_run(self):
        """Core-run."""
        self.model.build_approximation()

        self.calculate_index()

    def post_run(self):
        """Post-run."""
        if self.result_description is not None:
            if self.result_description["write_results"]:
                write_results(self.results, self.global_settings.result_file(".pickle"))

    def calculate_index(self):
        """Calculate Sobol indices.

        Run sensitivity analysis based on:

        Le Gratiet, Loic, Claire Cannamela, and Bertrand Iooss. ‘A Bayesian Approach
        for Global Sensitivity Analysis of (Multifidelity) Computer Codes’. SIAM/ASA Journal on
        Uncertainty Quantification 2, no. 1 (1 January 2014): 336–63.
        https://doi.org/10.1137/130926869.
        """
        start_run = time.time()

        # 1. Generate Monte-Carlo samples
        samples = self.sampler.sample()
        # 2. Samples realizations of metamodel at Monte-Carlo samples
        prediction = self.predictor.predict(samples)
        # 3. Calculate Sobol index estimates (including bootstrap)
        estimates = self.index_estimator.estimate(prediction, self.num_procs)

        # 4. Evaluate statistics
        self.evaluate_statistics(estimates)

        _logger.info("Time for full calculation: %s", time.time() - start_run)

    def evaluate_statistics(self, estimates):
        """Evaluate statistics of Sobol index estimates.

        Args:
            estimates (dict): Dictionary of Sobol index estimates of different order
        """
        if self.calculate_third_order:
            self.results["third_order"] = self.statistics[0].evaluate(estimates["third_order"])
            _logger.info(str(self.results["third_order"]))
        else:
            _logger.info("First-order Sobol indices:")
            self.results["first_order"] = self.statistics[0].evaluate(estimates["first_order"])
            _logger.info("Total-order Sobol indices:")
            self.results["total_order"] = self.statistics[0].evaluate(estimates["total_order"])

            if self.calculate_second_order:
                _logger.info("Second-order Sobol indices:")
                self.results["second_order"] = self.statistics[1].evaluate(
                    estimates["second_order"]
                )

    @classmethod
    def _choose_helpers(cls, calculate_third_order):
        """Choose helper objects.

        Choose helper objects for sampling and Sobol index estimating depending on whether we have
        a normal run or a third-order run.

        Returns:
            sampler (type): class type for sampling
            estimator (type): class type for Sobol index estimation
        """
        if calculate_third_order:
            sampler = ThirdOrderSampler
            estimator = SobolIndexEstimatorThirdOrder
        else:
            sampler = Sampler
            estimator = SobolIndexEstimator

        return sampler, estimator
