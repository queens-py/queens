#
# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (c) 2024-2025, QUEENS contributors.
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
"""Helper classes for Gaussian process prediction for Sobol indices."""

import logging
import time

import numpy as np
import xarray as xr

from queens.utils.logger_settings import log_init_args

_logger = logging.getLogger(__name__)


class Predictor:
    """Predictor class.

    Predict the output of the Gaussian process used as a surrogate model at all Monte-Carlo samples
    stored in the sample matrices (A, B, AB, BA,...). Predicting either mean sampling realizations
    of the Gaussian process or using the posterior mean of the Gaussian process (if the number of
    GP realizations is set to 1).

    Attributes:
        number_gp_realizations (int): number of Gaussian process realizations
        gp_model (Model): Gaussian process model
        seed_posterior_samples (int): seed for posterior samples
    """

    @log_init_args
    def __init__(self, gp_model, number_gp_realizations, seed_posterior_samples):
        """Initialize.

        Args:
            gp_model (Model): Gaussian process model
            number_gp_realizations (int): number of Gaussian process realizations
            seed_posterior_samples (int): seed for posterior samples
        """
        self.gp_model = gp_model
        self.number_gp_realizations = number_gp_realizations
        self.seed_posterior_samples = seed_posterior_samples

    @classmethod
    def from_config_create(cls, method_options, gp_model):
        """Create estimator from problem description.

        Args:
            method_options (dict): dictionary with method options
            gp_model (Model): Gaussian process model

        Returns:
            estimator: SobolIndexEstimator
        """
        number_gp_realizations = method_options["number_gp_realizations"]
        seed_posterior_samples = method_options.get("seed_posterior_samples", None)
        if number_gp_realizations == 1:
            _logger.info("Number of realizations = 1. Prediction is based on GP mean.")
        else:
            _logger.info("Number of realizations = %i", number_gp_realizations)
        return cls(
            gp_model=gp_model,
            number_gp_realizations=number_gp_realizations,
            seed_posterior_samples=seed_posterior_samples,
        )

    def predict(self, samples):
        """Predict output at Monte-Carlo samples.

        Sample realizations of Gaussian process or use the posterior mean of the Gaussian process.

        Args:
            samples (xr.Array): Monte-Carlo samples

        Returns:
            prediction (xr.Array): predictions
        """
        start_prediction = time.time()

        prediction = self._init_prediction(samples)

        inputs = np.array(samples).reshape(-1, samples.shape[-1])
        gp_output = self.gp_model.predict(inputs, support="f")

        if self.number_gp_realizations == 1:
            raw_prediction = gp_output["result"].reshape(*samples.shape[:2], 1)
        else:
            if self.seed_posterior_samples:
                np.random.seed(self.seed_posterior_samples)
            raw_prediction = (
                gp_output["result"]
                + np.random.randn(inputs.shape[0], self.number_gp_realizations)
                * np.sqrt(gp_output["variance"])
            ).reshape(*samples.shape[:2], self.number_gp_realizations)

        prediction.data = np.array(raw_prediction)

        _logger.info("Time for prediction: %f", time.time() - start_prediction)
        _logger.debug("Prediction %s", prediction.values)
        return prediction

    def _init_prediction(self, samples):
        """Initialize prediction data-array.

        Args:
            samples (xr.DataArray): Monte-Carlo samples

        Returns:
            prediction (xr.DataArray): predictions
        """
        dimensions = ("monte_carlo", "sample_matrix", "gp_realization")
        coordinates = {
            "monte_carlo": samples.coords["monte_carlo"].values,
            "sample_matrix": samples.coords["sample_matrix"].values,
            "gp_realization": np.arange(self.number_gp_realizations),
        }

        prediction = xr.DataArray(
            data=np.empty(
                (
                    samples.coords["monte_carlo"].size,
                    samples.coords["sample_matrix"].size,
                    self.number_gp_realizations,
                )
            ),
            dims=dimensions,
            coords=coordinates,
        )

        return prediction
