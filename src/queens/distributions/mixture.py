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
"""Mixture distribution."""

import logging
from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import logsumexp

from queens.distributions._distribution import Continuous
from queens.utils.logger_settings import log_init_args

_logger = logging.getLogger(__name__)


class Mixture(Continuous):
    """Mixture models."""

    @log_init_args
    def __init__(self, weights: ArrayLike, component_distributions: Sequence[Continuous]) -> None:
        """Initialize mixture model.

        Args:
            weights: Weights of the mixtures
            component_distributions: Component distributions of the mixture
        """
        self.weights = np.array(weights)
        self.component_distributions = list(component_distributions)
        self.number_of_components = len(self.weights)

        if len(self.component_distributions) != len(self.weights):
            raise ValueError(
                f"The number of weights {len(self.weights)} does not match the number of "
                f"distributions {len(self.component_distributions)}"
            )

        super().check_positivity(weights=self.weights)

        if np.sum(self.weights) != 1:
            _logger.info("Weights do not sum up to one, they are going to be normalized.")
            self.weights /= np.sum(self.weights)

        if len({d.dimension for d in self.component_distributions}) != 1:
            raise ValueError("Dimensions of the component distributions do not match")

        mean, covariance = self._compute_mean_and_covariance(
            self.weights, self.component_distributions
        )
        super().__init__(mean, covariance, self.component_distributions[0].dimension)

    @staticmethod
    def _compute_mean_and_covariance(
        weights: np.ndarray, component_distributions: list[Continuous]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute the mean value and covariance of the mixture model.

        Args:
            weights: Weights of the mixture
            component_distributions: Components of the mixture

        Returns:
            Mean value of the mixture
            Covariance of the mixture
        """
        mean = np.zeros_like(component_distributions[0].mean, dtype=float)
        covariance = np.zeros_like(component_distributions[0].covariance, dtype=float)
        for weight, component in zip(weights, component_distributions, strict=True):
            mean += weight * component.mean
            covariance += weight * (component.covariance + np.outer(component.mean, component.mean))
        covariance -= np.outer(mean, mean)
        return mean, covariance

    def draw(self, num_draws: int = 1) -> np.ndarray:
        """Draw *num_draw* samples from the variational distribution.

        Uses a two step process:
            1. From a multinomial distribution, based on the weights, select a component
            2. Sample from the selected component

        Args:
            num_draws: Number of samples to draw

        Returns:
            Row-wise samples of the variational distribution
        """
        components = np.random.multinomial(num_draws, self.weights)
        samples_lst = []
        for component, num_draw_component in enumerate(components):
            sample = self.component_distributions[component].draw(num_draw_component)
            if sample is None:
                raise ValueError(
                    f"Draw method of distribution {self.component_distributions[component]} "
                    f"returned None."
                )
            samples_lst.append(sample)
        samples: np.ndarray = np.concatenate(samples_lst, axis=0)

        # Strictly speaking this is not necessary, however, without it, if you only select x
        # samples, so `samples[:x]`, most samples would originate from the first components and this
        # would be biased
        np.random.shuffle(samples)

        return samples

    def cdf(self, x: np.ndarray) -> np.ndarray:
        """Cumulative distribution function.

        Args:
            x: Positions at which the CDF is evaluated

        Returns:
            CDF of the mixture model
        """
        cdf = np.zeros(
            x.reshape(-1, self.component_distributions[0].dimension).shape[0], dtype=float
        )
        for weights, component in zip(self.weights, self.component_distributions, strict=True):
            cdf += weights * component.cdf(x)
        return cdf

    def logpdf(self, x: np.ndarray) -> np.ndarray:
        """Log of the probability density function.

        Args:
            x: Positions at which the log-PDF is evaluated

        Returns:
            Log-PDF at positions
        """
        log_weights = np.log(self.weights)
        weighted_logpdf = []
        for log_weight, component in zip(log_weights, self.component_distributions, strict=True):
            weighted_logpdf.append(log_weight + component.logpdf(x))

        logpdf = logsumexp(weighted_logpdf, axis=0).flatten()

        return logpdf

    def pdf(self, x: np.ndarray) -> np.ndarray:
        """Probability density function.

        Args:
            x: Positions at which the PDF is evaluated

        Returns:
            PDF at positions
        """
        return np.exp(self.logpdf(x))

    def grad_logpdf(self, x: np.ndarray) -> np.ndarray:
        """Gradient of the log-PDF with respect to *x*.

        Args:
            x: Positions at which the gradient of log-PDF is evaluated

        Returns:
            Gradient of the log-PDF evaluated at positions
        """
        responsibilities = self.responsibilities(x)

        grad_logpdf = 0
        for responsibility, component in zip(
            responsibilities.T, self.component_distributions, strict=True
        ):
            grad_logpdf += responsibility.reshape(-1, 1) * component.grad_logpdf(x)

        return np.array(grad_logpdf).reshape(len(x), -1)

    def ppf(self, quantiles: np.ndarray) -> np.ndarray:
        """Percent point function (inverse of CDF — quantiles).

        Args:
            quantiles: Quantiles at which the PPF is evaluated
        """
        raise NotImplementedError("PPF not available for mixture models.")

    def responsibilities(self, x: np.ndarray) -> np.ndarray:
        r"""Compute the responsibilities.

        The responsibilities are defined as [1]:

        :math: `\gamma_j(x)=\frac{w_j p_j(x)}{\sum_{i=0}^{n_{components}-1}w_i p_i(x)}`

        [1]: Bishop, C. M. (2006). Pattern recognition and machine learning. Springer.

        Args:
            x: Positions at which the responsibilities are evaluated

        Returns:
            Responsibilities (number of samples x number of component)
        """
        log_weights = np.log(self.weights)
        inv_log_responsibility = []
        for log_weight_i, component_i in zip(
            log_weights, self.component_distributions, strict=True
        ):
            data_component_i = []
            for log_weight_j, component_j in zip(
                log_weights, self.component_distributions, strict=True
            ):
                log_ratio = (
                    -log_weight_i - component_i.logpdf(x) + log_weight_j + component_j.logpdf(x)
                )
                data_component_i.append(log_ratio)
            inv_log_responsibility.append(data_component_i)
        inv_log_responsibility = -logsumexp(  # pylint: disable=invalid-unary-operand-type
            inv_log_responsibility, axis=1
        )
        return np.exp(inv_log_responsibility).T

    def export_dict(self) -> dict:
        """Create a dict of the distribution.

        Returns:
            Dictionary containing distribution information
        """
        dictionary = super().export_dict()
        dictionary.pop("component_distributions")
        for i, components in enumerate(self.component_distributions):
            dictionary.update({f"component_{i}": components.export_dict()})
        return dictionary
