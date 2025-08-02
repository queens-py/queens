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
"""Learning rate decay for stochastic optimization."""

import abc
import logging

import numpy as np

_logger = logging.getLogger(__name__)


class LearningRateDecay(metaclass=abc.ABCMeta):
    """Base class for learning rate decay."""

    @abc.abstractmethod
    def __call__(self, learning_rate, params, gradient):
        """Adapt learning rate.

        Args:
            learning_rate (float): Current learning rate
            params (np.array): Current parameters
            gradient (np.array): Current gradient

        Returns:
            learning_rate (float): Adapted learning rate
        """


class LogLinearLearningRateDecay(LearningRateDecay):
    """Log linear learning rate decay.

    Attributes:
        slope (float): Logarithmic slope
        iteration (int): Current iteration
    """

    def __init__(self, slope):
        """Initialize LogLinearLearningRateDecay.

        Args:
            slope (float): Logarithmic slope
        """
        self.slope = slope
        self.iteration = 0

    def __call__(self, learning_rate, params, gradient):
        """Adapt learning rate.

        Args:
            learning_rate (float): Current learning rate
            params (np.array): Current parameters
            gradient (np.array): Current gradient

        Returns:
            learning_rate (float): Adapted learning rate
        """
        if self.iteration > 1:
            learning_rate *= (self.iteration - 1) ** self.slope / self.iteration**self.slope
        self.iteration += 1
        return learning_rate


class StepwiseLearningRateDecay(LearningRateDecay):
    """Step-wise learning rate decay.

    Attributes:
        decay_factor (float): Decay factor
        decay_interval (int): Decay interval
        iteration (int): Iteration number
    """

    def __init__(self, decay_factor, decay_interval):
        """Initialize StepwiseLearningRateDecay.

        Args:
            decay_factor (float): Decay factor
            decay_interval (int): Decay interval
        """
        self.decay_factor = decay_factor
        self.decay_interval = decay_interval
        self.iteration = 0

    def __call__(self, learning_rate, params, gradient):
        """Adapt learning rate.

        Args:
            learning_rate (float): Current learning rate
            params (np.array): Current parameters
            gradient (np.array): Current gradient

        Returns:
            learning_rate (float): Adapted learning rate
        """
        if self.iteration >= self.decay_interval:
            learning_rate *= self.decay_factor
            self.iteration = 0
            _logger.info("learning_rate=%.2e", learning_rate)
        else:
            self.iteration += 1
        return learning_rate


class DynamicLearningRateDecay(LearningRateDecay):
    """Dynamic learning rate decay.

    Attributes:
        alpha (float): Decay factor
        rho_min (float): Threshold for signal-to-noise ratio
        k_min (int): Minimum number of iterations before learning rate is decreased
        k (int): Iteration number
        a (np.array): Sum of parameters
        b (np.array): Sum of squared parameters
        c (np.array): Sum of parameters times iteration number
    """

    def __init__(self, alpha=0.1, rho_min=1.0):
        """Initialize DynamicLearningRateDecay.

        Args:
            alpha (float): Decay factor
            rho_min (float): Threshold for signal-to-noise ratio
        """
        if alpha >= 1.0 or alpha <= 0.0:
            raise ValueError("alpha must be between 0 and 1.")
        if rho_min <= 0.0:
            raise ValueError("rho_min must be greater than 0.")
        self.alpha = alpha
        self.rho_min = rho_min
        self.k_min = 2
        self._reset()

    def __call__(self, learning_rate, params, gradient):
        """Adapt learning rate.

        Args:
            learning_rate (float): Current learning rate
            params (np.array): Current parameters
            gradient (np.array): Current gradient

        Returns:
            learning_rate (float): Adapted learning rate
        """
        self.k += 1
        self.a += params
        self.b += params**2
        self.c += self.k * params

        if self.k >= self.k_min:
            rho = 1 / (
                (self.k * (self.k + 1) * (self.k + 2) / 12)
                * (self.b - self.a**2 / (self.k + 1))
                / (self.c - self.k / 2 * self.a) ** 2
                - 1
            )
            rho_mean = np.mean(rho)
            if rho_mean < self.rho_min:
                learning_rate *= self.alpha
                _logger.info(
                    "Criterion reached after %i iterations: learning_rate=%.2e",
                    self.k,
                    learning_rate,
                )
                self.k_min = self.k
                self._reset()

        return learning_rate

    def _reset(self):
        """Reset regression parameters."""
        self.k = -1
        self.a = 0
        self.b = 0
        self.c = 0
