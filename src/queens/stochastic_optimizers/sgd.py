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
"""SGD optimizer."""

import logging

import numpy as np

from queens.stochastic_optimizers._stochastic_optimizer import StochasticOptimizer

_logger = logging.getLogger(__name__)


class SGD(StochasticOptimizer):
    """Stochastic gradient descent optimizer."""

    _name = "SGD Stochastic Optimizer"

    def __init__(
        self,
        learning_rate,
        optimization_type,
        rel_l1_change_threshold,
        rel_l2_change_threshold,
        clip_by_l2_norm_threshold=np.inf,
        clip_by_value_threshold=np.inf,
        max_iteration=1e6,
        learning_rate_decay=None,
    ):
        """Initialize optimizer.

        Args:
            learning_rate (float): Learning rate for the optimizer
            optimization_type (str): "max" in case of maximization and "min" for minimization
            rel_l1_change_threshold (float): If the L1 relative change in parameters falls below
                                             this value, this criterion catches.
            rel_l2_change_threshold (float): If the L2 relative change in parameters falls below
                                             this value, this criterion catches.
            clip_by_l2_norm_threshold (float): Threshold to clip the gradient by L2-norm
            clip_by_value_threshold (float): Threshold to clip the gradient components
            max_iteration (int): Maximum number of iterations
            learning_rate_decay (LearningRateDecay): Object to schedule learning rate decay
        """
        # pylint: disable=duplicate-code
        super().__init__(
            learning_rate=learning_rate,
            optimization_type=optimization_type,
            rel_l1_change_threshold=rel_l1_change_threshold,
            rel_l2_change_threshold=rel_l2_change_threshold,
            clip_by_l2_norm_threshold=clip_by_l2_norm_threshold,
            clip_by_value_threshold=clip_by_value_threshold,
            max_iteration=max_iteration,
            learning_rate_decay=learning_rate_decay,
        )

    def scheme_specific_gradient(self, gradient):
        """SGD gradient computation.

        Args:
            gradient (np.array): Gradient

        Returns:
            gradient (np.array): SGD gradient
        """
        return gradient
