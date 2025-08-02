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
"""Stochastic optimizer."""

import abc
import logging

import numpy as np

from queens.utils.iterative_averaging import l1_norm, l2_norm, relative_change
from queens.utils.printing import get_str_table
from queens.utils.valid_options import get_option

_logger = logging.getLogger(__name__)


class StochasticOptimizer(metaclass=abc.ABCMeta):
    """Base class for stochastic optimizers.

    The optimizers are implemented as generators. This increases the modularity of this class,
    since an object can be used in different settings. Some examples:

    - **Example 1:** Simple optimization run (does not strongly benefit from its generator nature):
        1. Define a gradient function *gradient()*
        2. Create a optimizer object *optimizer* with the gradient function *gradient*
        3. Run the optimization by *optimizer.run_optimization()* in your script

    - **Example 2:** Adding additional functionality during the optimization
        1. Define a optimizer object using a gradient function.
        2. Example code snippet:

            for parameters in optimizer:

                rel_l2_change_params=optimizer.rel_l2_change

                iteration=optimizer.iteration

                # Verbose output

                print(f"Iter {iteration}, parameters {parameters}, rel L2 change "

                f"{rel_l2_change:.2f}")

                # Some additional condition to stop optimization

                if self.number_of_simulations >= 1000:

                    break

    - **Example 3:** Running multiple optimizer iteratively sequentially:
        1. Define *optimizer1* and *optimizer2* with different gradient functions
        2. Example code:

            while not done_bool:

                if not optimizer1.done:

                    self.parameters1=next(optimizer1)

                if not optimizer2.done:

                    self.parameters2=next(optimizer2)

                # Example on how to reduce the learning rate for optimizer2

                if optimizer2.iteration % 1000 == 0:

                    optimizer2.learning_rate *= 0.5

                done_bool = optimizer1.done and optimizer2.done

    Attributes:
        learning_rate (float): Learning rate for the optimizer.
        clip_by_l2_norm_threshold (float): Threshold to clip the gradient by L2-norm.
        clip_by_value_threshold (float): Threshold to clip the gradient components.
        max_iteration (int): Maximum number of iterations.
        precoefficient (int): Is 1 in case of maximization and -1 for minimization.
        rel_l1_change_threshold (float): If the L1 relative change in parameters falls below this
                                         value, this criterion catches.
        rel_l2_change_threshold (float): If the L2 relative change in parameters falls below this
                                         value, this criterion catches.
        iteration (int): Number of iterations done in the optimization so far.
        done (bool): True if the optimization is done.
        rel_l2_change (float): Relative change in L2-norm of variational params w.r.t. the previous
                              iteration.
        rel_l1_change (float): Relative change in L1-norm of variational params w.r.t. the previous
                              iteration.
        current_variational_parameters (np.array): Variational parameters.
        current_gradient_value (np.array): Current gradient vector w.r.t. the variational
                                           parameters.
        gradient (function): Function to compute the gradient.
        learning_rate_decay (LearningRateDecay): Object to schedule learning rate decay
    """

    _name = "Stochastic Optimizer"

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
        """Initialize stochastic optimizer.

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
        valid_options = {"min": -1, "max": 1}
        self.precoefficient = get_option(
            valid_options, optimization_type, error_message="Unknown optimization type."
        )
        self.learning_rate = learning_rate
        self.clip_by_l2_norm_threshold = clip_by_l2_norm_threshold
        self.clip_by_value_threshold = clip_by_value_threshold
        self.max_iteration = max_iteration
        self.rel_l2_change_threshold = rel_l2_change_threshold
        self.rel_l1_change_threshold = rel_l1_change_threshold
        self.iteration = 0
        self.done = False
        self.rel_l2_change = None
        self.rel_l1_change = None
        self.current_variational_parameters = None
        self.current_gradient_value = None
        self.gradient = None
        self.learning_rate_decay = learning_rate_decay

    @abc.abstractmethod
    def scheme_specific_gradient(self, gradient):
        """Scheme specific gradient computation.

        Here the gradient is transformed according to the desired stochastic optimization approach.

        Args:
            gradient (np.array): Current gradient
        """

    def set_gradient_function(self, gradient_function):
        """Set the gradient function.

        The *gradient_function* has to be a function of the parameters and returns
        the gradient value.

        Args:
            gradient_function (function): Gradient function.
        """
        self.gradient = gradient_function

    def _compute_rel_change(self, old_parameters, new_parameters):
        """Compute L1 and L2 based relative changes of variational parameters.

        Args:
            old_parameters (np.array): Old parameters
            new_parameters (np.array): New parameters
        """
        self.rel_l2_change = relative_change(
            old_parameters, new_parameters, lambda x: l2_norm(x, averaged=True)
        )
        self.rel_l1_change = relative_change(
            old_parameters, new_parameters, lambda x: l1_norm(x, averaged=True)
        )

    def do_single_iteration(self, gradient):
        r"""Single iteration for a given gradient.

        Iteration step for a given gradient :math:`g`:
            :math:`p^{(i+1)}=p^{(i)}+\beta \alpha g`
        where :math:`\beta=-1` for minimization and +1 for maximization and :math:`\alpha` is
        the learning rate.

        Args:
            gradient (np.array): Current gradient
        """
        self.current_variational_parameters = (
            self.current_variational_parameters
            + self.precoefficient * self.learning_rate * gradient
        )
        self.iteration += 1

    def clip_gradient(self, gradient):
        """Clip the gradient by value and then by norm.

        Args:
            gradient (np.array): Current gradient

        Returns:
            gradient (np.array): The clipped gradient
        """
        gradient = clip_by_value(gradient, self.clip_by_value_threshold)
        gradient = clip_by_l2_norm(gradient, self.clip_by_l2_norm_threshold)
        return gradient

    def __next__(self):
        """Python intern function to make this object a generator.

        Essentially this is a single iteration of the stochastic optimizer consiting of:

            1. Computing the noisy gradient
            2. Clipping the gradient
            3. Transform the gradient using the scheme specific approach
            4. Update the parameters
            5. Compute relative changes
            6. Check if optimization is done

        Returns:
            current_variational_parameters (np.array): current variational parameters of the
            optimization
        """
        if self.done:
            raise StopIteration
        old_parameters = self.current_variational_parameters.copy()
        current_gradient = self.gradient(self.current_variational_parameters)
        if self.learning_rate_decay:
            self.learning_rate = self.learning_rate_decay(
                self.learning_rate,
                self.current_variational_parameters,
                current_gradient,
            )
        current_gradient = self.clip_gradient(current_gradient)
        current_gradient = self.scheme_specific_gradient(current_gradient)
        self.current_gradient_value = current_gradient
        self.do_single_iteration(current_gradient)
        self._compute_rel_change(old_parameters, self.current_variational_parameters)
        self._check_if_done()
        return self.current_variational_parameters

    def __iter__(self):
        """Python intern function needed to make this object iterable.

        Hence it can be called as:
            for p in optimizer:
                print(f"Current parameters: {p}")

        Returns:
            self
        """
        return self

    def _check_if_done(self):
        """Check if optimization is done based on L1 and L2 norms.

        Criteria are based on the change of the variational parameters.
        """
        if np.any(np.isnan(self.current_variational_parameters)):
            raise ValueError("At least one of the variational parameters is NaN")
        self.done = (
            self.rel_l2_change <= self.rel_l2_change_threshold
            and self.rel_l1_change <= self.rel_l1_change_threshold
        ) or self.iteration >= self.max_iteration

    def run_optimization(self):
        """Run the optimization.

        Returns:
            np.array: variational parameters
        """
        for _ in self:
            pass
        return self.current_variational_parameters

    def _get_print_dict(self):
        """Get print dict.

        Returns:
            dict: dictionary with data to print
        """
        if self.precoefficient == 1:
            optimization_type = "maximization"
        else:
            optimization_type = "minimization"

        print_dict = {
            "Optimization type": optimization_type,
            "Learning_rate": self.learning_rate,
            "Iterations": self.iteration,
            "Max. number of iteration": self.max_iteration,
            "Rel. L1 change of the parameters": self.rel_l1_change,
            "Rel. L1 change threshold": self.rel_l1_change_threshold,
            "Rel. L2 change of the parameters": self.rel_l2_change,
            "Rel. L2 change threshold": self.rel_l2_change_threshold,
        }
        return print_dict

    def __str__(self):
        """String of optimizer.

        Returns:
            str: String version of the optimizer
        """
        print_dict = self._get_print_dict()
        return get_str_table(self._name, print_dict)


def clip_by_l2_norm(gradient, l2_norm_threshold=1e6):
    """Clip gradients by L2-norm.

    Args:
        gradient (np.array): Gradient
        l2_norm_threshold (float): Clipping threshold
    Returns:
        gradient (np.array): Clipped gradients
    """
    gradient = np.nan_to_num(gradient)
    gradient_l2_norm = l2_norm(gradient)
    if gradient_l2_norm > l2_norm_threshold:
        gradient /= gradient_l2_norm / l2_norm_threshold
        _logger.warning("Gradient clipped due to large norm!")
    return gradient


def clip_by_value(gradient, threshold=1e6):
    """Clip gradients by value.

    Clips if the absolute value op the component is larger than the threshold.

    Args:
        gradient (np.array): Gradient
        threshold (float): Threshold to clip
    Returns:
        gradient (np.array): Clipped gradients
    """
    gradient = np.nan_to_num(gradient)
    gradient = np.clip(gradient, -threshold, threshold)
    return gradient
