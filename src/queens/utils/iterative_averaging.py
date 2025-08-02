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
"""Iterative averaging utils."""

import abc

import numpy as np

from queens.utils.logger_settings import log_init_args
from queens.utils.printing import get_str_table


class IterativeAveraging(metaclass=abc.ABCMeta):
    """Base class for iterative averaging schemes.

    Attributes:
        current_average (np.array): Current average value.
        new_value (np.array): New value for the averaging process.
        rel_l1_change (float): Relative change in L1 norm of the average value.
        rel_l2_change (float): Relative change in L2 norm of the average value.
    """

    _name = "Iterative Averaging"

    def __init__(self):
        """Initialize iterative averaging."""
        self.current_average = None
        self.new_value = None
        self.rel_l1_change = 1
        self.rel_l2_change = 1

    def update_average(self, new_value):
        """Compute the actual average.

        Args:
            new_value (np.array): New observation for the averaging

        Returns:
            Current average value
        """
        if isinstance(new_value, (float, int)):
            new_value = np.array(new_value)
        if self.current_average is not None:
            old_average = self.current_average.copy()
            self.current_average = self.average_computation(new_value)
            self.rel_l2_change = relative_change(old_average, self.current_average, l2_norm)
            self.rel_l1_change = relative_change(old_average, self.current_average, l1_norm)
        else:
            # If it is the first observation
            self.current_average = new_value.copy()
        return self.current_average.copy()

    @abc.abstractmethod
    def average_computation(self, new_value):
        """Here the averaging approach is implemented."""

    def _get_print_dict(self):
        """Get print dict.

        Returns:
            dict: dictionary with data to print
        """
        print_dict = {
            "Rel. L1 change to previous average": self.rel_l1_change,
            "Rel. L2 change to previous average": self.rel_l2_change,
            "Current average": self.current_average,
        }
        return print_dict

    def __str__(self):
        """String of iterative averager.

        Returns:
            str: table of the averager
        """
        print_dict = self._get_print_dict()
        return get_str_table(self._name, print_dict)


class MovingAveraging(IterativeAveraging):
    r"""Moving averages.

    :math:`x^{(j)}_{avg}=\frac{1}{k}\sum_{i=0}^{k-1}x^{(j-i)}`

    where :math:`k-1` is the number of values from previous iterations that are used

    Attributes:
        num_iter_for_avg (int): Number of samples in the averaging window
        data (np.ndarray): data used to compute the average
    """

    _name = "Moving Averaging"

    @log_init_args
    def __init__(self, num_iter_for_avg):
        """Initialize moving averaging object.

        Args:
            num_iter_for_avg (int): Number of samples in the averaging window
        """
        super().__init__()
        self.num_iter_for_avg = num_iter_for_avg
        self.data = []

    def average_computation(self, new_value):
        """Compute the moving average.

        Args:
            new_value (float or np.array): New value to update the average

        Returns:
            average (np.array): The current average
        """
        self.data.append(new_value.copy())
        if len(self.data) > self.num_iter_for_avg:
            self.data = self.data[-self.num_iter_for_avg :]
        average = 0
        for data in self.data:
            average += data
        return average / len(self.data)

    def _get_print_dict(self):
        """Get print dict.

        Returns:
            dict: dictionary with data to print
        """
        print_dict = super()._get_print_dict()
        print_dict.update({"Averaging window size": self.num_iter_for_avg})

        return print_dict


class PolyakAveraging(IterativeAveraging):
    r"""Polyak averaging.

    :math:`x^{(j)}_{avg}=\frac{1}{j}\sum_{i=0}^{j}x^{(j)}`

    Attributes:
        iteration_counter (float): Number of samples.
        sum_over_iter (np.array): Sum over all samples.
    """

    _name = "Polyak Averaging"

    @log_init_args
    def __init__(self):
        """Initialize Polyak averaging object."""
        super().__init__()
        self.iteration_counter = 1
        self.sum_over_iter = 0

    def average_computation(self, new_value):
        """Compute the Polyak average.

        Args:
            new_value (float or np.array): New value to update the average

        Returns:
            current_average (np.array): Returns the current average
        """
        self.sum_over_iter += new_value
        self.iteration_counter += 1
        current_average = self.sum_over_iter / self.iteration_counter

        return current_average

    def _get_print_dict(self):
        """Get print dict.

        Returns:
            dict: dictionary with data to print
        """
        print_dict = super()._get_print_dict()
        print_dict.update({"Number of iterations": self.iteration_counter})

        return print_dict


class ExponentialAveraging(IterativeAveraging):
    r"""Exponential averaging.

    :math:`x^{(0)}_{avg}=x^{(0)}`

    :math:`x^{(j)}_{avg}= \alpha x^{(j-1)}_{avg}+(1-\alpha)x^{(j)}`

    Is also sometimes referred to as exponential smoothing.

    Attributes:
        coefficient (float): Coefficient in (0,1) for the average.
    """

    _name = "Exponential Averaging"

    @log_init_args
    def __init__(self, coefficient):
        """Initialize exponential averaging object.

        Args:
            coefficient (float): Coefficient in (0,1) for the average
        """
        if coefficient < 0 or coefficient > 1:
            raise ValueError("Coefficient for exponential averaging needs to be in (0,1)")
        super().__init__()
        self.coefficient = coefficient

    def average_computation(self, new_value):
        """Compute the exponential average.

        Args:
            new_value (float or np.array): New value to update the average.

        Returns:
            current_average (np.array): Returns the current average
        """
        current_average = (
            self.coefficient * self.current_average + (1 - self.coefficient) * new_value
        )
        return current_average

    def _get_print_dict(self):
        """Get print dict.

        Returns:
            dict: dictionary with data to print
        """
        print_dict = super()._get_print_dict()
        print_dict.update({"Coefficient": self.coefficient})

        return print_dict


def l1_norm(vector, averaged=False):
    """Compute the L1 norm of the vector.

    Args:
        vector (np.array): Vector
        averaged (bool): If enabled, the norm is divided by the number of components

    Returns:
        norm (float): L1 norm of the vector
    """
    vector = np.array(vector).flatten()
    vector = np.nan_to_num(vector)
    norm = np.sum(np.abs(vector))
    if averaged:
        norm /= len(vector)
    return norm


def l2_norm(vector, averaged=False):
    """Compute the L2 norm of the vector.

    Args:
        vector (np.array): Vector
        averaged (bool): If enabled the norm is divided by the square root of the number of
                         components

    Returns:
        norm (float): L2 norm of the vector
    """
    vector = np.array(vector).flatten()
    vector = np.nan_to_num(vector)
    norm = np.sum(vector**2) ** 0.5
    if averaged:
        norm /= len(vector) ** 0.5
    return norm


def relative_change(old_value, new_value, norm):
    """Compute the relative change of the old and new value for a given norm.

    Args:
        old_value (np.array): Old values
        new_value (np.array): New values
        norm (func): Function to compute a norm

    Returns:
        Relative change
    """
    increment = old_value - new_value
    increment = np.nan_to_num(increment)
    return norm(increment) / (norm(old_value) + 1e-16)


VALID_TYPES = {
    "moving_average": MovingAveraging,
    "polyak_averaging": PolyakAveraging,
    "exponential_averaging": ExponentialAveraging,
}
