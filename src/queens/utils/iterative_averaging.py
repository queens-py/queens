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
from typing import Callable, TypeAlias

import numpy as np

from queens.utils.logger_settings import log_init_args
from queens.utils.printing import get_str_table

NumericalValue: TypeAlias = np.ndarray | np.floating | int | float
NumpyValue: TypeAlias = np.ndarray | np.floating


class IterativeAveraging(metaclass=abc.ABCMeta):
    """Base class for iterative averaging schemes.

    Attributes:
        current_average: Current average value.
        new_value: New value for the averaging process.
        rel_l1_change: Relative change in L1 norm of the average value.
        rel_l2_change: Relative change in L2 norm of the average value.
    """

    _name = "Iterative Averaging"

    def __init__(self) -> None:
        """Initialize iterative averaging."""
        self.current_average: NumpyValue | None = None
        self.new_value: NumericalValue | None = None
        self.rel_l1_change: float | np.generic = 1
        self.rel_l2_change: float | np.generic = 1

    def update_average(self, new_value: NumericalValue) -> NumpyValue:
        """Compute the actual average.

        Args:
            new_value: New observation for the averaging

        Returns:
            Current average value
        """
        if isinstance(new_value, (float, int)):
            new_value = np.array(new_value)
        if self.current_average is not None:
            old_average: NumericalValue
            if isinstance(self.current_average, (np.floating, np.ndarray)):
                old_average = self.current_average.copy()
            else:
                old_average = self.current_average
            self.current_average = self.average_computation(new_value)
            self.rel_l2_change = relative_change(old_average, self.current_average, l2_norm)
            self.rel_l1_change = relative_change(old_average, self.current_average, l1_norm)
        else:
            # If it is the first observation
            self.current_average = new_value.copy()
        return self.current_average.copy()

    @abc.abstractmethod
    def average_computation(self, new_value: NumericalValue) -> NumpyValue:
        """Here, the averaging approach is implemented."""

    def _get_print_dict(self) -> dict:
        """Get print dict.

        Returns:
            Dictionary with data to print
        """
        print_dict = {
            "Rel. L1 change to previous average": self.rel_l1_change,
            "Rel. L2 change to previous average": self.rel_l2_change,
            "Current average": self.current_average,
        }
        return print_dict

    def __str__(self) -> str:
        """String of iterative averager.

        Returns:
            Table of the averager
        """
        print_dict = self._get_print_dict()
        return get_str_table(self._name, print_dict)


class MovingAveraging(IterativeAveraging):
    r"""Moving averages.

    :math:`x^{(j)}_{avg}=\frac{1}{k}\sum_{i=0}^{k-1}x^{(j-i)}`

    where :math:`k-1` is the number of values from previous iterations that are used

    Attributes:
        num_iter_for_avg: Number of samples in the averaging window
        data: Data used to compute the average
    """

    _name = "Moving Averaging"

    @log_init_args
    def __init__(self, num_iter_for_avg: int) -> None:
        """Initialize moving averaging object.

        Args:
            num_iter_for_avg: Number of samples in the averaging window
        """
        super().__init__()
        self.num_iter_for_avg: int = num_iter_for_avg
        self.data: list = []

    def average_computation(self, new_value: NumericalValue) -> NumpyValue:
        """Compute the moving average.

        Args:
            new_value: New value to update the average

        Returns:
            The current average
        """
        if isinstance(new_value, (np.floating, np.ndarray)):
            new_value = new_value.copy()
        self.data.append(new_value)
        if len(self.data) > self.num_iter_for_avg:
            self.data = self.data[-self.num_iter_for_avg :]
        average = np.zeros_like(new_value)
        for data in self.data:
            average += data
        return average / len(self.data)

    def _get_print_dict(self) -> dict:
        """Get print dict.

        Returns:
            Dictionary with data to print
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
    def __init__(self) -> None:
        """Initialize Polyak averaging object."""
        super().__init__()
        self.iteration_counter: int = 1
        self.sum_over_iter: NumpyValue = np.float64(0.0)

    def average_computation(self, new_value: NumericalValue) -> NumpyValue:
        """Compute the Polyak average.

        Args:
            new_value: New value to update the average

        Returns:
            Returns the current average
        """
        if isinstance(new_value, (np.ndarray)) and not isinstance(self.sum_over_iter, np.ndarray):
            self.sum_over_iter = np.zeros_like(new_value)

        if isinstance(new_value, (np.ndarray)):
            self.sum_over_iter += new_value
        else:
            self.sum_over_iter += np.float64(np.asarray(new_value))
        self.iteration_counter += 1
        current_average = self.sum_over_iter / self.iteration_counter

        return current_average

    def _get_print_dict(self) -> dict:
        """Get print dict.

        Returns:
            Dictionary with data to print
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
        coefficient: Coefficient in (0,1) for the average.
    """

    _name = "Exponential Averaging"

    @log_init_args
    def __init__(self, coefficient: float):
        """Initialize exponential averaging object.

        Args:
            coefficient: Coefficient in (0,1) for the average
        """
        if coefficient < 0 or coefficient > 1:
            raise ValueError("Coefficient for exponential averaging needs to be in (0,1)")
        super().__init__()
        self.coefficient: float = coefficient

    def average_computation(  # type: ignore[override]
        self, new_value: NumericalValue
    ) -> NumericalValue:
        """Compute the exponential average.

        Args:
            new_value: New value to update the average.

        Returns:
            Returns the current average
        """
        if self.current_average is None:
            raise ValueError("Current average has not been initialized.")

        current_average = (
            self.coefficient * self.current_average + (1 - self.coefficient) * new_value
        )
        return current_average

    def _get_print_dict(self) -> dict:
        """Get print dict.

        Returns:
            Dictionary with data to print
        """
        print_dict = super()._get_print_dict()
        print_dict.update({"Coefficient": self.coefficient})

        return print_dict


def l1_norm(vector: NumericalValue, averaged: bool = False) -> float | np.floating:
    """Compute the L1 norm of the vector.

    Args:
        vector: Vector
        averaged: If enabled, the norm is divided by the number of components

    Returns:
        L1 norm of the vector
    """
    vector = np.array(vector).flatten()
    vector = np.nan_to_num(vector)
    norm = np.sum(np.abs(vector))
    if averaged:
        norm /= len(vector)
    return norm


def l2_norm(vector: NumericalValue, averaged: bool = False) -> float | np.floating:
    """Compute the L2 norm of the vector.

    Args:
        vector: Vector
        averaged: If enabled the norm is divided by the square root of the number of components

    Returns:
        L2 norm of the vector
    """
    vector = np.array(vector).flatten()
    vector = np.nan_to_num(vector)
    norm = np.sum(vector**2) ** 0.5
    if averaged:
        norm /= len(vector) ** 0.5
    return norm


def relative_change(
    old_value: NumericalValue, new_value: NumericalValue, norm: Callable
) -> float | np.floating:
    """Compute the relative change of the old and new value for a given norm.

    Args:
        old_value: Old values
        new_value: New values
        norm: Function to compute a norm

    Returns:
        Relative change
    """
    increment = old_value - new_value
    increment = np.nan_to_num(increment)
    return norm(increment) / (norm(old_value) + 1e-16)
