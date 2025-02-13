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
"""Collection of utility functions and classes for PyMC."""

from typing import Union

import numpy as np
import pymc as pm
import pytensor.tensor as pt
from pytensor import Variable

from queens.distributions import beta, exponential, lognormal, mean_field_normal, normal, uniform


class PymcDistributionWrapper(pt.Op):
    """Op class for Data conversion.

    This PymcDistributionWrapper  class is a wrapper for PyMC Distributions in QUEENS.

    Attributes:
        logpdf (fun): The log-pdf function
        logpdf_gradients (fun): The function to evaluate the gradient of the log-pdf
        logpdf_grad (obj): Wrapper for the gradient function of the log-pdf
    """

    itypes = [pt.dmatrix]
    otypes = [pt.dvector]

    def __init__(self, logpdf, logpdf_gradients=None):
        """Initzialise the wrapper for the functions.

        Args:
        logpdf (fun): The log-pdf function
        logpdf_gradients (fun): The function to evaluate the gradient of the pdf
        """
        self.logpdf = logpdf
        self.logpdf_gradients = logpdf_gradients
        self.logpdf_grad = PymcGradientWrapper(self.logpdf_gradients)

    # pylint: disable-next=unused-argument
    def perform(self, _node, inputs, output_storage, params=None):
        """Call outside pdf function."""
        (sample,) = inputs

        value = self.logpdf(sample)
        output_storage[0][0] = np.array(value)

    def grad(self, inputs, output_grads):
        """Get gradient and multiply with upstream gradient."""
        (sample,) = inputs
        return [output_grads[0] * self.logpdf_grad(sample)]

    def R_op(
        self, inputs: list[Variable], eval_points: Union[Variable, list[Variable]]
    ) -> list[Variable]:
        """Construct a graph for the R-operator.

        This method is primarily used by `Rop`.
        For more information, see pymc documentation for the method.

        Args:
            inputs (list[Variable]): The input variables for the R operator.
            eval_points (Union[Variable, list[Variable]]): Should have the same length as inputs.
                                                           Each element of `eval_points` specifies
                                                           the value of the corresponding input at
                                                           the point where the R-operator is to be
                                                           evaluated.

        Returns:
            list[Variable]
        """
        raise NotImplementedError


class PymcGradientWrapper(pt.Op):
    """Op class for Data conversion.

    This Class is a wrapper for the gradient of the distributions in QUEENS.

    Attributes:
        gradient_func (fun): The function to evaluate the gradient of the pdf
    """

    itypes = [pt.dmatrix]
    otypes = [pt.dmatrix]

    def __init__(self, gradient_func):
        """Initzialise the wrapper for the functions.

        Args:
        gradient_func (fun): The function to evaluate the gradient of the pdf
        """
        self.gradient_func = gradient_func

    def perform(self, _node, inputs, output_storage, _params=None):
        """Evaluate the gradient."""
        (sample,) = inputs
        if self.gradient_func is not None:
            grads = self.gradient_func(sample)
            output_storage[0][0] = grads
        else:
            raise TypeError("Gradient function is not callable")

    def R_op(
        self, inputs: list[Variable], eval_points: Union[Variable, list[Variable]]
    ) -> list[Variable]:
        """Construct a graph for the R-operator.

        This method is primarily used by `Rop`.
        For more information, see pymc documentation for the method.

        Args:
            inputs (list[Variable]): The input variables for the R operator.
            eval_points (Union[Variable, list[Variable]]): Should have the same length as inputs.
                                                           Each element of `eval_points` specifies
                                                           the value of the corresponding input at
                                                           the point where the R-operator is to be
                                                           evaluated.

        Returns:
            list[Variable]
        """
        raise NotImplementedError


def from_config_create_pymc_distribution_dict(parameters, explicit_shape):
    """Get random variables in pymc distribution format.

    Args:
        parameters (obj): Parameters object

        explicit_shape (int): Explicit shape parameter for distribution dimension

    Returns:
        pymc distribution list
    """
    pymc_distribution_list = []

    # loop over random_variables and create list
    for name, distribution in zip(parameters.names, parameters.to_distribution_list()):
        pymc_distribution_list.append(
            from_config_create_pymc_distribution(distribution, name, explicit_shape)
        )
    # Pass the distribution list as arguments
    return pymc_distribution_list


def from_config_create_pymc_distribution(distribution, name, explicit_shape):
    """Create PyMC distribution object from queens distribution.

    Args:
        distribution (obj): Queens distribution object

        name (str): name of random variable
        explicit_shape (int): Explicit shape parameter for distribution dimension

    Returns:
        random_variable:     Random variable, distribution object in pymc format
    """
    shape = (explicit_shape, distribution.dimension)
    if isinstance(distribution, normal.Normal):
        random_variable = pm.MvNormal(
            name,
            mu=distribution.mean,
            cov=distribution.covariance,
            shape=shape,
        )
    elif isinstance(distribution, mean_field_normal.MeanFieldNormal):
        random_variable = pm.Normal(
            name,
            mu=distribution.mean,
            sigma=distribution.covariance,
            shape=shape,
        )

    elif isinstance(distribution, uniform.Uniform):
        if np.all(distribution.lower_bound == 0):
            random_variable = pm.Uniform(
                name,
                lower=0,
                upper=distribution.upper_bound,
                shape=shape,
            )

        elif np.all(distribution.upper_bound == 0):
            random_variable = pm.Uniform(
                name,
                lower=distribution.lower_bound,
                upper=0,
                shape=shape,
            )
        else:
            random_variable = pm.Uniform(
                name,
                lower=distribution.lower_bound,
                upper=distribution.upper_bound,
                shape=shape,
            )
    elif isinstance(distribution, lognormal.LogNormal):
        if distribution.covariance.size == 1:
            std = distribution.covariance[0, 0] ** (1 / 2)
        else:
            raise NotImplementedError("Only 1D lognormals supported")

        random_variable = pm.LogNormal(
            name,
            mu=distribution.mean,
            sigma=std,
            shape=shape,
        )
    elif isinstance(distribution, exponential.Exponential):
        random_variable = pm.Exponential(
            name,
            lam=distribution.rate,
            shape=shape,
        )
    elif isinstance(distribution, beta.Beta):
        random_variable = pm.Beta(
            name,
            alpha=distribution.a,
            beta=distribution.b,
            shape=shape,
        )
    else:
        raise NotImplementedError("Not supported distriubtion by QUEENS and/or PyMC")
    return random_variable
